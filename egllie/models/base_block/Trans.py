import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from egllie.models.base_block.ScConv_block import ECAResidualBlock,CA_layer

class SNR_enhance(nn.Module):
    def __init__(
        self,
        channel,
        snr_threshold,
        depth
    ):
        super().__init__()
        self.channel = channel
        self.depth = depth
        self.img_extractor = nn.ModuleList()
        self.ev_extractor = nn.ModuleList()

        for i in range(self.depth):
                self.img_extractor.append(ECAResidualBlock(self.channel))
                self.ev_extractor.append(ECAResidualBlock(self.channel))

        self.fea_align = nn.Sequential(
            CA_layer(self.channel*3),
            nn.LeakyReLU(negative_slope=0.01,inplace=True),
            nn.Conv2d(self.channel*3,self.channel*1,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope=0.01,inplace=True),
        )

        self.threshold = snr_threshold


    def forward(self, cnn_fea , snr_map, att_fea, event_free):
        """
        cnn_fea: [b,c,h,w] only cnn
        att_fea: [b,c,h,w] with att 
        snr_map: [b,1,h,w]         
        return out: [b,c,h,w]
        """
        # we assign the weight of low snr areas with 0.3 while high snr areas's with 0.7
        snr_map[snr_map <= self.threshold] = 0.3 
        snr_map[snr_map > self.threshold] = 0.7
        snr_reverse_map = 1-snr_map
        snr_map_enlarge = snr_map.repeat(1, self.channel,1,1)
        snr_reverse_map_enlarge = snr_reverse_map.repeat(1, self.channel,1,1)


        for i in range(self.depth):
                cnn_fea = self.img_extractor[i](cnn_fea)
                event_free = self.ev_extractor[i](event_free)
        # select high snr area from img feature
        out_img = torch.mul(cnn_fea,snr_map_enlarge)

        # select low snr area from img feature
        out_ev = torch.mul(event_free,snr_reverse_map_enlarge)


        # feature fusion 
        out = self.fea_align(torch.concat((out_img,out_ev,att_fea),dim=1))

        if self.depth==0:
            return att_fea

        return out

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class IG_MSA(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]         # input_feature   
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            (q_inp, k_inp, v_inp,),
        )
        v = v 
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = k @ q.transpose(-2, -1)  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(
            v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(
                dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult
            ),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class IGAB(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleList(
                    [
                        IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                        PreNorm(dim, FeedForward(dim=dim)),
                    ]
                )
            )

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for attn, ff in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class Unet_ReFormer(nn.Module):
    def __init__(
        self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4], snr_depth_list=[2,4,6], snr_threshold_list=[0.5,0.5,0.5]
    ):
        super(Unet_ReFormer, self).__init__()
        self.dim = dim
        self.level = level
        self.snr_threshold_list = snr_threshold_list
        self.snr_depth_list = snr_depth_list

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(
                nn.ModuleList(
                    [
                        IGAB(
                            dim=dim_level,
                            num_blocks=num_blocks[i],
                            dim_head=dim,
                            heads=dim_level // dim,
                        ),
                        nn.Conv2d(
                            dim_level, dim_level * 2, 4, 2, 1, bias=False
                        ),
                        nn.Conv2d(
                            dim_level, dim_level * 2, 4, 2, 1, bias=False
                        ),
                        nn.Conv2d(
                            dim_level, dim_level * 2, 4, 2, 1, bias=False
                        ),
                    ]
                )
            )
            dim_level *= 2

        # Bottleneck
        self.bottleneck = IGAB(
            dim=dim_level,
            dim_head=dim,
            heads=dim_level // dim,
            num_blocks=num_blocks[-1],
        )
        self.bottleneck_SNR = SNR_enhance(dim_level,snr_threshold_list[-1],snr_depth_list[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(
                            dim_level,
                            dim_level // 2,
                            stride=2,
                            kernel_size=2,
                            padding=0,
                            output_padding=0,
                        ),
                        nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                        IGAB(
                            dim=dim_level // 2,
                            num_blocks=num_blocks[level - 1 - i],
                            dim_head=dim,
                            heads=(dim_level // 2) // dim,
                        ),
                        SNR_enhance(dim_level // 2,snr_threshold_list[level - 1 - i],snr_depth_list[level-1-i])
                    ]
                )
            )
            dim_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        


    def forward(self, event_img, enhance_low_img, fea_img, SNR, event_free):
        """
        event_img:     [b,c,h,w]
        enhance_feat:  [b,c,h,w] 
        SNR:           [b,1,h,w]
        
        return out: [b,c,h,w]
        """
        
        # Embedding
        fea_event_img = event_img
        fea_img = fea_img

        # Encoder
        fea_event_img_encoder = []
        SNRDownsample_list = []
        fea_img_list = []
        event_free_list = []
        for IGAB, FeaDownSample, ImgDownSample,EvDownSample in self.encoder_layers:
            fea_event_img = IGAB(fea_event_img)  # bchw
            fea_event_img_encoder.append(fea_event_img)
            event_free_list.append(event_free)
            SNRDownsample_list.append(SNR)
            fea_img_list.append(fea_img)
            SNR = self.avg_pool(SNR)
            
            fea_event_img = FeaDownSample(fea_event_img)

            fea_img = ImgDownSample(fea_img)
            event_free = EvDownSample(event_free)

        # Bottleneck
        fea_event_img = self.bottleneck_SNR(fea_img, SNR, fea_event_img,event_free)
        fea_event_img = self.bottleneck(fea_event_img)
        
        # Decoder
        for i, (FeaUpSample, Fusion, REIGAB, RESNR_enhance) in enumerate(
            self.decoder_layers
        ):
            fea_event_img = FeaUpSample(fea_event_img)
            fea_event_img = Fusion(
                torch.cat([fea_event_img, fea_event_img_encoder[self.level - 1 - i]], dim=1)
            )
            SNR = SNRDownsample_list[self.level - 1 - i]
            fea_img = fea_img_list[self.level - 1 - i]
            event_free = event_free_list[self.level - 1 - i]
            fea_event_img = RESNR_enhance(fea_img, SNR, fea_event_img,event_free)
            fea_event_img = REIGAB(fea_event_img)
            

        # Mapping 
        out = self.mapping(fea_event_img)+enhance_low_img    

        return out
        