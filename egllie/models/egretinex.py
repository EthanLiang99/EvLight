import torch
from torch import nn
from egllie.models.base_block.Trans import Unet_ReFormer


class IllumiinationNet(nn.Module): #check
    def __init__(self, cfg):
        super().__init__()

        ### illumiantion maps fusion
        self.ill_extractor = nn.Sequential(
            nn.Conv2d(
                cfg.illumiantion_level+3,
                cfg.illumiantion_level * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(
                cfg.illumiantion_level * 2,
                cfg.base_chs,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.ill_level = cfg.illumiantion_level
        self.illumiantion_set = cfg.illumiantion_set

        
        self.reduce = nn.Sequential(
            nn.Conv2d(cfg.base_chs, 1, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, batch):
        ### selcet illumiantion map
        ill_list = [int(num) for num in self.illumiantion_set]
        inital_ill = torch.cat(
            [batch["ill_list"][i] for i in ill_list], dim=1
        )
        ### predict inital enhacned illumiantion map
        pred_illu_feature = self.ill_extractor(torch.concat((inital_ill,batch['lowligt_image']),dim=1))
        pred_illumaintion = self.reduce(pred_illu_feature)

        return pred_illumaintion,pred_illu_feature


class ImageEnhanceNet(nn.Module): #check

    def __init__(self, cfg):
        super().__init__()
        self.base_chs =cfg.base_chs
        self.snr_factor = float(cfg.snr_factor)

        self.ev_img_align = nn.Conv2d(
                cfg.base_chs*2,
                cfg.base_chs,
                kernel_size=1,
                stride=1,
                padding=0,
            )

        self.ev_extractor = nn.Sequential(
            nn.Conv2d(
                cfg.voxel_grid_channel,
                cfg.base_chs,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.img_extractor = nn.Sequential(
            nn.Conv2d(
                3,
                cfg.base_chs,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )


        self.Unet_ReFormer = Unet_ReFormer(dim=cfg.base_chs,snr_threshold_list=cfg.snr_threshold_list)
    
    def _snr_generate(self,low_img, low_img_blur): 
        """
        generate snr map
        """
        # convert to gray-scale img
        dark = low_img
        dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
        light = low_img_blur
        light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
        noise = torch.abs(dark - light)

        mask = torch.div(light, noise + 0.0001).contiguous()

        batch_size = mask.shape[0]
        height = mask.shape[2]
        width = mask.shape[3]
        mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        mask_max = mask_max.view(batch_size, 1, 1, 1)
        mask_max = mask_max.repeat(1, 1, height, width)

        # we apply snr factor here to increase the density of a snr map
        mask = mask * self.snr_factor / (mask_max + 0.0001) 
        # ensure range 0 ~ 1
        mask = torch.clamp(mask, min=0, max=1.0)
        mask = mask.float()
        return mask


    def forward(self, batch):
        # result of inital enhacnment
        low_light_img = batch["lowligt_image"]
        pred_illumaintion = batch["illumaintion"]
        low_light_img_blur = batch["lowlight_image_blur"]
        enhance_low_img_mid = low_light_img * pred_illumaintion + low_light_img
        enhance_low_img_blur = low_light_img_blur * pred_illumaintion + low_light_img_blur
        
        # obtain snr map 
        snr_lightup = self._snr_generate(enhance_low_img_mid,enhance_low_img_blur)

        batch['snr_lightup'] = snr_lightup
        event_free = batch["event_free"]

        snr_enhance = snr_lightup.detach()
        # encode img + event
        event_free = self.ev_extractor(event_free)
        enhance_low_img = self.img_extractor(enhance_low_img_mid)
        
        img_event = self.ev_img_align(torch.concat((event_free, enhance_low_img),dim=1))
        # holistic and regional operation
        pred_normal_img = self.Unet_ReFormer(img_event,enhance_low_img_mid,enhance_low_img, snr_enhance,event_free)

        return pred_normal_img

class EgLlie(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.IllumiinationNet = IllumiinationNet(cfg.IlluNet)
        self.ImageEnhanceNet = ImageEnhanceNet(cfg.ImageNet)


    def forward(self, batch):
        batch["illumaintion"],batch['illu_feature'] = self.IllumiinationNet(batch)
        output = self.ImageEnhanceNet(batch)

        outputs = {
                'pred':output,
                'gt':batch["normalligt_image"],
        }

        return outputs
    
