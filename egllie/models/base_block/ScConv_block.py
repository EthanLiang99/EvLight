import torch
import torch.nn.functional as F
import torch.nn as nn


class CA_layer(nn.Module):
    def __init__(self, in_ch) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_ch//2, in_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        res = self.conv1(x)
        cross = self.conv2(x)
        res = res * cross
        x = x + res
        return x

class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k_size = k_size
        self.conv = nn.Conv1d(
            channel, channel, kernel_size=k_size, bias=False, groups=channel
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = nn.functional.unfold(
            y.transpose(-1, -3),
            kernel_size=(1, self.k_size),
            padding=(0, (self.k_size - 1) // 2),
        )
        y = self.conv(y.transpose(-1, -2)).unsqueeze(-1)
        y = self.sigmoid(y)
        x = x * y.expand_as(x)
        return x



class ECAResidualBlock(nn.Module):
    def __init__(self, nf):
        super(ECAResidualBlock, self).__init__()
        self.conv1 =  nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.relu = nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.eca = eca_layer(nf)
        self.norm = nn.InstanceNorm2d(nf//2, affine=True)


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out_1, out_2 = torch.chunk(out, 2, dim=1)
        out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.eca(out)
        
        out += residual
        out = self.relu(out)
        
        return out