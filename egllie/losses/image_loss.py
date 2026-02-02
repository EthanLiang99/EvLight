from logging import info

import torch
from torch.nn.modules.loss import _Loss

import torch.nn.functional as F
import lpips
from torch.autograd import Variable
import torch.nn as nn
from math import exp



class L1CharbonnierLoss(_Loss):
    def __init__(self):
        super(L1CharbonnierLoss, self).__init__()
        self.eps = 1e-4
        self.scale = 1.0

    def forward(self, batch):
        x, y = batch["pred"], batch["gt"]
        diff = torch.add(x, -y)
        diff_sq = diff * diff
        diff_sq = torch.mean(diff_sq, 1, True)
        error = torch.sqrt(diff_sq + self.eps)
        loss = torch.mean(error)
        return loss*self.scale
    


class PerceptualLoss(_Loss):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        device = torch.device("cuda")
        # alex net by default
        self.lpips_loss = lpips.LPIPS().to(device)

    def forward(self, batch):
        pred, gt = batch["pred"], batch["gt"]
        loss = self.lpips_loss(pred, gt, normalize=True).mean()
        return loss


class frame_temporal_loss(_Loss):
    def __init__(self):
        super(frame_temporal_loss, self).__init__()
        self.eps = 1e-4

    def forward(self, batch_list):
        # stack the batch
        x = torch.stack([b["pred"] for b in batch_list], dim=1)
        y = torch.stack([b["gt"] for b in batch_list], dim=1)
        # event = torch.stack([b["event"] for b in batch_list], dim=1)
        pred_diff = x[:, 1:, :, :,:] - x[:, :-1, :, :,:]
        gt_diff = y[:, 1:, :, :,:] - y[:, :-1, :, :,:]
        diff = pred_diff - gt_diff
        diff_sq = diff * diff
        diff_sq = torch.mean(diff_sq, 2, True)
        error = torch.sqrt(diff_sq + self.eps)
        loss = torch.mean(error)

        return loss



class _PSNR(nn.Module):
    def __init__(self):
        super(_PSNR, self).__init__()
        self.eps = torch.tensor(1e-10)

        info(f"Init PSNR:")
        info(f"  Note: the psnr max value is {-10 * torch.log10(self.eps)}")

    def forward(self, x, y):
        d = x - y
        mse = torch.mean(d * d) + self.eps
        psnr = -10 * torch.log10(mse)
        return psnr


class EglliePSNR(nn.Module):
    def __init__(self):
        super(EglliePSNR, self).__init__()
        self.psnr = _PSNR()

    def forward(self, batch):
        pred, gt = batch["pred"], batch["gt"]
        psnr = self.psnr(pred, gt)
        return psnr

class EglliePSNR_star(nn.Module):
    def __init__(self):
        super(EglliePSNR_star, self).__init__()
        self.psnr = _PSNR()

    def forward(self, batch):
        pred, gt = batch["pred"], batch["gt"]
        gt_mean = gt.contiguous().view(gt.shape[0], -1).mean(dim=1)
        pred_mean = pred.contiguous().view(gt.shape[0], -1).mean(dim=1)
        ratio = gt_mean/pred_mean
        ratio = ratio.repeat(gt.shape[1],gt.shape[2],gt.shape[3],1).permute(3,0,1,2)
        pred = torch.clamp(pred*ratio, 0, 1)
        
        psnr = self.psnr(pred, gt)
        return psnr



def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    if img1.is_cuda:
        window = window.cuda(img1.get_device())

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel)
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel)
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(nn.Module):
    def __init__(
        self,
        value_range=1.0,
        window_size=11,
        size_average=True,
    ):
        super(SSIM, self).__init__()
        self.value_range = value_range
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = create_window(self.window_size, self.channel)
        self.eps = 0.00001
        info(f"Init SSIM:")
        info(f"  value_range    : {value_range}")
        info(f"  window_size    : {window_size}")
        info(f"  size_average   : {size_average}")

    def forward(self, img1, img2):
        if img1.dim() == 5:
            img1 = torch.flatten(img1, start_dim=0, end_dim=1)
            img2 = torch.flatten(img2, start_dim=0, end_dim=1)

        img1 = img1 / self.value_range
        img2 = img2 / self.value_range
        (_, channel, _, _) = img1.size()
        if self.channel != channel:
            self.channel = channel
            self.window = create_window(self.window_size, self.channel)
        return _ssim(
            img1,
            img2,
            self.window,
            self.window_size,
            channel,
            self.size_average,
        )


class EgllieSSIM(nn.Module):
    def __init__(self):
        super(EgllieSSIM, self).__init__()
        self.ssim = SSIM()

    def forward(self, batch):
        pred, gt = batch["pred"], batch["gt"]
        ssim = self.ssim(pred, gt)
        return ssim
    


