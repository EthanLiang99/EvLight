from absl.logging import info
from torch import nn

from egllie.losses.image_loss import EglliePSNR, EgllieSSIM,EglliePSNR_star


def get_single_metric(config):
    if config.NAME == "SSIM":
        return EgllieSSIM()
    elif config.NAME == "PSNR":
        return EglliePSNR()
    elif config.NAME == "PSNR_star":
        return EglliePSNR_star()
    else:
        raise ValueError(f"Unknown config: {config}")


class MixedMetric(nn.Module):
    def __init__(self, configs):
        super(MixedMetric, self).__init__()
        self.metric = []
        self.eval = []
        for config in configs:
            self.metric.append(config.NAME)
            self.eval.append(get_single_metric(config))
        info(f"Init Mixed Metric: {configs}")

    def forward(self, batch):
        r = []
        for m, e in zip(self.metric, self.eval):
            r.append((m, e(batch)))
        return r
