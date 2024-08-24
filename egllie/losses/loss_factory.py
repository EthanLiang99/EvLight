from absl.logging import info
from torch.nn.modules.loss import _Loss

from .image_loss import (
    L1CharbonnierLoss,
    PerceptualLoss
)


def get_single_loss(config):
    if config.NAME == "normal-light-reconstructed-loss": 
        return L1CharbonnierLoss()
    elif config.NAME == "normal-light-perceptual-loss":
        return PerceptualLoss()
    else:
        raise ValueError(f"Unknown loss: {config.NAME}")


class MixedLoss(_Loss):
    def __init__(self, configs):
        super(MixedLoss, self).__init__()
        self.loss = []
        self.weight = []
        self.criterion = []
        for item in configs:
            self.loss.append(item.NAME)
            self.weight.append(item.WEIGHT)
            self.criterion.append(get_single_loss(item))
        info(f"Init Mixed Loss: {configs}")

    def forward(self, batch):
        name_to_loss = []
        total = 0
        for n, w, fun in zip(self.loss, self.weight, self.criterion):
            tmp = fun(batch)
            name_to_loss.append((n, tmp))
            total = total + tmp * w
        return total, name_to_loss
