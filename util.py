from torch import nn
import torch.nn.functional as F

def upscale(x, mode, size=None, scale=2):
    if size is not None:
        return F.interpolate(x, size=size, mode=mode)
    return F.interpolate(x, scale_factor=scale, mode=mode)

class ConvBNReLU(nn.Sequential):
    def __init__(self, cin: int, cout: int, ksize: int, **kwargs):
        super().__init__(
            nn.Conv2d(cin, cout, ksize, **kwargs),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
        )
