import torch
from torch import nn
from util import ConvBNReLU, upscale

class SwiftNet(nn.Module):
    def __init__(
        self,
        backbone,
        pretrained=True,
        num_classes=19,
        up_channels=128,
        up_mode='nearest',
    ):
        super().__init__()

        self.up_mode = up_mode
        self.backbone = backbone(pretrained=pretrained)
        channels = backbone.channels

        self.connection1 = ConvBNReLU(channels[0], up_channels, 1)
        self.connection2 = ConvBNReLU(channels[1], up_channels, 1)
        self.connection3 = ConvBNReLU(channels[2], up_channels, 1)

        self.spp = PyramidPoolingModule(
            channels[3],
            up_channels,
            up_mode,
            bias=True
        )

        self.up_stage3 = ConvBNReLU(up_channels, up_channels, 3, padding=1)
        self.up_stage2 = ConvBNReLU(up_channels, up_channels, 3, padding=1)
        self.up_stage1 = ConvBNReLU(up_channels, num_classes, 3, padding=1)

    def forward(self, x):
        size = x.size()[2:]
        x1, x2, x3, x4 = self.backbone(x)

        x1 = self.connection1(x1)
        x2 = self.connection2(x2)
        x3 = self.connection3(x3)

        y = self.spp(x4)

        y = upscale(y, self.up_mode) + x3
        y = self.up_stage3(y)

        y = upscale(y, self.up_mode) + x2
        y = self.up_stage2(y)

        y = upscale(y, self.up_mode) + x1
        y = self.up_stage1(y)

        y = upscale(y, self.up_mode, size=size)

        return y

class PyramidPoolingModule(nn.Module):
    def __init__(self, c_in, c_out, up_mode, pool_sizes=[1,2,4,8], bias=False):
        super().__init__()
        c_hid = int(c_in // 4)
        self.stage1 = self._stage(c_in, c_hid, pool_sizes[0])
        self.stage2 = self._stage(c_in, c_hid, pool_sizes[1])
        self.stage3 = self._stage(c_in, c_hid, pool_sizes[2])
        self.stage4 = self._stage(c_in, c_hid, pool_sizes[3])
        self.conv = ConvBNReLU(2*c_in, c_out, 1, bias=bias)
        self.up_mode = up_mode

    def _stage(self, c_in, c_out, pool_size):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Conv2d(c_in, c_out, 1),
        )

    def forward(self, x):
        size = x.size()[2:]
        x1 = upscale(self.stage1(x), self.up_mode, size=size)
        x2 = upscale(self.stage2(x), self.up_mode, size=size)
        x3 = upscale(self.stage3(x), self.up_mode, size=size)
        x4 = upscale(self.stage4(x), self.up_mode, size=size)
        x = self.conv(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x
