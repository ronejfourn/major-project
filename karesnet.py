from torch import nn
from relukan import ReLUKANConv2DLayer

class KAResNet9(nn.Module):
    channels = [64, 96, 160, 256]

    def __init__(self, pretrained=False):
        super().__init__()
        _ = pretrained
        self.conv1 = ReLUKANConv2DLayer(3, self.channels[0], kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = KABasicBlock(self.channels[0], self.channels[0], 1)
        self.layer2 = KABasicBlock(self.channels[0], self.channels[1])
        self.layer3 = KABasicBlock(self.channels[1], self.channels[2])
        self.layer4 = KABasicBlock(self.channels[2], self.channels[3])

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4

class KABasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, s=2):
        super().__init__()
        self.conv1 = ReLUKANConv2DLayer(in_channels, out_channels, kernel_size=3, stride=s, padding=1)
        self.s = s
        if s != 1:
            self.dconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=s, padding=0)
            self.dbn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.s != 1:
            y = self.dbn(self.dconv(x))
        else:
            y = x
        x = self.conv1(x)
        x = x + y
        return x
