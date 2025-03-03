from torch import nn

class ResNet18(nn.Module):
    channels = [64, 128, 256, 512]

    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import resnet18

        resnet = resnet18(pretrained=pretrained)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4

class ResNet9(nn.Module):
    channels = [64, 128, 256, 512]

    def __init__(self, pretrained=False):
        super().__init__()
        _ = pretrained
        self.conv1 = nn.Conv2d(3, self.channels[0], kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = BasicBlock(self.channels[0], self.channels[0], 1)
        self.layer2 = BasicBlock(self.channels[0], self.channels[1])
        self.layer3 = BasicBlock(self.channels[1], self.channels[2])
        self.layer4 = BasicBlock(self.channels[2], self.channels[3])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, s=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.s = s
        if s != 1:
            self.dconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=s, padding=0)
            self.dbn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.s != 1:
            y = self.dbn(self.dconv(x))
        else:
            y = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x + y
        return x
