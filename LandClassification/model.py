import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(outChannels)
        self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(outChannels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.convBlock = ConvBlock(inChannels, outChannels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.convBlock(x)
        x = self.pool(skip)
        return x, skip


class DecoderBlock(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(inChannels, outChannels, kernel_size=2, stride=2)
        self.convBlock = ConvBlock(inChannels, outChannels)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.convBlock(x)
        return x


class MiniUNet(nn.Module):
    def __init__(self, inChannels=4, numClasses=19, baseFilters=32):
        super().__init__()

        self.enc1 = EncoderBlock(inChannels, baseFilters)
        self.enc2 = EncoderBlock(baseFilters, baseFilters * 2)
        self.enc3 = EncoderBlock(baseFilters * 2, baseFilters * 4)
        self.enc4 = EncoderBlock(baseFilters * 4, baseFilters * 8)

        self.bottleneck = ConvBlock(baseFilters * 8, baseFilters * 16)

        self.dec4 = DecoderBlock(baseFilters * 16, baseFilters * 8)
        self.dec3 = DecoderBlock(baseFilters * 8, baseFilters * 4)
        self.dec2 = DecoderBlock(baseFilters * 4, baseFilters * 2)
        self.dec1 = DecoderBlock(baseFilters * 2, baseFilters)

        self.outConv = nn.Conv2d(baseFilters, numClasses, kernel_size=1)

        self._initWeights()

    def _initWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1, skip1 = self.enc1(x)
        x2, skip2 = self.enc2(x1)
        x3, skip3 = self.enc3(x2)
        x4, skip4 = self.enc4(x3)

        bottleneck = self.bottleneck(x4)

        x = self.dec4(bottleneck, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        x = self.outConv(x)

        return x


def getMiniUNet(inChannels=None, numClasses=None, baseFilters=None):
    if inChannels is None:
        inChannels = config.IN_CHANNELS
    if numClasses is None:
        numClasses = config.NUM_CLASSES
    if baseFilters is None:
        baseFilters = config.BASE_FILTERS

    model = MiniUNet(inChannels, numClasses, baseFilters)
    return model


def countParameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
