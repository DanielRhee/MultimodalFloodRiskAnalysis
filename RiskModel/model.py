import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, 3, padding=1, bias=False),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(outChannels, outChannels, 3, padding=1, bias=False),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(inChannels, outChannels)

    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.up = nn.ConvTranspose2d(inChannels, inChannels // 2, 2, stride=2)
        self.conv = DoubleConv(inChannels, outChannels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class FloodRiskUNet(nn.Module):
    def __init__(self, inChannels=4, outChannels=1, baseFilters=32):
        super().__init__()

        self.inc = DoubleConv(inChannels, baseFilters)
        self.down1 = Down(baseFilters, baseFilters * 2)
        self.down2 = Down(baseFilters * 2, baseFilters * 4)
        self.down3 = Down(baseFilters * 4, baseFilters * 8)
        self.down4 = Down(baseFilters * 8, baseFilters * 16)

        self.up1 = Up(baseFilters * 16, baseFilters * 8)
        self.up2 = Up(baseFilters * 8, baseFilters * 4)
        self.up3 = Up(baseFilters * 4, baseFilters * 2)
        self.up4 = Up(baseFilters * 2, baseFilters)

        self.outc = nn.Conv2d(baseFilters, outChannels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        output = self.outc(x)
        return torch.sigmoid(output)

class FloodRiskEfficientNet(nn.Module):
    def __init__(self, inChannels=4, outChannels=1):
        super().__init__()

        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        originalConv = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            inChannels,
            originalConv.out_channels,
            kernel_size=originalConv.kernel_size,
            stride=originalConv.stride,
            padding=originalConv.padding,
            bias=False
        )

        with torch.no_grad():
            meanWeight = originalConv.weight.mean(dim=1, keepdim=True)
            for i in range(inChannels):
                backbone.features[0][0].weight[:, i:i+1] = meanWeight

        self.encoder = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.decoder = nn.Sequential(
            nn.Conv2d(1280, 256, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, outChannels, 3, padding=1),
        )

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return torch.sigmoid(output)

def getModel(architecture='unet', inChannels=4, outChannels=1, baseFilters=32):
    if architecture == 'unet':
        return FloodRiskUNet(inChannels, outChannels, baseFilters)
    elif architecture == 'efficientnet':
        return FloodRiskEfficientNet(inChannels, outChannels)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
