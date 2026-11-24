import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class EndoscopyDepthModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = resnet18(weights=None)

        self.decoder4 = ConvBlock(512 + 256, 256)
        self.decoder3 = ConvBlock(256 + 128, 128)
        self.decoder2 = ConvBlock(128 + 64, 64)
        self.decoder1 = ConvBlock(64 + 64, 64)

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        e1 = self.encoder.relu(
            self.encoder.bn1(self.encoder.conv1(x))
        )

        e2 = self.encoder.layer1(
            self.encoder.maxpool(e1)
        )

        e3 = self.encoder.layer2(e2)
        e4 = self.encoder.layer3(e3)
        e5 = self.encoder.layer4(e4)

        d4 = F.interpolate(
            e5, size=e4.shape[2:],
            mode="bilinear", align_corners=False
        )
        d4 = self.decoder4(torch.cat([d4, e4], dim=1))

        d3 = F.interpolate(
            d4, size=e3.shape[2:],
            mode="bilinear", align_corners=False
        )
        d3 = self.decoder3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(
            d3, size=e2.shape[2:],
            mode="bilinear", align_corners=False
        )
        d2 = self.decoder2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(
            d2, size=e1.shape[2:],
            mode="bilinear", align_corners=False
        )
        d1 = self.decoder1(torch.cat([d1, e1], dim=1))

        d1 = F.interpolate(
            d1, size=x.shape[2:],
            mode="bilinear", align_corners=False
        )

        return self.final_conv(d1)