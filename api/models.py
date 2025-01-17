import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Définition de VGG16 U-Net
class VGG16UNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(VGG16UNet, self).__init__()
        vgg16 = models.vgg16(pretrained=pretrained)
        features = list(vgg16.features.children())
        self.enc1 = nn.Sequential(*features[:5])
        self.enc2 = nn.Sequential(*features[5:10])
        self.enc3 = nn.Sequential(*features[10:17])
        self.enc4 = nn.Sequential(*features[17:24])
        self.center = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dec4 = self._decoder_block(512 + 512, 256)
        self.dec3 = self._decoder_block(256 + 256, 128)
        self.dec2 = self._decoder_block(128 + 128, 64)
        self.dec1 = self._decoder_block(64 + 64, 64)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2))
        center = self.center(F.max_pool2d(enc4, kernel_size=2))
        dec4 = self.dec4(torch.cat([F.interpolate(center, enc4.size()[2:], mode='bilinear', align_corners=False), enc4], dim=1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, enc3.size()[2:], mode='bilinear', align_corners=False), enc3], dim=1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, enc2.size()[2:], mode='bilinear', align_corners=False), enc2], dim=1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, enc1.size()[2:], mode='bilinear', align_corners=False), enc1], dim=1))
        return F.interpolate(self.final(dec1), x.size()[2:], mode='bilinear', align_corners=False)
