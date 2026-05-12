"""
model.py — U-Net-DC architecture (512x512 full-slice variant).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class DACBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        make = lambda d: nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=d, dilation=d, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.conv1 = make(1); self.conv3 = make(3)
        self.conv5 = make(5); self.conv7 = make(7)
        self.project = nn.Sequential(
            nn.Conv2d(ch*4, ch, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
    def forward(self, x):
        y = torch.cat([self.conv1(x), self.conv3(x), self.conv5(x), self.conv7(x)], dim=1)
        return self.project(y) + x


class RMPBlock(nn.Module):
    def __init__(self, ch, pool_channels=None):
        super().__init__()
        pc = pool_channels or max(8, ch // 4)
        self.pools = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(s), nn.Conv2d(ch, pc, 1), nn.ReLU(inplace=True))
            for s in (1, 2, 3, 6)])
        self.project = nn.Sequential(
            nn.Conv2d(ch + 4*pc, ch, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
    def forward(self, x):
        h, w = x.shape[-2:]
        pooled = [F.interpolate(p(x), size=(h,w), mode="bilinear", align_corners=False) for p in self.pools]
        return self.project(torch.cat([x]+pooled, dim=1))


class UNetDC512(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, base=32):
        super().__init__()
        b = base
        self.enc1 = DoubleConv(in_channels, b);   self.enc2 = DoubleConv(b,   b*2)
        self.enc3 = DoubleConv(b*2, b*4);         self.enc4 = DoubleConv(b*4, b*8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(b*8, b*16)
        self.dac = DACBlock(b*16);  self.rmp = RMPBlock(b*16)
        self.up4  = nn.ConvTranspose2d(b*16, b*8, 2, stride=2); self.dec4 = DoubleConv(b*16, b*8)
        self.up3  = nn.ConvTranspose2d(b*8,  b*4, 2, stride=2); self.dec3 = DoubleConv(b*8,  b*4)
        self.up2  = nn.ConvTranspose2d(b*4,  b*2, 2, stride=2); self.dec2 = DoubleConv(b*4,  b*2)
        self.up1  = nn.ConvTranspose2d(b*2,  b,   2, stride=2); self.dec1 = DoubleConv(b*2,  b)
        self.out  = nn.Conv2d(b, num_classes, 1)

    def forward(self, x):
        e1=self.enc1(x); e2=self.enc2(self.pool(e1))
        e3=self.enc3(self.pool(e2)); e4=self.enc4(self.pool(e3))
        b=self.bottleneck(self.pool(e4)); b=self.dac(b); b=self.rmp(b)
        d4=self.up4(b);  d4=self.dec4(torch.cat([d4,e4],dim=1))
        d3=self.up3(d4); d3=self.dec3(torch.cat([d3,e3],dim=1))
        d2=self.up2(d3); d2=self.dec2(torch.cat([d2,e2],dim=1))
        d1=self.up1(d2); d1=self.dec1(torch.cat([d1,e1],dim=1))
        return self.out(d1)
