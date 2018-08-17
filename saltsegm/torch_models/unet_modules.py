"""Sub-parts of the U-Net model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Performs simple double convolution.
            (conv => BN => ReLU) * 2"""
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Residual block."""
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        x_conv = self.conv(x)
        return x + x_conv


class InitConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        """Initial convolution in Unet."""
        super(InitConv, self).__init__()
        
        conv_block = DoubleConv
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool_type='Max'):
        """Block for downsampling and convolution in Unet"""
        assert pool_type in ('Max', 'Avg'), \
            f'block type should be Max or Avg, {pool_type} given'

        super(DownBlock, self).__init__()

        conv_block = DoubleConv

        if pool_type == 'Max':
            pool_block = nn.MaxPool2d(2)
        elif pool_type == 'Avg':
            pool_block = nn.AvgPool2d(2)

        self.pool_conv = nn.Sequential(
            pool_block,
            conv_block(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.pool_conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, learnable_upsample=True):
        """Block for upsampling, concat and conv in Unet"""
        super(UpBlock, self).__init__()

        # TODO: would be a nice idea if the upsampling could be learned too,
        if learnable_upsample:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        conv_block = DoubleConv
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_x = x1.size()[-2] - x2.size()[-2]
        diff_y = x1.size()[-1] - x2.size()[-1]

        x2 = F.pad(x2, (diff_x // 2, int(diff_x / 2),
                        diff_y // 2, int(diff_y / 2)))

        x = torch.cat([x2, x1], dim=1)

        x = self.conv(x)

        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x
