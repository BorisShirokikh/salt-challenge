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
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        """Residual block."""
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size,
                      padding=padding, bias=False)
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


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x
