"""Full assembly of the sub-parts to form the complete net."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import InitConv, ResBlock, OutConv, DoubleConv


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0, pool_type='Max'):
        """Block for downsampling and convolution in U-Net"""
        super(DownBlock, self).__init__()

        assert pool_type in ('Max', 'Avg'), \
            f'block type should be Max or Avg, {pool_type} given'

        super(DownBlock, self).__init__()

        if pool_type == 'Max':
            pool_block = nn.MaxPool2d(2)
        elif pool_type == 'Avg':
            pool_block = nn.AvgPool2d(2)

        dropout = nn.Dropout2d(p=dropout_rate, inplace=True)
        
        conv_block = DoubleConv

        self.pool_conv = nn.Sequential(
            pool_block,
            dropout,
            conv_block(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.pool_conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0, learnable_upsample=True):
        """Block for upsampling, concat and conv in U-Net"""
        super(UpBlock, self).__init__()

        # TODO: would be a nice idea if the upsampling could be learned too,
        if learnable_upsample:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        conv_block = DoubleConv
        self.conv = conv_block(in_ch, out_ch)
        
        self.dropout = nn.Dropout2d(p=dropout_rate, inplace=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_x = x1.size()[-2] - x2.size()[-2]
        diff_y = x1.size()[-1] - x2.size()[-1]

        x2 = F.pad(x2, (diff_x // 2, int(diff_x / 2),
                        diff_y // 2, int(diff_y / 2)))

        x = torch.cat([x2, x1], dim=1)
        
        x = self.dropout(x)

        x = self.conv(x)

        return x


class ResHead(nn.Module):
    def __init__(self, n_channels):
        super(ResHead, self).__init__()

        conv_block = ResBlock

        self.res_block_1 = conv_block(in_ch=n_channels, out_ch=n_channels, kernel_size=3,
                                      padding=1)
        self.res_block_2 = conv_block(in_ch=n_channels, out_ch=n_channels, kernel_size=3,
                                      padding=1)
        self.res_block_final = conv_block(in_ch=n_channels, out_ch=n_channels,
                                          kernel_size=1, padding=0)

    def forward(self, x):
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_final(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_filters=64, dropout_rate=0):
        super(UNet, self).__init__()

        # start -- initial convolution
        self.inconv = InitConv(in_ch=n_channels, out_ch=64)

        # encoder part -- downsampling and conv blocks
        self.down1 = DownBlock(in_ch=n_filters, out_ch=n_filters * 2,
                               dropout_rate=dropout_rate)
        self.down2 = DownBlock(in_ch=n_filters * 2, out_ch=n_filters * 4,
                               dropout_rate=dropout_rate)
        self.down3 = DownBlock(in_ch=n_filters * 4, out_ch=n_filters * 8,
                               dropout_rate=dropout_rate)
        self.down4 = DownBlock(in_ch=n_filters * 8, out_ch=n_filters * 8,
                               dropout_rate=dropout_rate)

        # decoder part -- upsampling and conv blocks
        self.up1 = UpBlock(in_ch=n_filters * 8 * 2, out_ch=n_filters * 4,
                           dropout_rate=dropout_rate)
        self.up2 = UpBlock(in_ch=n_filters * 4 * 2, out_ch=n_filters * 2,
                           dropout_rate=dropout_rate)
        self.up3 = UpBlock(in_ch=n_filters * 2 * 2, out_ch=n_filters,
                           dropout_rate=dropout_rate)
        self.up4 = UpBlock(in_ch=n_filters * 2, out_ch=n_filters,
                           dropout_rate=dropout_rate)

        # conv 1x1 as FCL
        self.outconv = OutConv(in_ch=n_filters, out_ch=n_classes)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outconv(x)
        return x


def get_UNet(n_channels=1, n_classes=1, n_filters=64, dropout_rate=0):
    model = UNet(n_channels=n_channels, n_classes=n_classes,
                 n_filters=n_filters, dropout_rate=dropout_rate)
    return model


class UNetRes(nn.Module):
    def __init__(self, n_channels, n_classes, n_filters=64, dropout_rate=0):
        super(UNetRes, self).__init__()

        # start -- initial convolution
        self.inconv = InitConv(in_ch=n_channels, out_ch=64)

        # encoder part -- downsampling and conv blocks
        self.down1 = DownBlock(in_ch=n_filters, out_ch=n_filters * 2,
                               dropout_rate=dropout_rate)
        self.down2 = DownBlock(in_ch=n_filters * 2, out_ch=n_filters * 4,
                               dropout_rate=dropout_rate)
        self.down3 = DownBlock(in_ch=n_filters * 4, out_ch=n_filters * 8,
                               dropout_rate=dropout_rate)
        self.down4 = DownBlock(in_ch=n_filters * 8, out_ch=n_filters * 8,
                               dropout_rate=dropout_rate)

        # decoder part -- upsampling and conv blocks
        self.up1 = UpBlock(in_ch=n_filters * 8 * 2, out_ch=n_filters * 4,
                           dropout_rate=dropout_rate)
        self.up2 = UpBlock(in_ch=n_filters * 4 * 2, out_ch=n_filters * 2,
                           dropout_rate=dropout_rate)
        self.up3 = UpBlock(in_ch=n_filters * 2 * 2, out_ch=n_filters,
                           dropout_rate=dropout_rate)
        self.up4 = UpBlock(in_ch=n_filters * 2, out_ch=n_filters,
                           dropout_rate=dropout_rate)

        # resnet head
        self.res_block = ResHead(n_channels=n_filters)

        # conv 1x1 as FCL
        self.outconv = OutConv(in_ch=n_filters, out_ch=n_classes)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.res_block(x)
        x = self.outconv(x)
        return x


def get_UNetRes(n_channels=1, n_classes=1, n_filters=64, dropout_rate=0):
    model = UNetRes(n_channels=n_channels, n_classes=n_classes,
                    n_filters=n_filters, dropout_rate=dropout_rate)
    return model
