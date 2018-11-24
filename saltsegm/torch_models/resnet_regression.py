"""Regression net with ResBlocks."""

import sys

import torch.nn as nn

from .blocks import ResBlock
from .unet import DownBlock


# TODO: `dropout_rate` is not used
# TODO: change `DownBlock` on max/avg pooling
# TODO: increase number of filters while constructing `ResBlock` instead of `DownBlock`
# TODO: 16 is hardcoded!
class ResReg(nn.Module):
    def __init__(self, n_channels, n_filters=16, dropout_rate=0):
        super(ResReg, self).__init__()

        # initial convolution
        self.preact = nn.Sequential(
            nn.Conv2d(in_channels=n_channels,
                      out_channels=n_filters,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
        )

        # 128->64 img_size; 16->32 filters
        curr_filters = n_filters
        self.res1 = nn.Sequential(
            ResBlock(in_ch=curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            ResBlock(in_ch=curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            DownBlock(in_ch=curr_filters, out_ch=curr_filters * 2)
        )
        curr_filters *= 2

        # 64->32 img_size; 32->64 filters
        self.res2 = nn.Sequential(
            ResBlock(in_ch=curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            ResBlock(in_ch=curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            DownBlock(in_ch=curr_filters, out_ch=curr_filters * 2)
        )
        curr_filters *= 2

        # 32->16 img_size; 64->128 filters
        self.res3 = nn.Sequential(
            ResBlock(in_ch=curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            ResBlock(in_ch=curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            DownBlock(in_ch=curr_filters, out_ch=curr_filters * 2)
        )
        curr_filters *= 2

        # 16->8 img_size; 128->256 filters
        self.res4 = nn.Sequential(
            ResBlock(in_ch=curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            ResBlock(in_ch=curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            DownBlock(in_ch=curr_filters, out_ch=curr_filters * 2)
        )
        curr_filters *= 2

        # 8->4 img_size; 256->256 filters
        self.res5 = nn.Sequential(
            ResBlock(in_ch=curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            ResBlock(in_ch=curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            DownBlock(in_ch=curr_filters, out_ch=curr_filters)
        )

        # 4->1 img_size; 256->1024 filters
        self.res_flatten = nn.Sequential(
            ResBlock(in_ch=curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            nn.Conv2d(in_channels=curr_filters,
                      out_channels=curr_filters * 4,
                      kernel_size=4)
        )
        curr_filters *= 4

        # 1x1 res block
        self.res_final = ResBlock(in_ch=curr_filters,
                                  out_ch=curr_filters,
                                  kernel_size=1,
                                  padding=0)

        # 1->1 img_size, 1024->1 filters final conv
        self.final_conv = nn.Conv2d(in_channels=curr_filters,
                                    out_channels=1,
                                    kernel_size=1,
                                    padding=0)

    def forward(self, x):
        x = self.preact(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res_flatten(x)
        x = self.res_final(x)
        x = self.final_conv(x)
        return x


def get_ResReg(n_channels=1, n_filters=16):
    model = ResReg(n_channels=n_channels, n_filters=n_filters)

    return model


class DownConv(nn.Module):
    def __init__(self, in_ch, out_ch, downratio=2, dropout=0, pool_type='Max'):
        super(DownConv, self).__init__()

        assert pool_type in ('Max', 'Avg'), \
            f'block type should be Max or Avg, {pool_type} given'

        if pool_type == 'Max':
            pool_block = nn.MaxPool2d(downratio)
        elif pool_type == 'Avg':
            pool_block = nn.AvgPool2d(downratio)

        preconv = nn.Conv2d(in_channels=in_ch,
                            out_channels=out_ch,
                            kernel_size=3,
                            padding=1,
                            bias=False)

        self.conv = nn.Sequential(
            preconv,
            pool_block,
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ResRegBasic(nn.Module):
    def __init__(self, n_channels, n_filters=16, dropout_rate=0, pool_type='Max'):
        super(ResRegBasic, self).__init__()

        # initial convolution
        self.preact = nn.Sequential(
            nn.Conv2d(in_channels=n_channels,
                      out_channels=n_filters,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
        )

        # 128->64 img_size; 16->32 filters
        curr_filters = n_filters
        self.res1 = nn.Sequential(
            ResBlock(in_ch=curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            ResBlock(in_ch=curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            DownConv(in_ch=curr_filters, out_ch=curr_filters * 2, downratio=2),
        )
        curr_filters *= 2

        # 64->32 img_size; 32->64 filters
        self.res2 = nn.Sequential(
            ResBlock(in_ch=curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            ResBlock(in_ch=curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            DownConv(in_ch=curr_filters, out_ch=curr_filters * 2, downratio=2),
        )
        curr_filters *= 2

        # 32->16 img_size; 64->128 filters
        self.res3 = nn.Sequential(
            ResBlock(in_ch=curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            ResBlock(in_ch=curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            DownConv(in_ch=curr_filters, out_ch=curr_filters * 2, downratio=2),
        )
        curr_filters *= 2

        # 16->8 img_size; 128->256 filters
        self.res4 = nn.Sequential(
            ResBlock(in_ch=curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            ResBlock(in_ch=curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            DownConv(in_ch=curr_filters, out_ch=curr_filters * 2, downratio=2),
        )
        curr_filters *= 2

        # 8->4 img_size; 256->256 filters
        self.res5 = nn.Sequential(
            ResBlock(in_ch=curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            ResBlock(in_ch=curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            DownConv(in_ch=curr_filters, out_ch=curr_filters * 2, downratio=2),
        )

        # 4->1 img_size; 256->1024 filters
        self.res_flatten = nn.Sequential(
            ResBlock(in_ch=curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            nn.Conv2d(in_channels=curr_filters,
                      out_channels=curr_filters * 4,
                      kernel_size=4)
        )
        curr_filters *= 4

        # 1x1 res block
        self.res_final = ResBlock(in_ch=curr_filters,
                                  out_ch=curr_filters,
                                  kernel_size=1,
                                  padding=0)

        # 1->1 img_size, 1024->1 filters final conv
        self.final_conv = nn.Conv2d(in_channels=curr_filters,
                                    out_channels=1,
                                    kernel_size=1,
                                    padding=0)

    def forward(self, x):
        print(x.size(), file=sys.stderr)
        x = self.preact(x)
        print(x.size(), file=sys.stderr)
        x = self.res1(x)
        print(x.size(), file=sys.stderr)
        x = self.res2(x)
        print(x.size(), file=sys.stderr)
        x = self.res3(x)
        print(x.size(), file=sys.stderr)
        x = self.res4(x)
        print(x.size(), file=sys.stderr)
        x = self.res5(x)
        print(x.size(), file=sys.stderr)
        x = self.res_flatten(x)
        print(x.size(), file=sys.stderr)
        x = self.res_final(x)
        print(x.size(), file=sys.stderr)
        x = self.final_conv(x)
        print(x.size(), file=sys.stderr)
        return x


def get_ResRegBasic(n_channels=1, n_filters=16, dropout_rate=0, pool_type='Max'):
    model = ResRegBasic(n_channels=n_channels, n_filters=n_filters,
                        dropout_rate=dropout_rate, pool_type=pool_type)
    return model
