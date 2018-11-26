"""Sub-parts of the U-Net model."""

import torch.nn as nn
from functools import partial


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


def identity(x):
    return x


# pad_modules = {'Replication': nn.ReplicationPad2d, 'Constant': nn.ConstantPad2d}


# def sum_channelwise(x, x_conv):
#     n_ch = x.shape[1]
#     n_conv_ch = x_conv.shape[1]
#     if n_ch == n_conv_ch:
#         return x_conv + x
#     else:
#         return torch.cat((x_conv[:, :n_ch, ...] + x, x_conv[:, n_ch:, ...]), dim=1)


# class 
# def adjust_channels(x, x_conv):
#     in_ch = x.shape[1]
#     out_ch = x_conv.shape[1]
#     return x_conv + nn.Conv2d(in_ch, out_ch, kernel_size=1)(x)


# class ResBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, pad_type='Constant',
#                  adjust_n_ch_with_conv=True):
#         """Residual block."""
#         super(ResBlock, self).__init__()
#         assert pad_type in ('Constant', 'Replication'), \
#             f'`pad_type` should be `Constant` or `Replication`, {pad_type} given.'
#         pad = pad_modules[pad_type]
#         if pad_type == 'Constant':
#             pad = partial(pad, value=0)

#         self.conv = nn.Sequential(
#             nn.BatchNorm2d(in_ch),
#             nn.ReLU(inplace=True),
#             pad(padding=padding),
#             nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size,
#                       bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             pad(padding=padding),
#             nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size,
#                       bias=False)
#         )

#         if in_ch <= out_ch:
#             if adjust_n_ch_with_conv:
#                 self.adjust_to_ch = adjust_channels
#             else:
#                 self.adjust_to_ch = sum_channelwise
#         else:
#             assert adjust_n_ch_with_conv, \
#                 'Cannot adjust in_ch > out_ch without conv.'
#             self.adjust_to_ch = adjust_channels

#     def forward(self, x):
#         x_conv = self.conv(x)
#         x = self.adjust_to_ch(x, x_conv)
#         return x


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
        """Residual block."""
        super(ResBlock, self).__init__()

        if in_ch != out_ch or stride != 1:
            self.adjust_to_stride = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)
        else:
            self.adjust_to_stride = identity

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
        x = self.adjust_to_stride(x)
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
