"""Regression net with Resblocks"""

from .unet_parts import *

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
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # 128->64 img_size; 16->32 filters
        curr_filters = n_filters
        self.res1 = nn.Sequential(
            ResBlock(in_ch= curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            ResBlock(in_ch= curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            DownBlock(in_ch=curr_filters, out_ch= curr_filters * 2)
        )
        curr_filters *= 2

        # 64->32 img_size; 32->64 filters
        self.res2 = nn.Sequential(
            ResBlock(in_ch= curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            ResBlock(in_ch= curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            DownBlock(in_ch=curr_filters, out_ch= curr_filters * 2)
        )
        curr_filters *= 2

        # 32->16 img_size; 64->128 filters
        self.res3 = nn.Sequential(
            ResBlock(in_ch= curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            ResBlock(in_ch= curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            DownBlock(in_ch=curr_filters, out_ch= curr_filters * 2)
        )
        curr_filters *= 2

        # 16->4 img_size; 128->256 filters
        self.res4 = nn.Sequential(
            ResBlock(in_ch= curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            ResBlock(in_ch= curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            DownBlock(in_ch=curr_filters, out_ch= curr_filters * 2)
        )
        curr_filters *= 2

        # 4->1 img_size; 256->1024 filters
        self.res_flattern = nn.Sequential(
            ResBlock(in_ch= curr_filters, out_ch=curr_filters,
                     kernel_size=3, padding=1),
            nn.Conv2d(in_channels=curr_filters,
                      out_channels=curr_filters*4,
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

    def forward(self,x):
        x = self.preact(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res_flattern(x)
        x = self.res_final(x)
        x = self.final_conv(x)
        return x


def get_ResReg(n_channels=1):
    model = ResReg(n_channels=n_channels, n_filters=16)

    return model
