"""Full assembly of the sub-parts to form the complete net."""

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, dropout_rate=0):
        super(UNet, self).__init__()

        self.inconv = InitConv(in_ch=n_channels, out_ch=64)

        self.down1 = DownBlock(in_ch=64, out_ch=128, dropout_rate=dropout_rate)
        self.down2 = DownBlock(in_ch=128, out_ch=256, dropout_rate=dropout_rate)
        self.down3 = DownBlock(in_ch=256, out_ch=512, dropout_rate=dropout_rate)
        self.down4 = DownBlock(in_ch=512, out_ch=512, dropout_rate=dropout_rate)

        self.up1 = UpBlock(in_ch=512+512, out_ch=256, dropout_rate=dropout_rate)
        self.up2 = UpBlock(in_ch=256+256, out_ch=128, dropout_rate=dropout_rate)
        self.up3 = UpBlock(in_ch=128+128, out_ch=64, dropout_rate=dropout_rate)
        self.up4 = UpBlock(in_ch=64+64, out_ch=64, dropout_rate=dropout_rate)

        self.outconv = OutConv(in_ch=64, out_ch=n_classes)

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


def get_UNet(n_channels=1, n_classes=1, dropout_rate=0):
    model = UNet(n_channels=n_channels, n_classes=n_classes,
                 dropout_rate=dropout_rate)
    model.cuda()
    return model
