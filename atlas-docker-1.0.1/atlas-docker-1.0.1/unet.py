import torch.nn as nn
import torch.nn.functional as F
import math


def conv_block(in_channels, out_channels, kernel_size=3):
    block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1,
                      bias=False),
            nn.ReLU(inplace=True)
    )
    return block


def last_block(in_channels, out_channels, kernel_size=3):
    block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1,
                      bias=False),
            nn.Softmax(dim=1)
    )
    return block


class UNet3D(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(UNet3D, self).__init__()

        self.enc = conv_block(in_channels=in_channel, out_channels=16)
        self.max = nn.MaxPool3d(kernel_size=2)
        self.enc2 = conv_block(in_channels=16, out_channels=32)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.fin = last_block(in_channels=32, out_channels=out_channel)

    def forward(self, x):
        x = self.enc(x)
        x_down = self.max(x)
        x_down = self.enc2(x_down)
        x_up = self.up(x_down)
        z_diff = x.shape[2] - x_up.shape[2]
        y_diff = x.shape[3] - x_up.shape[3]
        x_diff = x.shape[4] - x_up.shape[4]
        x = F.pad(x_up, (math.floor(x_diff / 2), math.ceil(x_diff / 2), math.floor(y_diff / 2), math.ceil(y_diff / 2),
                         math.floor(z_diff / 2), math.ceil(z_diff / 2)))
        x = self.fin(x)

        return x
