# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1(in_ch, out_ch, groups=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, groups=groups, stride=1)

class double_conv(nn.Module):
    '''
    (conv -> BN -> ReLU) * 2
    '''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class in_conv(nn.Module):
    '''
    input -> double_conv
    '''
    def __init__(self, in_ch, out_ch):
        super(in_conv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down_conv(nn.Module):
    '''
    maxpool -> double_conv
    '''
    def __init__(self, in_ch, out_ch):
        super(down_conv, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class up_conv(nn.Module):
    '''
    (upsample -> conv1) or (transpose conv) -> concate -> double_conv
    '''
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up_conv, self).__init__()

        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                conv1(in_ch, out_ch)
            )
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class out_conv(nn.Module):
    '''
    conv -> output
    '''
    def __init__(self, in_ch, out_ch):
        super(out_conv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

