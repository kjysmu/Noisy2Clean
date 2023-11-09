import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import Base, init_layer, init_bn, act
from task.separation import Separation


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, momentum):
        super().__init__()

        padding = kernel_size// 2

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=1,
            padding=padding,
            bias=False,
        )

        
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=1,
            padding=padding,
            bias=False,
        )

        self.bn1 = nn.BatchNorm1d(out_channels, momentum=momentum)
        self.bn2 = nn.BatchNorm1d(out_channels, momentum=momentum)


    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample, momentum):
        super().__init__()

        self.conv_block = ConvBlock(in_channels, out_channels, kernel_size, momentum)
        self.downsample = downsample

    def forward(self, x):
        encoder = self.conv_block(x)
        encoder_pool = F.avg_pool1d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, upsample, momentum):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = upsample

        self.conv1 = torch.nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.stride,
            stride=self.stride,
            padding=0,
            bias=False,
            dilation=1,
        )

        self.bn1 = nn.BatchNorm1d(out_channels, momentum=momentum)

        self.conv_block2 = ConvBlock(
            out_channels*3, out_channels, kernel_size, momentum
        )

    def forward(self, x, x_skip):
        x = self.bn1(self.conv1(x))
        x = torch.cat((x, x_skip), dim=1)
        x = self.conv_block2(x)
        return x



class UNet(Separation):
    def __init__(self, channels_num, task_args=None):
        super().__init__()

        momentum = 0.01   
        self.time_downsample_ratio = 2 ** 6  # This number equals 2^{#encoder_blcoks}

        self.bn0 = nn.BatchNorm1d(channels_num, momentum=momentum)

        self.encoder_block1 = EncoderBlock(
            in_channels=channels_num,
            out_channels=16,
            kernel_size=3,
            downsample=2,
            momentum=momentum,
        )
        self.encoder_block2 = EncoderBlock(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            downsample=2,
            momentum=momentum,
        )
        self.encoder_block3 = EncoderBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            downsample=2,
            momentum=momentum,
        )
        self.encoder_block4 = EncoderBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            downsample=2,
            momentum=momentum,
        )

        self.decoder_block1 = DecoderBlock(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            upsample=2,
            momentum=momentum,
        )
        self.decoder_block2 = DecoderBlock(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            upsample=2,
            momentum=momentum,
        )
        self.decoder_block3 = DecoderBlock(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            upsample=2,
            momentum=momentum,
        )
        self.decoder_block4 = DecoderBlock(
            in_channels=16,
            out_channels=8,
            kernel_size=3,
            upsample=2,
            momentum=momentum,
        )


        self.after_conv_block1 = ConvBlock(
            in_channels=8,
            out_channels=8,
            kernel_size=3,
            momentum=momentum,
        )

        self.after_conv2 = nn.Conv1d(
            in_channels=8,
            out_channels=8,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x):
                     
        x = self.bn0(x)
        x, x1_skip = self.encoder_block1(x)
        x, x2_skip = self.encoder_block2(x)
        x, x3_skip = self.encoder_block3(x)
        x, x4_skip = self.encoder_block4(x)
        
        
        x = self.decoder_block1(x, x4_skip)
        x = self.decoder_block2(x, x3_skip)
        x = self.decoder_block3(x, x2_skip)
        x = self.decoder_block4(x, x1_skip)
        
        x = self.after_conv_block1(x)
        x = self.after_conv2(x)

        return x 
