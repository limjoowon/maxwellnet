""" 
Adapted and modified from https://github.com/jvanvugt/pytorch-unet

Modified parts:
Copyright (c) 2022 Joowon Lim, limjoowon@gmail.com

Original parts:
MIT License

Copyright (c) 2018 Joris

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, depth=5, wf=6, norm='weight', up_mode='upconv'):
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.down_path = nn.ModuleList()
        self.up_path = nn.ModuleList()

        prev_channels = int(in_channels)

        for i in range(depth):
            if i != depth - 1:
                if i == 0:
                    self.down_path.append(UNetConvBlock(
                        prev_channels, [wf * (2 ** i), wf * (2 ** i)], 3, 0, norm))
                else:
                    self.down_path.append(UNetConvBlock(
                        prev_channels, [wf * (2 ** i), wf * (2 ** i)], 3, 0, norm))
                prev_channels = int(wf * (2 ** i))
                self.down_path.append(nn.AvgPool2d(2))
            else:
                self.down_path.append(UNetConvBlock(
                    prev_channels, [wf * (2 ** i), wf * (2 ** (i - 1))], 3, 0, norm))
                prev_channels = int(wf * (2 ** (i - 1)))

        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, [wf * (2 ** i), int(wf * (2 ** (i - 1)))], up_mode, 3, 0, norm))
            prev_channels = int(wf * (2 ** (i - 1)))

        self.last_conv = nn.Conv2d(
            prev_channels, out_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, scat_pot):
        blocks = []
        x = scat_pot
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i % 2 == 0 and i != (len(self.down_path) - 1):
                blocks.append(x)
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last_conv(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kersize, padding, norm):
        super(UNetConvBlock, self).__init__()
        block = []
        if norm == 'weight':
            block.append(nn.ReplicationPad2d(1))
            block.append(nn.utils.weight_norm((nn.Conv2d(in_size, out_size[0], kernel_size=int(kersize),
                                                         padding=int(0), bias=True)), name='weight'))
            block.append(nn.CELU())
            block.append(nn.ReplicationPad2d(1))
            block.append(nn.utils.weight_norm((nn.Conv2d(out_size[0], out_size[1], kernel_size=int(kersize),
                                                         padding=int(0), bias=True)), name='weight'))
        elif norm == 'batch':
            block.append(nn.ReflectionPad2d(1))
            block.append(nn.Conv2d(in_size, out_size[0], kernel_size=int(kersize),
                                   padding=int(padding), bias=True))
            block.append(nn.BatchNorm2d(out_size[0]))
            block.append(nn.CELU())

            block.append(nn.ReflectionPad2d(1))
            block.append(nn.Conv2d(out_size[0], out_size[1], kernel_size=int(kersize),
                                   padding=int(padding), bias=True))
            block.append(nn.BatchNorm2d(out_size[1]))

        elif norm == 'no':
            block.append(nn.ReplicationPad2d(1))
            block.append((nn.Conv2d(in_size, out_size[0], kernel_size=int(kersize),
                                    padding=int(0), bias=True)))
            block.append(nn.CELU())
            block.append(nn.ReplicationPad2d(1))
            block.append((nn.Conv2d(out_size[0], out_size[1], kernel_size=int(kersize),
                                    padding=int(0), bias=True)))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, kersize, padding, norm):
        super(UNetUpBlock, self).__init__()
        block = []
        if up_mode == 'upconv':
            block.append(nn.ConvTranspose2d(in_size, in_size,
                         kernel_size=2, stride=2, bias=False))
        elif up_mode == 'upsample':
            block.append(nn.Upsample(mode='bilinear', scale_factor=2))
            block.append(nn.Conv2d(in_size, in_size,
                         kernel_size=1, bias=False))

        self.block = nn.Sequential(*block)
        self.conv_block = UNetConvBlock(
            in_size * 2, out_size, kersize, padding, norm)

    def forward(self, x, bridge):
        up = self.block(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out
