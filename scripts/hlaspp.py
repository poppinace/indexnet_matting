"""
This implementation is modified from the following repository:
https://github.com/jfzhang95/pytorch-deeplab-xception

IndexNet Matting

Indices Matter: Learning to Index for Deep Image Matting
IEEE/CVF International Conference on Computer Vision, 2019

This software is strictly limited to academic purposes only
Copyright (c) 2019, Hao Lu (hao.lu@adelaide.edu.au)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
  
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.nn import SynchronizedBatchNorm2d


def depth_sep_dilated_conv_3x3_bn(inp, oup, padding, dilation, BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, 1, padding=padding, dilation=dilation, groups=inp, bias=False),
        BatchNorm2d(inp),
        nn.ReLU6(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, padding=0, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def dilated_conv_3x3_bn(inp, oup, padding, dilation, BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, 1, padding=padding, dilation=dilation, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class _ASPPModule(nn.Module):
    def __init__(self, inp, planes, kernel_size, padding, dilation, batch_norm):
        super(_ASPPModule, self).__init__()
        BatchNorm2d = batch_norm
        if kernel_size == 1:
            self.atrous_conv = nn.Sequential(
                nn.Conv2d(inp, planes, kernel_size=1, stride=1, padding=padding, dilation=dilation, bias=False),
                BatchNorm2d(planes),
                nn.ReLU6(inplace=True)
            )
        elif kernel_size == 3:
            # we use depth-wise separable convolution to save the number of parameters
            self.atrous_conv = depth_sep_dilated_conv_3x3_bn(inp, planes, padding, dilation, BatchNorm2d)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inp, oup, output_stride=32, batch_norm=SynchronizedBatchNorm2d, width_mult=1.):
        super(ASPP, self).__init__()

        if output_stride == 32:
            dilations = [1, 2, 4, 8]
        elif output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        BatchNorm2d = batch_norm

        self.aspp1 = _ASPPModule(inp, int(256*width_mult), 1, padding=0, dilation=dilations[0], batch_norm=BatchNorm2d)
        self.aspp2 = _ASPPModule(inp, int(256*width_mult), 3, padding=dilations[1], dilation=dilations[1], batch_norm=BatchNorm2d)
        self.aspp3 = _ASPPModule(inp, int(256*width_mult), 3, padding=dilations[2], dilation=dilations[2], batch_norm=BatchNorm2d)
        self.aspp4 = _ASPPModule(inp, int(256*width_mult), 3, padding=dilations[3], dilation=dilations[3], batch_norm=BatchNorm2d)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inp, int(256*width_mult), 1, stride=1, padding=0, bias=False),
            BatchNorm2d(int(256*width_mult)),
            nn.ReLU6(inplace=True)
        )

        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(int(256*width_mult)*5, oup, 1, stride=1, padding=0, bias=False),
            BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )
        
        self.dropout = nn.Dropout(0.5)

        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='nearest')
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.bottleneck_conv(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()