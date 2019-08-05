"""
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
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.nn import SynchronizedBatchNorm2d

from hlconv import hlconv

class DeepLabDecoder(nn.Module):
    def __init__(self, conv_operator='std_conv', kernel_size=5, batch_norm=SynchronizedBatchNorm2d):
        super(DeepLabDecoder, self).__init__()
        hlConv2d = hlconv[conv_operator]
        BatchNorm2d = batch_norm

        self.first_dconv = nn.Sequential(
            nn.Conv2d(24, 48, 1, bias=False),
            BatchNorm2d(48),
            nn.ReLU6(inplace=True)
        )

        self.last_dconv = nn.Sequential(
            hlConv2d(304, 256, kernel_size, 1, BatchNorm2d),
            hlConv2d(256, 256, kernel_size, 1, BatchNorm2d)
        )

        self._init_weight()

    def forward(self, l, l_low):
        l_low = self.first_dconv(l_low)
        l = F.interpolate(l, size=l_low.size()[2:], mode='bilinear', align_corners=True)
        l = torch.cat((l, l_low), dim=1)
        l = self.last_dconv(l)
        return l

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


# max-pooling indices-guided decoding
class IndexedDecoder(nn.Module):
    def __init__(self, inp, oup, conv_operator='std_conv', kernel_size=5, batch_norm=SynchronizedBatchNorm2d):
        super(IndexedDecoder, self).__init__()
        hlConv2d = hlconv[conv_operator]
        BatchNorm2d = batch_norm

        self.upsample = nn.MaxUnpool2d((2, 2), stride=2)
        # inp, oup, kernel_size, stride, batch_norm
        self.dconv = hlConv2d(inp, oup, kernel_size, 1, BatchNorm2d)

        self._init_weight()

    def forward(self, l_encode, l_low, indices=None):
        l_encode = self.upsample(l_encode, indices) if indices is not None else l_encode
        l_encode = torch.cat((l_encode, l_low), dim=1)    
        return self.dconv(l_encode)

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
    
    def visualize(self, x, indices=None):
        l = self.upsample(x, indices) if indices is not None else x
        l = l.mean(dim=1).squeeze()
        l = l.cpu().numpy()
        l = l / l.max() * 255.
        plt.figure()
        plt.imshow(l, cmap='viridis')
        plt.show()


class IndexedUpsamlping(nn.Module):
    def __init__(self, inp, oup, conv_operator='std_conv', kernel_size=5, batch_norm=SynchronizedBatchNorm2d):
        super(IndexedUpsamlping, self).__init__()
        self.oup = oup

        hlConv2d = hlconv[conv_operator]
        BatchNorm2d = batch_norm

        # inp, oup, kernel_size, stride, batch_norm
        self.dconv = hlConv2d(inp, oup, kernel_size, 1, BatchNorm2d)

        self._init_weight()

    def forward(self, l_encode, l_low, indices=None):
        _, c, _, _ = l_encode.size()
        if indices is not None:
            l_encode = indices * F.interpolate(l_encode, size=l_low.size()[2:], mode='nearest')
        l_cat = torch.cat((l_encode, l_low), dim=1)
        return self.dconv(l_cat)

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
    
    def visualize(self, x, indices=None):
        l = self.upsample(x, indices) if indices is not None else x
        l = l.mean(dim=1).squeeze()
        l = l.detach().cpu().numpy()
        l = l / l.max() * 255.
        plt.figure()
        plt.imshow(l, cmap='viridis')
        plt.axis('off')
        plt.show()
