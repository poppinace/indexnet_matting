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


import torch
import torch.nn as nn
import torch.nn.functional as F

class HolisticIndexBlock(nn.Module):
    def __init__(self, inp, use_nonlinear=False, use_context=False, batch_norm=None):
        super(HolisticIndexBlock, self).__init__()

        BatchNorm2d = batch_norm
        
        if use_context:
            kernel_size, padding = 4, 1
        else:
            kernel_size, padding = 2, 0

        if use_nonlinear:
            self.indexnet = nn.Sequential(
                nn.Conv2d(inp, 2*inp, kernel_size=kernel_size, stride=2, padding=padding, bias=False),
                BatchNorm2d(2*inp),
                nn.ReLU6(inplace=True),
                nn.Conv2d(2*inp, 4, kernel_size=1, stride=1, padding=0, bias=False)
            )
        else:
            self.indexnet = nn.Conv2d(inp, 4, kernel_size=kernel_size, stride=2, padding=padding, bias=False)

    def forward(self, x):
        x = self.indexnet(x)
        
        y = torch.sigmoid(x)
        z = F.softmax(y, dim=1)

        idx_en = F.pixel_shuffle(z, 2)
        idx_de = F.pixel_shuffle(y, 2)

        return idx_en, idx_de


class DepthwiseO2OIndexBlock(nn.Module):
    def __init__(self, inp, use_nonlinear=False, use_context=False, batch_norm=nn.BatchNorm2d):
        super(DepthwiseO2OIndexBlock, self).__init__()

        self.indexnet1 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        self.indexnet2 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        self.indexnet3 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        self.indexnet4 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        
    def _build_index_block(self, inp, use_nonlinear, use_context, BatchNorm2d):
        
        if use_context:
            kernel_size, padding = 4, 1
        else:
            kernel_size, padding = 2, 0

        if use_nonlinear:
            return nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=2, padding=padding, groups=inp, bias=False),
                BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0, groups=inp, bias=False)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=2, padding=padding, groups=inp, bias=False)
            )
        
    def forward(self, x):
        bs, c, h, w = x.size()

        x1 = self.indexnet1(x).unsqueeze(2)
        x2 = self.indexnet2(x).unsqueeze(2)
        x3 = self.indexnet3(x).unsqueeze(2)
        x4 = self.indexnet4(x).unsqueeze(2)

        x = torch.cat((x1, x2, x3, x4), dim=2)
        
        # normalization
        y = torch.sigmoid(x)
        z = F.softmax(y, dim=2)
        # pixel shuffling
        y = y.view(bs, c*4, int(h/2), int(w/2))
        z = z.view(bs, c*4, int(h/2), int(w/2))
        idx_en = F.pixel_shuffle(z, 2)
        idx_de = F.pixel_shuffle(y, 2)

        return idx_en, idx_de


class DepthwiseM2OIndexBlock(nn.Module):
    def __init__(self, inp, use_nonlinear=False, use_context=False, batch_norm=nn.BatchNorm2d):
        super(DepthwiseM2OIndexBlock, self).__init__()
        self.use_nonlinear = use_nonlinear

        self.indexnet1 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        self.indexnet2 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        self.indexnet3 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        self.indexnet4 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        
    def _build_index_block(self, inp, use_nonlinear, use_context, BatchNorm2d):
        
        if use_context:
            kernel_size, padding = 4, 1
        else:
            kernel_size, padding = 2, 0

        if use_nonlinear:
            return nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=2, padding=padding, bias=False),
                BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0, bias=False)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=2, padding=padding, bias=False)
            )
        
    def forward(self, x):
        bs, c, h, w = x.size()

        x1 = self.indexnet1(x).unsqueeze(2)
        x2 = self.indexnet2(x).unsqueeze(2)
        x3 = self.indexnet3(x).unsqueeze(2)
        x4 = self.indexnet4(x).unsqueeze(2)

        x = torch.cat((x1, x2, x3, x4), dim=2)
        
        # normalization
        y = torch.sigmoid(x)
        z = F.softmax(y, dim=2)
        # pixel shuffling
        y = y.view(bs, c*4, int(h/2), int(w/2))
        z = z.view(bs, c*4, int(h/2), int(w/2))
        idx_en = F.pixel_shuffle(z, 2)
        idx_de = F.pixel_shuffle(y, 2)

        return idx_en, idx_de