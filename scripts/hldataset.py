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

import os
import random
import cv2
import numpy as np
from PIL import Image
from scipy import misc, ndimage

import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
from scipy.ndimage import morphology

# ignore warnings
import warnings
warnings.filterwarnings("ignore")


class Normalize(object):

    def __init__(self, scale, mean, std):
        self.scale = scale
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, alpha = sample['image'], sample['alpha']
        image, alpha = image.astype('float32'), alpha.astype('float32')

        image = (self.scale * image - self.mean) / self.std
        alpha[:, :, 0] = self.scale * alpha[:, :, 0]
        if alpha.shape[2] > 2:
            alpha[:, :, 2:11] = self.scale * alpha[:, :, 2:11]
        return {'image': image.astype('float32'), 'alpha': alpha.astype('float32')}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        pass

    def __call__(self, sample):
        image, alpha = sample['image'], sample['alpha']
        # swap color axis
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        alpha = alpha.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'alpha': torch.from_numpy(alpha)}


class AdobeImageMattingDataset(Dataset):
    """Adobe Image Matting dataset"""

    def __init__(self, data_file, data_dir, train=False, transform=None):
        self.datalist = [name.split('\t') for name in open(data_file).read().splitlines()]
        self.data_dir = data_dir
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        image_name = os.path.join(self.data_dir, self.datalist[idx][0])
        alpha_name = os.path.join(self.data_dir, self.datalist[idx][1])
        trimap_name = os.path.join(self.data_dir, self.datalist[idx][2])

        def read_image(x):
            img_arr = np.array(Image.open(x))
            if len(img_arr.shape) == 2:  # grayscale
                img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
            return img_arr

        image = read_image(image_name)
        alpha = np.array(Image.open(alpha_name))
        trimap = np.array(Image.open(trimap_name))
        
        alpha = alpha[:, :, 0] if alpha.ndim == 3 else alpha
        alpha = alpha.reshape(alpha.shape[0], alpha.shape[1], 1)
        trimap = trimap.reshape(trimap.shape[0], trimap.shape[1], 1)

        image = np.concatenate((image, trimap), axis=2)
        alpha = np.concatenate((alpha, trimap), axis=2)
            
        sample = {
            'image': image,
            'alpha': alpha
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
