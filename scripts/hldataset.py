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

def resize_image_alpha(image, alpha, nh, nw):
    alpha_chn = alpha.shape[2]

    trimap = image[:, :, 3]
    image = image[:, :, 0:3]
    mask = alpha[:, :, 1]
    if alpha_chn > 2:
        fg = alpha[:, :, 2:5]
        bg = alpha[:, :, 5:8]
        ori_image = alpha[:, :, 8:11]
    alpha = alpha[:, :, 0]

    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    trimap = cv2.resize(trimap, (nw, nh), interpolation=cv2.INTER_NEAREST)
    alpha = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    if alpha_chn > 2:
        fg = cv2.resize(fg, (nw, nh), interpolation=cv2.INTER_CUBIC)
        bg = cv2.resize(bg, (nw, nh), interpolation=cv2.INTER_CUBIC)
        ori_image = cv2.resize(ori_image, (nw, nh), interpolation=cv2.INTER_CUBIC)

    trimap = trimap.reshape(trimap.shape[0], trimap.shape[1], 1)
    alpha = alpha.reshape(alpha.shape[0], alpha.shape[1], 1)
    mask = mask.reshape(mask.shape[0], mask.shape[1], 1)

    image = np.concatenate((image, trimap), axis=2)
    if alpha_chn > 2:
        alpha = np.concatenate((alpha, mask, fg, bg, ori_image), axis=2)
    else:
        alpha = np.concatenate((alpha, mask), axis=2)

    return image, alpha


class RandomCrop(object):
    """Crop randomly the image

    Args:
        output_size (int): Desired output size. If int, square crop
            is made.
        scales (list): Desired scales
    """

    def __init__(self, output_size, scales):
        assert isinstance(output_size, int)

        self.output_size = output_size
        self.scales = scales

    def __call__(self, sample):
        image, alpha = sample['image'], sample['alpha']
        h, w = image.shape[:2]

        if min(h, w) < self.output_size:
            s = (self.output_size + 180) / min(h, w)
            nh, nw = int(np.floor(h * s)), int(np.floor(w * s))
            image, alpha = resize_image_alpha(image, alpha, nh, nw)
            h, w = image.shape[:2]

        crop_size = np.floor(self.output_size * np.array(self.scales)).astype('int')
        crop_size = crop_size[crop_size < min(h, w)]
        crop_size = int(random.choice(crop_size))

        c = int(np.ceil(crop_size / 2))
        mask = np.equal(image[:, :, 3], 128).astype(np.uint8)
        if mask[c:h-c+1, c:w-c+1].sum() != 0:
            mask_center = np.zeros((h, w), dtype=np.uint8)
            mask_center[c:h-c+1, c:w-c+1] = 1
            mask = (mask & mask_center)
            idh, idw = np.where(mask == 1)
            ids = random.choice(range(len(idh)))
            hc, wc = idh[ids], idw[ids]
            h1, w1 = hc-c, wc-c
        else:
            idh, idw = np.where(mask == 1)
            ids = random.choice(range(len(idh)))
            hc, wc = idh[ids], idw[ids]
            h1, w1 = np.clip(hc-c, 0, h), np.clip(wc-c, 0, w)
            h2, w2 = h1+crop_size, w1+crop_size
            h1 = h-crop_size if h2 > h else h1
            w1 = w-crop_size if w2 > w else w1

        image = image[h1:h1+crop_size, w1:w1+crop_size, :]
        alpha = alpha[h1:h1+crop_size, w1:w1+crop_size, :]

        if crop_size != self.output_size:
            nh = nw = self.output_size
            image, alpha = resize_image_alpha(image, alpha, nh, nw)

        return {'image': image, 'alpha': alpha}


class RandomFlip(object):

    def __init__(self):
        pass

    def __call__(self, sample):
        image, alpha = sample['image'], sample['alpha']
        do_mirror = np.random.randint(2)
        if do_mirror:
            image = cv2.flip(image, 1)
            alpha = cv2.flip(alpha, 1)
        return {'image': image, 'alpha': alpha}


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
        # self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def __len__(self):
        return len(self.datalist)

    def generate_trimap(self, alpha):
        # alpha \in [0, 1] should be taken into account
        # be careful when dealing with regions of alpha=0 and alpha=1
        fg = np.array(np.equal(alpha, 255).astype(np.float32))
        unknown = np.array(np.not_equal(alpha, 0).astype(np.float32)) # unknown = alpha > 0
        unknown = unknown - fg
        # image dilation implemented by Euclidean distance transform
        unknown = morphology.distance_transform_edt(unknown==0) <= np.random.randint(1, 20)
        trimap = fg * 255
        trimap[unknown] = 128
        return trimap.astype(np.uint8)

    def __getitem__(self, idx):
        image_name = os.path.join(self.data_dir, self.datalist[idx][0])
        alpha_name = os.path.join(self.data_dir, self.datalist[idx][1])
        if self.train:
            fg_name = os.path.join(self.data_dir, self.datalist[idx][2])
            bg_name = os.path.join(self.data_dir, self.datalist[idx][3])
        else:
            trimap_name = os.path.join(self.data_dir, self.datalist[idx][2])

        def read_image(x):
            img_arr = np.array(Image.open(x))
            if len(img_arr.shape) == 2:  # grayscale
                img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
            return img_arr

        image = read_image(image_name)
        alpha = np.array(Image.open(alpha_name))
        if self.train:
            fg = read_image(fg_name)
            bg = read_image(bg_name)
            fgh, fgw = fg.shape[0:2]
            bgh, bgw = bg.shape[0:2]
            rh, rw = fgh/float(bgh), fgw/float(bgw)
            r =  rh if rh > rw else rw
            nh, nw = int(np.ceil(bgh*r)), int(np.ceil(bgw*r))
            bg = cv2.resize(bg, (nw, nh), interpolation=cv2.INTER_CUBIC)
            bg = bg[0:fgh, 0:fgw, :]

            trimap = self.generate_trimap(alpha)
            mask = np.equal(trimap, 128).astype(np.uint8)

            alpha = alpha.reshape(alpha.shape[0], alpha.shape[1], 1)
            trimap = trimap.reshape(trimap.shape[0], trimap.shape[1], 1)
            mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
            
            alpha = np.concatenate((alpha, mask, fg, bg, image), axis=2)
            image = np.concatenate((image, trimap), axis=2)
        else:
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


if __name__ == "__main__":
    IMG_SCALE = 1./255
    IMG_MEAN = np.array([0.485, 0.456, 0.406, 0]).reshape((1, 1, 4))
    IMG_STD = np.array([0.229, 0.224, 0.225, 1]).reshape((1, 1, 4))

    dataset = AdobeImageMattingDataset(
        data_file='train.txt',
        data_dir='/media/hao/DATA/Combined_Dataset',
        train=True,
        transform=transforms.Compose([
            RandomCrop(320, [1, 1.5, 2]),
            RandomFlip(),
            Normalize(IMG_SCALE, IMG_MEAN, IMG_STD),
            ToTensor()]
        )
    )
    datasetloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    print(len(datasetloader))
    for i, data in enumerate(datasetloader, 0):
        images, targets = data['image'], data['alpha']
        print(i)