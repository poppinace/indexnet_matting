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
import cv2
from time import time
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from hlvggnet import hlvgg16
from hlmobilenetv2 import hlmobilenetv2
from hldataset import AdobeImageMattingDataset, Normalize, ToTensor
from utils import *

IMG_SCALE = 1./255
IMG_MEAN = np.array([0.485, 0.456, 0.406, 0]).reshape((1, 1, 4))
IMG_STD = np.array([0.229, 0.224, 0.225, 1]).reshape((1, 1, 4))
DATA_DIR = '/media/hao/DATA/Combined_Dataset'
DATA_TEST_LIST = './lists/test.txt'

STRIDE = 32
RESTORE_FROM = './pretrained/indexnet_matting.pth.tar'
RESULT_DIR = './results/indexnet_matting'

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# instantiate network
net = hlmobilenetv2(
        pretrained=False,
        freeze_bn=True, 
        output_stride=STRIDE,
        apply_aspp=True,
        conv_operator='std_conv',
        decoder='indexnet',
        decoder_kernel_size=5,
        indexnet='depthwise',
        index_mode='m2o',
        use_nonlinear=True,
        use_context=True
    )
net = nn.DataParallel(net)
net.to(device)

try:
    checkpoint = torch.load(RESTORE_FROM)
    pretrained_dict = checkpoint['state_dict']
except:
    raise Exception('Please download the pretrained model!')
    
net.load_state_dict(pretrained_dict)
net.to(device)

dataset = AdobeImageMattingDataset
testset = dataset(
    data_file=DATA_TEST_LIST,
    data_dir=DATA_DIR,
    train=False,
    transform=transforms.Compose([
        Normalize(IMG_SCALE, IMG_MEAN, IMG_STD),
        ToTensor()]
    )
)
test_loader = DataLoader(
    testset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

image_list = [name.split('\t') for name in open(DATA_TEST_LIST).read().splitlines()]
# switch to eval mode
net.eval()

with torch.no_grad():
    sad = []
    mse = []
    grad = []
    conn = []
    avg_frame_rate = 0
    start = time()
    for i, sample in enumerate(test_loader):
        image, target = sample['image'], sample['alpha']

        h, w = image.size()[2:]
        image = image.squeeze().numpy().transpose(1, 2, 0)
        image = image_alignment(image, STRIDE, odd=False)
        inputs = torch.from_numpy(np.expand_dims(image.transpose(2, 0, 1), axis=0))
        
        # inference
        torch.cuda.synchronize()
        start = time()
        outputs = net(inputs.cuda()).squeeze().cpu().numpy()
        torch.cuda.synchronize()
        end = time()

        alpha = cv.resize(outputs, dsize=(w,h), interpolation=cv.INTER_CUBIC)
        alpha = np.clip(alpha, 0, 1) * 255.
        trimap = target[:, 1, :, :].squeeze().numpy()
        mask = np.equal(trimap, 128).astype(np.float32)

        alpha = (1 - mask) * trimap + mask * alpha
        gt_alpha = target[:, 0, :, :].squeeze().numpy() * 255.

        _, image_name = os.path.split(image_list[i][0])
        Image.fromarray(alpha.astype(np.uint8)).save(
            os.path.join(RESULT_DIR, image_name)
        )
        # Image.fromarray(gt_alpha.astype(np.uint8)).show()

        sad.append(compute_sad_loss(alpha, gt_alpha, mask))
        mse.append(compute_mse_loss(alpha, gt_alpha, mask))

        running_frame_rate = 1 * float(1 / (end - start)) # batch_size = 1
        avg_frame_rate = (avg_frame_rate*i + running_frame_rate)/(i+1)
        print(
            'test: {0}/{1}, sad: {2:.2f}, SAD: {3:.2f}, MSE: {4:.4f},'
            ' framerate: {5:.2f}Hz/{6:.2f}Hz'
            .format(i+1, len(test_loader), sad[-1], np.mean(sad), np.mean(mse),
            running_frame_rate, avg_frame_rate)
        )
         