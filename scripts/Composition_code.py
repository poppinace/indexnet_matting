##Copyright 2017 Adobe Systems Inc.
##
##Licensed under the Apache License, Version 2.0 (the "License");
##you may not use this file except in compliance with the License.
##You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
##Unless required by applicable law or agreed to in writing, software
##distributed under the License is distributed on an "AS IS" BASIS,
##WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##See the License for the specific language governing permissions and
##limitations under the License.


##############################################################
#Set your paths here

#path to provided foreground images
fg_path = 'fg/'

#path to provided alpha mattes
a_path = 'alpha/'

#Path to background images (MSCOCO)
bg_path = 'train2014/'

#Path to folder where you want the composited images to go
out_path = 'merged_cv/'

##############################################################

import numpy as np
from PIL import Image
import os 
import math
import time
import cv2 as cv

def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg = np.array(bg[0:h, 0:w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    comp = alpha * fg + (1 - alpha) * bg
    comp = comp.astype(np.uint8)
    return comp

num_bgs = 100

fg_files = [name for name in open('training_fg_names.txt').read().splitlines()]
bg_files = [name for name in open('training_bg_names.txt').read().splitlines()]

bg_iter = iter(bg_files)
for k, im_name in enumerate(fg_files):
    
    im = cv.imread(fg_path + im_name)
    a = cv.imread(a_path + im_name, 0)
    h, w = im.shape[:2]
    
    bcount = 0
    for i in range(num_bgs):

        bg_name = next(bg_iter)        
        bg = cv.imread(bg_path + bg_name)
        bh, bw = bg.shape[:2]
        wratio = float(w) / float(bw)
        hratio = float(h) / float(bh)
        ratio = wratio if wratio > hratio else hratio     
        if ratio > 1:        
            bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)
                     
        out = composite4(im, bg, a, w, h)   
        filename = out_path + im_name[:len(im_name)-4] + '_' + str(bcount) + '.png'
        
        cv.imwrite(filename, out, [cv.IMWRITE_PNG_COMPRESSION, 9])        
        
        bcount += 1
        print(k*num_bgs + bcount)
