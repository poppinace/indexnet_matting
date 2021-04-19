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
import argparse
from time import time

import cv2 as cv
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from lib.nn import patch_replication_callback
from hlmobilenetv2 import hlmobilenetv2
from hlvggnet import hlvgg16
from hldataset import AdobeImageMattingDataset, RandomCrop, RandomFlip, Normalize, ToTensor
from utils import *

# prevent dataloader deadlock
cv.setNumThreads(0)

# constant
IMG_SCALE = 1./255
IMG_MEAN = np.array([0.485, 0.456, 0.406, 0]).reshape((1, 1, 4))
IMG_STD = np.array([0.229, 0.224, 0.225, 1]).reshape((1, 1, 4))
SCALES = [1, 1.5, 2]

# system-io-related parameters
DATASET = 'Adobe_Image_Matting'
DATA_DIR = '/media/hao/DATA/Combined_Dataset'
EXP = 'indexnet_matting'
DATA_LIST = './lists/train.txt'
DATA_VAL_LIST = './lists/test.txt'
RESTORE_FROM = 'model_ckpt.pth.tar'
SNAPSHOT_DIR = './snapshots'
RESULT_DIR = './results'

# model-related parameters
OUTPUT_STRIDE = 32
CONV_OPERATOR = 'std_conv' # choose in ['std_conv', 'dep_sep_conv']
DECODER = 'indexnet' # choose in ['unet_style', 'deeplabv3+', 'refinenet', 'indexnet']
DECODER_KERNEL_SIZE = 5
BACKBONE = 'mobilenetv2' # choose in ['mobilenetv2', 'vgg16']
INDEXNET = 'depthwise' # choose in ['holistic', 'depthwise']
INDEX_MODE = 'm2o' # choose in ['o2o', 'm2o']

#---------------------------------------------------------------------------------
# TIPS:
# for deeplabv3+ and refinenet, we expect the input size is odd, say 321, to be
# consistent with the behaviour of bilinear upsamping, while for vgg16 or our
# modified mobilenetv2, the input size should be even number, say 320. This
# facilitates the corner alignment between the image and the feature map
#---------------------------------------------------------------------------------
# training-related parameters
BATCH_SIZE = 2
CROP_SIZE = 320
LEARNING_RATE = 1e-5
MOMENTUM = 0.9
MULT = 100
NUM_EPOCHS = 100
NUM_CPU_WORKERS = 0
PRINT_EVERY = 1
RANDOM_SEED = 6
WEIGHT_DECAY = 1e-4
RECORD_EVERY = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hlbackbone = {
    'mobilenetv2': hlmobilenetv2,
    'vgg16': hlvgg16
}

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Deep-Image-Matting")
    # constant
    parser.add_argument("--image-scale", type=float, default=IMG_SCALE, help="Scale factor used in normalization.")
    parser.add_argument("--image-mean", type=float, default=IMG_MEAN, help="Mean used in normalization.")
    parser.add_argument("--image-std", type=float, default=IMG_STD, help="Std used in normalization.")
    parser.add_argument("--scales", type=int, default=SCALES, help="Scales of crop.")
    # system-related parameters                    
    parser.add_argument("--dataset", type=str, default=DATASET, help="Dataset type.")
    parser.add_argument("--exp", type=str, default=EXP, help="Experiment path.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Path to the directory containing the dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST, help="Path to the file listing the images in the dataset.")
    parser.add_argument("--data-val-list", type=str, default=DATA_VAL_LIST, help="Path to the file listing the images in the val dataset.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM, help="Where restore model parameters from.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR, help="Where to save snapshots of the model.")
    parser.add_argument("--result-dir", type=str, default=RESULT_DIR, help="Where to save inferred results.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED, help="Random seed to have reproducible results.")
    parser.add_argument("--evaluate-only", action="store_true", help="Whether to perform evaluation.")
    # model-related parameters
    parser.add_argument("--output-stride", type=int, default=OUTPUT_STRIDE, help="Output stride of the model.")
    parser.add_argument("--conv-operator", type=str, default=CONV_OPERATOR, help="Convolutional operator used in decoder.")
    parser.add_argument("--backbone", type=str, default=BACKBONE, help="Backbone used.")
    parser.add_argument("--decoder", type=str, default=DECODER, help="Decoder style.")
    parser.add_argument("--decoder-kernel-size", type=int, default=DECODER_KERNEL_SIZE, help="Decoder kernel size.")
    parser.add_argument("--indexnet", type=str, default=INDEXNET, choices=['holistic', 'depthwise'], help="Use holistic or depthwise index networks.")
    parser.add_argument("--index-mode", type=str, default=INDEX_MODE, choices=['o2o', 'm2o'], help="Type of depthwise index network.")
    parser.add_argument("--use-nonlinear", action="store_true", help="Whether to use nonlinearity in IndexNet.")
    parser.add_argument("--use-context", action="store_true", help="Whether to use context in IndexNet.")
    parser.add_argument("--apply-aspp", action="store_true", help="Whether to perform ASPP.")
    parser.add_argument("--sync-bn", action="store_true", help="Whether to apply synchronized batch normalization.")
    # training-related parameters
    parser.add_argument("--crop-size", type=int, default=CROP_SIZE, help="Size of crop.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Number of images sent to the network in one step.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Base learning rate for training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM, help="Momentum component of the optimiser.")
    parser.add_argument("--mult", type=float, default=MULT, help="LR multiplier for pretrained layers.")
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS, help="Number of training steps.")
    parser.add_argument("--num-workers", type=int, default=NUM_CPU_WORKERS, help="Number of CPU cores used.")
    parser.add_argument("--print-every", type=int, default=PRINT_EVERY, help="Print information every often.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="Regularisation parameter for L2-loss.")
    parser.add_argument("--record-every", type=int, default=RECORD_EVERY, help="Record loss every often.")
    return parser.parse_args()

def save_checkpoint(state, snapshot_dir, filename='.pth.tar'):
    torch.save(state, '{}/{}'.format(snapshot_dir, filename))

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# implementation of the composition loss and alpha loss
def weighted_loss(pd, gt, wl=0.5, epsilon=1e-6):
    bs, _, h, w = pd.size()
    mask = gt[:, 1, :, :].view(bs, 1, h, w)
    alpha_gt = gt[:, 0, :, :].view(bs, 1, h, w)
    diff_alpha = (pd - alpha_gt) * mask
    loss_alpha = torch.sqrt(diff_alpha * diff_alpha + epsilon ** 2)
    loss_alpha = loss_alpha.sum(dim=2).sum(dim=2) / mask.sum(dim=2).sum(dim=2)
    loss_alpha = loss_alpha.sum() / bs

    fg = gt[:, 2:5, :, :]
    bg = gt[:, 5:8, :, :]
    c_p = pd * fg + (1 - pd) * bg
    c_g = gt[:, 8:11, :, :]
    diff_color = (c_p - c_g) * mask
    loss_composition = torch.sqrt(diff_color * diff_color + epsilon ** 2)
    loss_composition = loss_composition.sum(dim=2).sum(dim=2) / mask.sum(dim=2).sum(dim=2)
    loss_composition = loss_composition.sum() / bs

    return wl * loss_alpha + (1 - wl) * loss_composition

def train(net, train_loader, optimizer, epoch, scheduler, args):
    # switch to train mode
    net.train()
    
    running_loss = 0.0
    avg_frame_rate = 0.0
    start = time()
    for i, sample in enumerate(train_loader):
        inputs, targets = sample['image'], sample['alpha']
        inputs, targets = inputs.cuda(), targets.cuda()
        # forward
        outputs = net(inputs)
        # zero the parameter gradients
        optimizer.zero_grad()
        # compute loss
        loss = weighted_loss(outputs, targets)
        # backward + optimize
        loss.backward()
        optimizer.step()
        # collect and print statistics
        running_loss += loss.item()

        end = time()
        running_frame_rate = args.batch_size * float(1 / (end - start))
        avg_frame_rate = (avg_frame_rate*i + running_frame_rate)/(i+1)
        if i % args.record_every == args.record_every-1:
            net.train_loss['running_loss'].append(running_loss / (i+1))
        if i % args.print_every == args.print_every-1:
            print('epoch: %d, train: %d/%d, '
                  'loss: %.5f, frame: %.2fHz/%.2fHz' % (
                      epoch,
                      i+1,
                      len(train_loader),
                      running_loss / (i+1),
                      running_frame_rate,
                      avg_frame_rate
                  ))
        start = time()
    net.train_loss['epoch_loss'].append(running_loss / (i+1))


def validate(net, val_loader, epoch, args):
    # switch to eval mode
    net.eval()
    
    image_list = [name.split('\t') for name in open(args.data_val_list).read().splitlines()]
    
    epoch_result_dir = os.path.join(args.result_dir, str(epoch))
    if not os.path.exists(epoch_result_dir):
        os.makedirs(epoch_result_dir)

    with torch.no_grad():
        sad = []
        mse = []
        grad = []
        conn = []
        avg_frame_rate = 0.0
        # scale = 0.5
        stride = args.output_stride
        start = time()
        for i, sample in enumerate(val_loader):
            image, targets = sample['image'], sample['alpha']
            
            h, w = image.size()[2:]
            image = image.squeeze().numpy().transpose(1, 2, 0)
            # image = image_rescale(image, scale)
            image = image_alignment(image, stride, odd=args.crop_size%2==1)
            inputs = torch.from_numpy(np.expand_dims(image.transpose(2, 0, 1), axis=0))
            
            # inference
            outputs = net(inputs.cuda()).squeeze().cpu().numpy()

            alpha = cv.resize(outputs, dsize=(w,h), interpolation=cv.INTER_CUBIC)
            alpha = np.clip(alpha, 0, 1) * 255.
            trimap = targets[:, 1, :, :].squeeze().numpy()
            mask = np.equal(trimap, 128).astype(np.float32)

            alpha = (1 - mask) * trimap + mask * alpha
            gt_alpha = targets[:, 0, :, :].squeeze().numpy() * 255.

            _, image_name = os.path.split(image_list[i][0])
            Image.fromarray(alpha.astype(np.uint8)).save(
                os.path.join(epoch_result_dir, image_name)
            )
            # Image.fromarray(alpha.astype(np.uint8)).show()

            # compute loss
            sad.append(compute_sad_loss(alpha, gt_alpha, mask))
            mse.append(compute_mse_loss(alpha, gt_alpha, mask))
            grad.append(compute_gradient_loss(alpha, gt_alpha, mask))
            conn.append(compute_connectivity_loss(alpha, gt_alpha, mask))

            end = time()
            running_frame_rate = 1 * float(1 / (end - start)) # batch_size = 1
            avg_frame_rate = (avg_frame_rate*i + running_frame_rate)/(i+1)
            if i % args.print_every == args.print_every - 1:
                print(
                    'epoch: {0}, test: {1}/{2}, sad: {3:.2f}, SAD: {4:.2f}, MSE: {5:.4f}, '
                    'Grad: {6:.2f}, Conn: {7:.2f}, frame: {8:.2f}Hz/{9:.2f}Hz'
                    .format(epoch, i+1, len(val_loader), sad[-1], np.mean(sad), np.mean(mse),
                    np.mean(grad), np.mean(conn), running_frame_rate, avg_frame_rate)
                )
            start = time()
    # write to files        
    with open(os.path.join(args.result_dir, args.exp+'.txt'), 'a') as f:
        print(
            'epoch: {0}, test: {1}/{2}, SAD: {3:.2f}, MSE: {4:.4f}, Grad: {5:.2f}, Conn: {6:.2f}'
            .format(epoch, i+1, len(val_loader), np.mean(sad), np.mean(mse), np.mean(grad), np.mean(conn)), 
            file=f
        )
    # save stats
    net.val_loss['epoch_loss'].append(np.mean(sad))
    net.measure['sad'].append(np.mean(sad))
    net.measure['mse'].append(np.mean(mse))
    net.measure['grad'].append(np.mean(grad))
    net.measure['conn'].append(np.mean(conn))


def main():
    args = get_arguments()
    
    # seeding for reproducbility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    # fix random seed bugs in numpy
    # np.random.seed(args.random_seed) 

    # instantiate dataset
    dataset = AdobeImageMattingDataset
    
    snapshot_dir = os.path.join(args.snapshot_dir, args.dataset.lower(), args.exp)
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    args.result_dir = os.path.join(args.result_dir, args.exp)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    args.restore_from = os.path.join(args.snapshot_dir, args.dataset.lower(), args.exp, args.restore_from)

    arguments = vars(args)
    for item in arguments:
        print(item, ':\t' , arguments[item])

    # instantiate network
    hlnet = hlbackbone[args.backbone]
    net = hlnet(
        pretrained=True,
        freeze_bn=True, 
        output_stride=args.output_stride, 
        input_size=args.crop_size, 
        apply_aspp=args.apply_aspp,
        conv_operator=args.conv_operator,
        decoder=args.decoder,
        decoder_kernel_size=args.decoder_kernel_size,
        indexnet=args.indexnet,
        index_mode=args.index_mode,
        use_nonlinear=args.use_nonlinear,
        use_context=args.use_context,
        sync_bn=args.sync_bn
    )
    
    if args.backbone == 'mobilenetv2':
        net = nn.DataParallel(net)
    if args.sync_bn:
        patch_replication_callback(net)
    net.cuda()
    
    # filter parameters
    pretrained_params = []
    learning_params = []
    for p in net.named_parameters():
        if 'dconv' in p[0] or 'pred' in p[0] or 'index' in p[0]:
            learning_params.append(p[1])
        else:
            pretrained_params.append(p[1])

    # define optimizer
    optimizer = torch.optim.Adam(
        [
            {'params': learning_params},
            {'params': pretrained_params, 'lr': args.learning_rate / args.mult},
        ],
        lr=args.learning_rate
    )

    # restore parameters
    start_epoch = 0
    net.train_loss = {
        'running_loss': [],
        'epoch_loss': []
    }
    net.val_loss = {
        'running_loss': [],
        'epoch_loss': []
    }
    net.measure = {
        'sad': [],
        'mse': [],
        'grad': [],
        'conn': []
    }
    if args.restore_from is not None:
        if os.path.isfile(args.restore_from):
            checkpoint = torch.load(args.restore_from)
            net.load_state_dict(checkpoint['state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'train_loss' in checkpoint:
                net.train_loss = checkpoint['train_loss']
            if 'val_loss' in checkpoint:
                net.val_loss = checkpoint['val_loss']
            if 'measure' in checkpoint:
                net.measure = checkpoint['measure']
            print("==> load checkpoint '{}' (epoch {})"
                  .format(args.restore_from, start_epoch))
        else:
            with open(os.path.join(args.result_dir, args.exp+'.txt'), 'a') as f:
                for item in arguments:
                    print(item, ':\t' , arguments[item], file=f)
            print("==> no checkpoint found at '{}'".format(args.restore_from))

    # define transform
    transform_train_val = [
        RandomCrop(args.crop_size, args.scales),
        RandomFlip()
    ]
    transform_all = [
        Normalize(args.image_scale, args.image_mean, args.image_std),
        ToTensor()
    ]
    composed_transform_train = transforms.Compose(transform_train_val + transform_all)
    composed_transform_val = transforms.Compose(transform_all)

    # define dataset loader
    trainset = dataset(
        data_file=args.data_list,
        data_dir=args.data_dir,
        train=True,
        transform=composed_transform_train
    )
    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        drop_last=True
    )
    valset = dataset(
        data_file=args.data_val_list,
        data_dir=args.data_dir,
        train=False,
        transform=composed_transform_val
    )
    val_loader = DataLoader(
        valset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print('alchemy start...')
    if args.evaluate_only:
        validate(net, val_loader, start_epoch+1, args)
        return
    
    resume_epoch = -1 if start_epoch == 0 else start_epoch
    scheduler = MultiStepLR(optimizer, milestones=[20, 26], gamma=0.1, last_epoch=resume_epoch)
    for epoch in range(start_epoch, args.num_epochs):
        scheduler.step()
        np.random.seed()
        # train
        train(net, train_loader, optimizer, epoch+1, scheduler, args)
        # val
        validate(net, val_loader, epoch+1, args)
        # save checkpoint
        state = {
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch+1,
            'train_loss': net.train_loss,
            'val_loss': net.val_loss,
            'measure': net.measure
        }
        save_checkpoint(state, snapshot_dir, filename='model_ckpt.pth.tar')
        print(args.exp+' epoch {} finished!'.format(epoch+1))
        if len(net.measure['grad']) > 1 and net.measure['grad'][-1] <= min(net.measure['grad'][:-1]):
            save_checkpoint(state, snapshot_dir, filename='model_best.pth.tar')
    print('Experiments with '+args.exp+' done!')

if __name__ == "__main__":
    main()