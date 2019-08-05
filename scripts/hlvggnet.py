"""
Implementation of Deep Image Matting @ CVPR2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

MODEL_URLS = {
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}

CORRESP_NAME = {
    # conv1
    "features.0.weight": "conv11.weight",
    "features.0.bias": "conv11.bias",
    "features.1.weight": "bn11.weight",
    "features.1.bias": "bn11.bias",
    "features.1.running_mean": "bn11.running_mean",
    "features.1.running_var": "bn11.running_var",

    "features.3.weight": "conv12.weight",
    "features.3.bias": "conv12.bias",
    "features.4.weight": "bn12.weight",
    "features.4.bias": "bn12.bias",
    "features.4.running_mean": "bn12.running_mean",
    "features.4.running_var": "bn12.running_var",
    # conv2
    "features.7.weight": "conv21.weight",
    "features.7.bias": "conv21.bias",
    "features.8.weight": "bn21.weight",
    "features.8.bias": "bn21.bias",
    "features.8.running_mean": "bn21.running_mean",
    "features.8.running_var": "bn21.running_var",

    "features.10.weight": "conv22.weight",
    "features.10.bias": "conv22.bias",
    "features.11.weight": "bn22.weight",
    "features.11.bias": "bn22.bias",
    "features.11.running_mean": "bn22.running_mean",
    "features.11.running_var": "bn22.running_var",
    # conv3
    "features.14.weight": "conv31.weight",
    "features.14.bias": "conv31.bias",
    "features.15.weight": "bn31.weight",
    "features.15.bias": "bn31.bias",
    "features.15.running_mean": "bn31.running_mean",
    "features.15.running_var": "bn31.running_var",

    "features.17.weight": "conv32.weight",
    "features.17.bias": "conv32.bias",
    "features.18.weight": "bn32.weight",
    "features.18.bias": "bn32.bias",
    "features.18.running_mean": "bn32.running_mean",
    "features.18.running_var": "bn32.running_var",

    "features.20.weight": "conv33.weight",
    "features.20.bias": "conv33.bias",
    "features.21.weight": "bn33.weight",
    "features.21.bias": "bn33.bias",
    "features.21.running_mean": "bn33.running_mean",
    "features.21.running_var": "bn33.running_var",
    # conv4
    "features.24.weight": "conv41.weight",
    "features.24.bias": "conv41.bias",
    "features.25.weight": "bn41.weight",
    "features.25.bias": "bn41.bias",
    "features.25.running_mean": "bn41.running_mean",
    "features.25.running_var": "bn41.running_var",

    "features.27.weight": "conv42.weight",
    "features.27.bias": "conv42.bias",
    "features.28.weight": "bn42.weight",
    "features.28.bias": "bn42.bias",
    "features.28.running_mean": "bn42.running_mean",
    "features.28.running_var": "bn42.running_var",

    "features.30.weight": "conv43.weight",
    "features.30.bias": "conv43.bias",
    "features.31.weight": "bn43.weight",
    "features.31.bias": "bn43.bias",
    "features.31.running_mean": "bn43.running_mean",
    "features.31.running_var": "bn43.running_var",

    # conv5
    "features.34.weight": "conv51.weight",
    "features.34.bias": "conv51.bias",
    "features.35.weight": "bn51.weight",
    "features.35.bias": "bn51.bias",
    "features.35.running_mean": "bn51.running_mean",
    "features.35.running_var": "bn51.running_var",

    "features.37.weight": "conv52.weight",
    "features.37.bias": "conv52.bias",
    "features.38.weight": "bn52.weight",
    "features.38.bias": "bn52.bias",
    "features.38.running_mean": "bn52.running_mean",
    "features.38.running_var": "bn52.running_var",

    "features.40.weight": "conv53.weight",
    "features.40.bias": "conv53.bias",
    "features.41.weight": "bn53.weight",
    "features.41.bias": "bn53.bias",
    "features.41.running_mean": "bn53.running_mean",
    "features.41.running_var": "bn53.running_var",
    # fc6
    "classifier.0.weight": "conv6.weight",
    "classifier.0.bias": "conv6.bias",
    # # fc7
    # "classifier.3.weight": "conv7.weight",
    # "classifier.3.bias": "conv7.bias",
    # # classifier
    # "classifier.6.weight": "classifier.weight",
    # "classifier.6.bias": "classifier.bias"
}


class DeepMatting(nn.Module):
    def __init__(self, input_chn, output_chn, use_pretrained=True):
        super(DeepMatting, self).__init__()
        self.input_chn = input_chn

        # encoding
        self.conv11 = nn.Conv2d(input_chn, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((2, 2), stride=2, return_indices=True)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d((2, 2), stride=2, return_indices=True)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d((2, 2), stride=2, return_indices=True)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d((2, 2), stride=2, return_indices=True)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d((2, 2), stride=2, return_indices=True)

        self.conv6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)

        self.freeze_bn()

        # decoding
        self.dconv6 = nn.Conv2d(4096, 512, kernel_size=1, padding=0)

        self.unpool5 = nn.MaxUnpool2d((2, 2), stride=2)
        self.dconv5 = nn.Conv2d(512, 512, kernel_size=5, padding=2)

        self.unpool4 = nn.MaxUnpool2d((2, 2), stride=2)
        self.dconv4 = nn.Conv2d(512, 256, kernel_size=5, padding=2)

        self.unpool3 = nn.MaxUnpool2d((2, 2), stride=2)
        self.dconv3 = nn.Conv2d(256, 128, kernel_size=5, padding=2)

        self.unpool2 = nn.MaxUnpool2d((2, 2), stride=2)
        self.dconv2 = nn.Conv2d(128, 64, kernel_size=5, padding=2)

        self.unpool1 = nn.MaxUnpool2d((2, 2), stride=2)
        self.dconv1 = nn.Conv2d(64, 64, kernel_size=5, padding=2)

        self.alpha_pred = nn.Conv2d(64, 1, kernel_size=5, padding=2)
        
        # weights initialization
        self.weights_init_random()

    def forward(self, x):
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1p, idx1p = self.pool1(x12)

        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2p, idx2p = self.pool2(x22)

        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3p, idx3p = self.pool3(x33)

        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4p, idx4p = self.pool4(x43)

        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5p, idx5p = self.pool5(x53)

        x6 = F.relu(self.conv6(x5p))

        x6d = F.relu(self.dconv6(x6))

        x5d = self.unpool5(x6d, indices=idx5p)
        x5d = F.relu(self.dconv5(x5d))

        x4d = self.unpool4(x5d, indices=idx4p)
        x4d = F.relu(self.dconv4(x4d))

        x3d = self.unpool3(x4d, indices=idx3p)
        x3d = F.relu(self.dconv3(x3d))

        x2d = self.unpool2(x3d, indices=idx2p)
        x2d = F.relu(self.dconv2(x2d))

        x1d = self.unpool1(x2d, indices=idx1p)
        x1d = F.relu(self.dconv1(x1d))

        xpred = self.alpha_pred(x1d)

        return xpred

    def weights_init_random(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight,
                    mode='fan_in',
                    nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def hlvgg16(pretrained=False, **kwargs):
    """Constructs a VGG-16 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DeepMatting(input_chn=4, output_chn=1, use_pretrained=pretrained)
    
    if pretrained:
        corresp_name = CORRESP_NAME
        # load the state dict of pretrained model
        pretrained_dict = model_zoo.load_url(MODEL_URLS['vgg16_bn'])
        model_dict = model.state_dict()
        for name in pretrained_dict:
            if name not in corresp_name:
                continue
            if name == "features.0.weight":
                model_weight = model_dict[corresp_name[name]]
                assert(model_weight.shape[1] == 4)
                model_weight[:, 0:3, :, :] = pretrained_dict[name]
                model_weight[:, 3, :, :] = torch.tensor(0)
                model_dict[corresp_name[name]] = model_weight
            elif name == "classifier.0.weight":
                model_dict[corresp_name[name]] = pretrained_dict[name].view(4096, 512, 7, 7)
            else:
                model_dict[corresp_name[name]] = pretrained_dict[name]
        model.load_state_dict(model_dict)
    return model


if __name__ == "__main__":
    net = DeepMatting(input_chn=4, output_chn=1, use_pretrained=True)
    net.eval()
    net.cuda()
    
    from modelsummary import get_model_summary

    dump_x = torch.randn(1, 4, 224, 224).cuda()
    print(get_model_summary(net, dump_x))

    from time import time
    import numpy as np
    frame_rate = np.zeros((10, 1))
    for i in range(10):
        x = torch.randn(1, 4, 320, 320).cuda()
        torch.cuda.synchronize()
        start = time()
        y = net(x)
        torch.cuda.synchronize()
        end = time()
        del y
        running_frame_rate = 1 * float(1 / (end - start))
        frame_rate[i] = running_frame_rate
    print(np.mean(frame_rate))
    # print(y.shape)