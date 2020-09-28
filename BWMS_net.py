"""
CNN model for IGARSS2020 paper: Band-Wise Multi-Scale CNN Architecture for Remote Sensing Image Scene Classification
"""


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function
from torchvision import models


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def weights_init_kaiming(m):
    # https://github.com/lizhengwei1992/ResidualDenseNetwork-Pytorch/blob/master/main.py

    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data)

def fc_init_weights(m):
    # https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch/49433937#49433937
    if type(m) == nn.Linear:
        init.kaiming_normal_(m.weight.data)




class KB_MDP_RS50_FF_LReLU(nn.Module):

    def __init__(self, numClass=19):
        super().__init__()
        from xresnet import xresnet50

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(p=0.25)
        # self.relu = nn.ReLU(inplace=True)
        self.act_fn = nn.LeakyReLU(negative_slope=0.1,inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        # resnet = models.resnet18(pretrained=False)
        xresnet = xresnet50()

        f_list = list(xresnet.children())[:-3]

        self.b10_sp_scl3 = nn.Conv2d(4, 4, kernel_size=3, padding=1, stride=2, groups=4, bias=False)
        self.b10_sp_scl5 = nn.Conv2d(4, 4, kernel_size=5, padding=2, stride=2, groups=4, bias=False)
        self.b10_sp_scl7 = nn.Conv2d(4, 4, kernel_size=7, padding=3, stride=2, groups=4, bias=False)
        self.b10_sp_scl1 = nn.Conv2d(4, 4, kernel_size=1, stride=2, groups=4, bias=False)

        self.b20_sp_scl3 = nn.Conv2d(6, 6, kernel_size=3, padding=1, stride=1, groups=6, bias=False)
        self.b20_sp_scl5 = nn.Conv2d(6, 6, kernel_size=5, padding=2, stride=1, groups=6, bias=False)
        self.b20_sp_scl7 = nn.Conv2d(6, 6, kernel_size=7, padding=3, stride=1, groups=6, bias=False)
        self.b20_sp_scl1 = nn.Conv2d(6, 6, kernel_size=1, groups=6, bias=False)

        self.fusion = nn.Sequential(
            nn.Conv2d(40, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            self.act_fn
        )

        self.convs = nn.Sequential(
            *(f_list[1:])
        )

        self.FC1 = nn.Linear(2048, 128)
        self.FC2 = nn.Linear(128, numClass)

        self.initialize()

    def initialize(self):
        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x10, x20):

        x10_1 = self.b10_sp_scl1(x10)
        x10_3 = self.b10_sp_scl3(x10)
        x10_5 = self.b10_sp_scl5(x10)
        x10_7 = self.b10_sp_scl7(x10)

        x20_1 = self.b20_sp_scl1(x20)
        x20_3 = self.b20_sp_scl3(x20)
        x20_5 = self.b20_sp_scl5(x20)
        x20_7 = self.b20_sp_scl7(x20)

        x_f = torch.cat((x10_1, x10_3, x10_5, x10_7, x20_1, x20_3, x20_5, x20_7), dim=1)
        x_f = self.fusion(x_f)
        # print(x_f.shape)
        
        # check the output of each layer
        # for layer in self.convs:
        #     x_f = layer(x_f)
        #     print(x_f.shape)

        x_f = self.convs(x_f)
        # print(x_f.shape)

        x_f = self.avgpool(x_f).view(x10.size(0), -1)
        x_f = self.dropout(x_f)

        x_f = self.FC1(x_f)
        x_f = self.act_fn(x_f)
        x_f = self.dropout(x_f)

        logits = self.FC2(x_f)

        return logits



class KB_MDP_RS18_FF_LReLU(nn.Module):

    def __init__(self, numClass=19):
        super().__init__()
        from xresnet import xresnet18

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(p=0.25)
        # self.relu = nn.ReLU(inplace=True)
        self.act_fn = nn.LeakyReLU(negative_slope=0.1,inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        # resnet = models.resnet18(pretrained=False)
        xresnet = xresnet18()

        f_list = list(xresnet.children())[:-3]

        self.b10_sp_scl3 = nn.Conv2d(4, 4, kernel_size=3, padding=1, stride=2, groups=4, bias=False)
        self.b10_sp_scl5 = nn.Conv2d(4, 4, kernel_size=5, padding=2, stride=2, groups=4, bias=False)
        self.b10_sp_scl7 = nn.Conv2d(4, 4, kernel_size=7, padding=3, stride=2, groups=4, bias=False)
        self.b10_sp_scl1 = nn.Conv2d(4, 4, kernel_size=1, stride=2, groups=4, bias=False)

        self.b20_sp_scl3 = nn.Conv2d(6, 6, kernel_size=3, padding=1, stride=1, groups=6, bias=False)
        self.b20_sp_scl5 = nn.Conv2d(6, 6, kernel_size=5, padding=2, stride=1, groups=6, bias=False)
        self.b20_sp_scl7 = nn.Conv2d(6, 6, kernel_size=7, padding=3, stride=1, groups=6, bias=False)
        self.b20_sp_scl1 = nn.Conv2d(6, 6, kernel_size=1, groups=6, bias=False)


        self.fusion = nn.Sequential(
            nn.Conv2d(40, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            self.act_fn
        )

        self.convs = nn.Sequential(
            *(f_list[1:])
        )

        self.FC1 = nn.Linear(512, 128)
        self.FC2 = nn.Linear(128, numClass)

        self.initialize()

    def initialize(self):
        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x10, x20):

        x10_1 = self.b10_sp_scl1(x10)
        x10_3 = self.b10_sp_scl3(x10)
        x10_5 = self.b10_sp_scl5(x10)
        x10_7 = self.b10_sp_scl7(x10)

        x20_1 = self.b20_sp_scl1(x20)
        x20_3 = self.b20_sp_scl3(x20)
        x20_5 = self.b20_sp_scl5(x20)
        x20_7 = self.b20_sp_scl7(x20)

        x_f = torch.cat((x10_1, x10_3, x10_5, x10_7, x20_1, x20_3, x20_5, x20_7), dim=1)
        x_f = self.fusion(x_f)
        # print(x_f.shape)
        
        # check the output of each layer
        # for layer in self.convs:
        #     x_f = layer(x_f)
        #     print(x_f.shape)

        x_f = self.convs(x_f)
        # print(x_f.shape)

        x_f = self.avgpool(x_f).view(x10.size(0), -1)
        x_f = self.dropout(x_f)

        x_f = self.FC1(x_f)
        x_f = self.act_fn(x_f)
        x_f = self.dropout(x_f)

        logits = self.FC2(x_f)

        return logits


