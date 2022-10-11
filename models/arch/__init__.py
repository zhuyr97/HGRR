# Add your custom network here
from .default import DRNet
from .default1 import DRNet1
from .default2 import DRNet2
from .default3 import DRNet3,DRNet3_T
from .default4 import DRNet4
from .default5 import DRNet5_ori

import torch


def basenet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, bottom_kernel_size=1, **kwargs)


def errnet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True, **kwargs)

def train_DB_wboosting_T(in_channels, out_channels, **kwargs):
    return DRNet3_T(in_channels, out_channels, 128, 4, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True,last_sigmoid=False,**kwargs)


def train_DB_wboosting(in_channels, out_channels, **kwargs):
    return DRNet3(in_channels, out_channels, 128, 4, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True,last_sigmoid=False,**kwargs)

def train_DB_wboosting1(in_channels, out_channels, **kwargs):
    return DRNet3(in_channels, out_channels, 96, 6, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True,last_sigmoid=False,**kwargs)

def train_DB_wboosting2(in_channels, out_channels, **kwargs):
    return DRNet3(in_channels, out_channels, 72, 8, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True,last_sigmoid=False,**kwargs)

def train_DB_DRNet4_wboosting(in_channels, out_channels, **kwargs):
    return DRNet4(in_channels, out_channels, 128, 4, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True,last_sigmoid=False,**kwargs)

def train_DB_wboosting2_ori(in_channels, out_channels, **kwargs):
    return DRNet5_ori(in_channels, out_channels, 72, 8, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True,last_sigmoid=False,**kwargs)
