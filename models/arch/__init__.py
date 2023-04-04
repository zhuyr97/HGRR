# Add your custom network here
from .default import DRNet
from .default1 import DRNet1
from .default2 import DRNet2
from .default3 import DRNet3
from .default3_retrain import DRNet3_re
from .default4 import DRNet4,DRNet4_
import torch


def basenet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, bottom_kernel_size=1, **kwargs)


def errnet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True, **kwargs)

def errnet_whyper(in_channels, out_channels, **kwargs):
    return DRNet1(in_channels, out_channels, 128, 4, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True,hyper=True,**kwargs)

def errnet_wohyper(in_channels, out_channels, **kwargs):
    return DRNet1(in_channels, out_channels, 128, 4, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True,hyper=False,**kwargs)

def errnet_wGlobal(in_channels, out_channels, **kwargs):
    return DRNet2(in_channels, out_channels, 144, 4, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True,global_residual=True,**kwargs)

def errnet_woGlobal(in_channels, out_channels, **kwargs):
    return DRNet2(in_channels, out_channels, 144, 4, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True,global_residual=False,**kwargs)

def errnet_wboosting(in_channels, out_channels, **kwargs):
    return DRNet3(in_channels, out_channels, 128, 4, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True,last_sigmoid=False,**kwargs)

def HGRR_wboosting1128(in_channels, out_channels, **kwargs):
    return DRNet3_re(in_channels, out_channels, 128, 4, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True,last_sigmoid=False,**kwargs)


def errnet_wboosting1(in_channels, out_channels, **kwargs):
    return DRNet3(in_channels, out_channels, 144, 4, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True,last_sigmoid=False,**kwargs)

def errnet_wboosting1_wact(in_channels, out_channels, **kwargs):
    return DRNet3(in_channels, out_channels, 144, 4, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True,last_sigmoid=True,**kwargs)


def errnet_wboosting_woR3attention(in_channels, out_channels, **kwargs):
    return DRNet4(in_channels, out_channels, 128, 4, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True,last_sigmoid=False,**kwargs)
def errnet_wboosting_woDalitedBlock(in_channels, out_channels, **kwargs):
    return DRNet4_(in_channels, out_channels, 128, 4, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True,last_sigmoid=False,**kwargs)
