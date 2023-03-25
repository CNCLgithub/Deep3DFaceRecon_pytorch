"""This script defines deep neural networks for Deep3DFaceRecon_pytorch
"""

import os
import numpy as np
from torchvision.models import vgg19
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch
from torch import Tensor
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from .arcface_torch.backbones import get_model
from kornia.geometry import warp_affine
from .resnet import resnet18, resnet50

def resize_n_crop(image, M, dsize=112):
    # image: (b, c, h, w)
    # M   :  (b, 2, 3)
    return warp_affine(image, M, dsize=(dsize, dsize))

def filter_state_dict(state_dict, remove_name='fc'):
    new_state_dict = {}
    for key in state_dict:
        if remove_name in key:
            continue
        new_state_dict[key] = state_dict[key]
    return new_state_dict

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_epochs, gamma=0.2)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_net_recon(net_recon, use_last_fc=False, pretrained=True):
    return ReconNetWrapper(net_recon, use_last_fc=use_last_fc, pretrained=pretrained)

def define_net_recog(net_recog, pretrained_path=None):
    net = RecogNetWrapper(net_recog=net_recog, pretrained_path=pretrained_path)
    net.eval()
    return net

# https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/2
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

class ReconNetWrapper(nn.Module):
    fc_dim=257
    def __init__(self, net_recon, use_last_fc=False, pretrained=True):
        super(ReconNetWrapper, self).__init__()
        if net_recon not in func_dict:
            return  NotImplementedError('network [%s] is not implemented', net_recon)

        self.get_backbone(net_recon, pretrained, use_last_fc)

    def get_backbone(self, net, pretrained, use_last_fc):
        func, last_dim = func_dict[net]
        weights = None
        if pretrained:
            weights = "DEFAULT"
            print(f"loading init net_recon {net} with pretrained weights {weights}")
        self.final_layers = nn.ModuleList([nn.Identity()])

        if "resnet" in net:
            self.backbone = func(weights=weights, use_last_fc=use_last_fc)
            if not use_last_fc:
                self.final_layers = nn.ModuleList([
                    conv1x1(last_dim, 80, bias=True), # id layer
                    conv1x1(last_dim, 64, bias=True), # exp layer
                    conv1x1(last_dim, 80, bias=True), # tex layer
                    conv1x1(last_dim, 3, bias=True),  # angle layer
                    conv1x1(last_dim, 27, bias=True), # gamma layer
                    conv1x1(last_dim, 2, bias=True),  # tx, ty
                    conv1x1(last_dim, 1, bias=True)   # tz
                ])
                for m in self.final_layers:
                    nn.init.constant_(m.weight, 0.)
                    nn.init.constant_(m.bias, 0.)
            else:
                self.backbone.fc = \
                    nn.Linear(self.backbone.fc.in_features, self.fc_dim)

        elif "vgg" in net:
            self.backbone = func(weights=weights)
            self.backbone.classifier.apply(weight_reset)
            self.backbone.classifier[-1] = \
                nn.Linear(self.backbone.classifier[-1].in_features,
                          self.fc_dim)

    def forward(self, x):
        x = self.backbone(x)
        output = []
        for layer in self.final_layers:
            output.append(layer(x))
        x = torch.flatten(torch.cat(output, dim=1), 1)
        return x

class RecogNetWrapper(nn.Module):
    def __init__(self, net_recog, pretrained_path=None, input_size=112):
        super(RecogNetWrapper, self).__init__()
        net = get_model(name=net_recog, fp16=False)
        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            net.load_state_dict(state_dict)
            print("loading pretrained net_recog %s from %s" %(net_recog, pretrained_path))
        for param in net.parameters():
            param.requires_grad = False
        self.net = net
        self.preprocess = lambda x: 2 * x - 1
        self.input_size=input_size
        
    def forward(self, image, M):
        image = self.preprocess(resize_n_crop(image, M, self.input_size))
        id_feature = F.normalize(self.net(image), dim=-1, p=2)
        return id_feature

def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

func_dict = {
    'resnet18': (resnet18, 512),
    'resnet50': (resnet50, 2048),
    'vgg19': (vgg19, 0),
}
