# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .backbone import build_backbone
from .ASPP import ASPP


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
class deeplabv3plus(nn.Module):
    def __init__(self, cfg):
        super(deeplabv3plus, self).__init__()
        self.backbone = None        
        self.backbone_layers = None
        input_channel = 2048        
        self.aspp = ASPP(dim_in=input_channel, 
                dim_out=cfg.model_aspp_outdim, 
                rate=16//cfg.model_output_stride,
                bn_mom = cfg.train_bn_mom)
        self.dropout1 = nn.Dropout(0.5)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.model_output_stride//4)
        indim = 256
        self.shortcut_conv = nn.Sequential(
                nn.Conv2d(indim, cfg.model_shortcut_dim, cfg.model_shortcut_kernel, 1, padding=cfg.model_shortcut_kernel//2,bias=True),
                nn.BatchNorm2d(cfg.model_shortcut_dim),
                nn.ReLU(inplace=True),        
        )        
        self.cat_conv = nn.Sequential(
                nn.Conv2d(cfg.model_aspp_outdim+cfg.model_shortcut_dim, cfg.model_aspp_outdim, 3, 1, padding=1,bias=True),
                nn.BatchNorm2d(cfg.model_aspp_outdim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Conv2d(cfg.model_aspp_outdim, cfg.model_aspp_outdim, 3, 1, padding=1,bias=True),
                nn.BatchNorm2d(cfg.model_aspp_outdim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(cfg.model_aspp_outdim, cfg.model_num_classes, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = build_backbone(cfg.model_backbone, os=cfg.model_output_stride)
        self.backbone_layers = self.backbone.get_layers()

    def forward(self, x):
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)

        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp,feature_shallow],1)
        result = self.cat_conv(feature_cat)
        # result = self.cls_conv(result)
        # result = self.upsample4(result)
        return result



class DeepLabV3PDect(nn.Module):
    def __init__(self,model,cfg,heads, head_conv,pretrained=True):
        super(DeepLabV3PDect,self).__init__()
        model=model(cfg)
        if pretrained:
            params_new=dict()
            params=torch.load('/home/mayx/project/github/CenterNet/models/deeplabv3plus_res101_atrous_VOC2012_epoch46_all.pth')
            for k,v in params.items():
                if 'cls_conv' not in k:
                    params_new[k[7:]]=v
            state_dict=model.state_dict()
            state_dict.update(params_new)
            model.load_state_dict(state_dict)
        self.backbone=model
        self.heads=heads
        self.head_conv=head_conv
        
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                  nn.Conv2d(256, head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=1, stride=1, 
                    padding=0, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(256, classes, 
                  kernel_size=1, stride=1, 
                  padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)
        
    def forward(self, x):
        x=self.backbone(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]


def get_deeplabv3plus(pretrain=True):
    class Config():
        model_aspp_outdim=256
        model_output_stride=16
        train_bn_mom = 0.0003
        model_shortcut_dim=48
        model_shortcut_kernel=1
        model_num_classes=20
        model_backbone='res101_atrous'
    cfg=Config()
    heads = {'hm': 20,'wh': 2,'reg': 2 }

    deeplab_model=DeepLabV3PDect(deeplabv3plus,cfg,heads, head_conv=256)

    return deeplab_model

