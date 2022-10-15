# -*- coding: utf-8 -*-
# @Time    : 2022/6/8 14:49
# @Author  : Jinwu Hu
# @FileName: PGNet.py
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from module.baseline.base_models.Res2Net_v1b import res2net50_v1b_26w_4s
from module.baseline.base_models.BasicConv import BasicConv2d
from simplecv import registry
from simplecv.interface import CVModule
from module.baseline.base_models.transformerblock import VisionT
from module.baseline.base_models.MSFM import MSFM
@registry.MODEL.register('PGNet')
class Remoteseg(CVModule):
    def __init__(self,config):
        super(Remoteseg, self).__init__(config)
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.num_classes = self.config['num_classes']
        self.channel = self.config['channel']

        self.Translayer1 = BasicConv2d(in_planes=256, out_planes=self.channel, kernel_size=1)
        self.Translayer2 = BasicConv2d(in_planes=512, out_planes=self.channel, kernel_size=1)
        self.Translayer3 = BasicConv2d(in_planes=1024, out_planes=self.channel, kernel_size=1)
        self.Translayer4 = BasicConv2d(in_planes=2048, out_planes=self.channel, kernel_size=1)
        self.LGM = nn.Sequential(BasicConv2d(in_planes=2048,out_planes=320,kernel_size=1),
                                 VisionT(img_size=28,patch_size=4,in_chans=320,embed_dim=512,num_heads=[8],qkv_bias=True,
                                         norm_layer=partial(nn.LayerNorm, eps=1e-6),depths=[2],sr_ratios=[1],drop_rate=0,
                                         drop_path_rate=0.1,stride=1),
                                 BasicConv2d(in_planes=512,out_planes=self.channel,kernel_size=1))
        self.MSF = MSFM(in_channel=256,out_channel=128)


        self.uper =  nn.Sequential(nn.ConvTranspose2d(in_channels=self.channel, out_channels=self.channel, kernel_size=4, stride=2, padding=1),
                                   BasicConv2d(in_planes=self.channel, out_planes=self.channel,kernel_size=3,stride=1,padding=1))

        self.prediction = nn.Sequential(BasicConv2d(in_planes=self.channel,out_planes=self.channel,kernel_size=3, stride=1,padding=1),
                                        nn.Conv2d(self.channel, self.num_classes, 1))


    def forward(self, x,y=None):

        #feature extraction
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # bs, 64, 224, 224
        x1 = self.resnet.layer1(x)      # bs, 256, 224, 224
        x2 = self.resnet.layer2(x1)     # bs, 512, 112, 112
        x3 = self.resnet.layer3(x2)     # bs, 1024, 56, 56
        x4 = self.resnet.layer4(x3)     # bs, 2048, 28, 28
        lgm = self.LGM(x4)
        x1 = self.Translayer1(x1) + F.interpolate(lgm, scale_factor=8, mode='bilinear', align_corners=False)
        x2 = self.Translayer2(x2) + F.interpolate(lgm, scale_factor=4, mode='bilinear', align_corners=False)
        x3 = self.Translayer3(x3) + F.interpolate(lgm, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = self.Translayer4(x4)

        x3 = x3 + self.uper(x4)
        x3 = self.MSF(x3)
        x2 = x2 + self.uper(x3)
        x2 = self.MSF(x2)
        x1 = x1 + self.uper(x2)
        x1 = self.MSF(x1)
        cls_pred  = self.prediction(x1)
        cls_pred = F.interpolate(cls_pred, scale_factor=4, mode='bilinear', align_corners=False)
        if self.training:
            cls_true = y['cls']
            loss_dict = {
                'cls_loss': self.cls_loss(cls_pred,cls_true)
            }

            return loss_dict

        cls_prob = torch.softmax(cls_pred, dim=1)

        return cls_prob

    def cls_loss(self, y_pred,y_true):
        loss = F.cross_entropy(y_pred, y_true.long(), ignore_index=self.config.loss.ignore_index)
        return loss



    def set_defalut_config(self):
        self.config.update(dict(
            num_classes=16,
            channel = 256
        )
        )
