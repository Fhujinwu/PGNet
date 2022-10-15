# -*- coding: utf-8 -*-
# @Time    : 2022/5/25 13:31
# @Author  : Jinwu Hu
# @FileName: SMIM.py

import torch
import torch.nn as nn
from module.baseline.base_models.BasicConv import BasicConv2d
from module.baseline.base_models.utils import cus_sample

class MSFM(nn.Module):
    def __init__(self,out_channel = 256):
        super(MSFM, self).__init__()
        #first conv
        self.down_pool = nn.AvgPool2d((2, 2), stride=2)
        self.u_up = cus_sample

        self.m2l_0 = nn.Sequential(nn.Conv2d(out_channel,out_channel,kernel_size=(1, 3), padding=(0, 1)),
                                   nn.Conv2d(out_channel,out_channel,kernel_size=(3,1),padding=(1,0)))
        self.m2m_0 = nn.Sequential(nn.Conv2d(out_channel,out_channel,kernel_size=(1, 3), padding=(0, 1)),
                                   nn.Conv2d(out_channel,out_channel,kernel_size=(3,1),padding=(1,0)))
        self.m2h_0 = nn.Sequential(nn.Conv2d(out_channel,out_channel,kernel_size=(1, 3), padding=(0, 1)),
                                   nn.Conv2d(out_channel,out_channel,kernel_size=(3,1),padding=(1,0)))
        self.bnl_0 = nn.BatchNorm2d(out_channel)
        self.bnm_0 = nn.BatchNorm2d(out_channel)
        self.bnh_0 = nn.BatchNorm2d(out_channel)

        self.h2h_1 = nn.Sequential(nn.Conv2d(out_channel,out_channel,kernel_size=(1, 3), padding=(0, 1)),
                                   nn.Conv2d(out_channel,out_channel,kernel_size=(3,1),padding=(1,0)))
        self.m2h_1 = nn.Sequential(nn.Conv2d(out_channel,out_channel,kernel_size=(1, 3), padding=(0, 1)),
                                   nn.Conv2d(out_channel,out_channel,kernel_size=(3,1),padding=(1,0)))
        self.m2m_1 = nn.Sequential(nn.Conv2d(out_channel,out_channel,kernel_size=(1, 3), padding=(0, 1)),
                                   nn.Conv2d(out_channel,out_channel,kernel_size=(3,1),padding=(1,0)))
        self.m2l_1 = nn.Sequential(nn.Conv2d(out_channel,out_channel,kernel_size=(1, 3), padding=(0, 1)),
                                   nn.Conv2d(out_channel,out_channel,kernel_size=(3,1),padding=(1,0)))
        self.l2l_1 = nn.Sequential(nn.Conv2d(out_channel,out_channel,kernel_size=(1, 3), padding=(0, 1)),
                                   nn.Conv2d(out_channel,out_channel,kernel_size=(3,1),padding=(1,0)))
        self.l2m_1 = nn.Sequential(nn.Conv2d(out_channel,out_channel,kernel_size=(1, 3), padding=(0, 1)),
                                   nn.Conv2d(out_channel,out_channel,kernel_size=(3,1),padding=(1,0)))
        self.h2m_1 = nn.Sequential(nn.Conv2d(out_channel,out_channel,kernel_size=(1, 3), padding=(0, 1)),
                                   nn.Conv2d(out_channel,out_channel,kernel_size=(3,1),padding=(1,0)))
        self.bnl_1 = nn.BatchNorm2d(out_channel)
        self.bnm_1 = nn.BatchNorm2d(out_channel)
        self.bnh_1 = nn.BatchNorm2d(out_channel)

        self.h2m_2 = nn.Sequential(nn.Conv2d(out_channel,out_channel,kernel_size=(1, 3), padding=(0, 1)),
                                   nn.Conv2d(out_channel,out_channel,kernel_size=(3,1),padding=(1,0)))
        self.m2m_2 = nn.Sequential(nn.Conv2d(out_channel,out_channel,kernel_size=(1, 3), padding=(0, 1)),
                                   nn.Conv2d(out_channel,out_channel,kernel_size=(3,1),padding=(1,0)))
        self.l2m_2 = nn.Sequential(nn.Conv2d(out_channel,out_channel,kernel_size=(1, 3), padding=(0, 1)),
                                   nn.Conv2d(out_channel,out_channel,kernel_size=(3,1),padding=(1,0)))
        self.bnm_2 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU(True)

    def forward(self, x):
        h, w = x.shape[2:]
        # first conv
        # x = self.conv_1(x)
        x_h = self.relu(self.bnh_0(self.m2h_0(self.u_up(x,size=(2*h, 2*w)))))
        x_m = self.relu(self.bnm_0(self.m2m_0(x)))
        x_l = self.relu(self.bnl_0(self.m2l_0(self.down_pool(x))))

        # mid conv
        x_h2h = self.h2h_1(x_h)
        x_h2m = self.h2m_1(self.down_pool(x_h))
        x_m2m = self.m2m_1(x_m)
        x_m2h = self.m2h_1(self.u_up(x_m,size=(2*h, 2*w)))
        x_m2l = self.m2l_1(self.down_pool(x_m))
        x_l2l = self.l2l_1(x_l)
        x_l2m = self.l2m_1(self.u_up(x_l,size=(h, w)))
        x_h = self.relu(self.bnh_1(x_h2h+x_m2h))
        x_m = self.relu(self.bnm_1(x_h2m+x_m2m+x_l2m))
        x_l = self.relu(self.bnl_1(x_l2l+x_m2l))

        x_h2m = self.h2m_2(self.down_pool(x_h))
        x_m2m = self.m2m_2(x_m)
        x_l2m = self.l2m_2(self.u_up(x_l,size=(h, w)))

        x_out = self.relu(self.bnm_2(x_h2m+x_m2m+x_l2m))
        # x_out = self.conv_2(x_out)

        return x_out


if __name__ == '__main__':
    model = MSFM(in_channel=256,out_channel=128).cuda()
    input_tensor = torch.randn(1, 256, 224, 224).cuda()

    prediction = model(input_tensor)
    print(prediction.size())
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params)
