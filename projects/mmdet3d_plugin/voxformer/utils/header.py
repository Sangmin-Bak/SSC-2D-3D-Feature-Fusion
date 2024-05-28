# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Header(nn.Module):
    def __init__(
        self,
        class_num,
        feature,
        num_layers=1
    ):
        super(Header, self).__init__()
        self.feature = feature
        self.class_num = class_num
        mlp_head = []
        # for _ in range(num_layers):
        #     mlp_head.append(nn.Linear(self.feature, self.feature))
        #     mlp_head.append(nn.LayerNorm(self.feature))
        #     mlp_head.append(nn.ReLU(inplace=True))
        # mlp_head.append(nn.Linear(self.feature, self.class_num))
        # self.mlp_head = nn.Sequential(*mlp_head)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.feature),
            nn.Linear(self.feature, self.class_num),
        )
        # self.fcn_head = nn.Sequential(
        #     nn.Conv3d(self.feature, self.feature, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm3d(self.feature),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(self.feature, self.class_num, kernel_size=1, stride=1),
        # )
        # self.fcn_head = nn.Conv3d(self.feature, self.class_num, kernel_size=1, stride=1)

        self.up_scale_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
    def forward(self, input_dict):
        res = {}

        x3d_l1 = input_dict["x3d"] # [1, 64, 128, 128, 16]

        x3d_up_l1 = self.up_scale_2(x3d_l1) # [1, dim, 128, 128, 16] -> [1, dim, 256, 256, 32]

        _, feat_dim, w, l, h  = x3d_up_l1.shape
        
        # ssc_logit_full = self.fcn_head(x3d_up_l1)
        
        # res["ssc_logit"] = ssc_logit_full

        x3d_up_l1 = x3d_up_l1.squeeze().permute(1,2,3,0).reshape(-1, feat_dim)

        ssc_logit_full = self.mlp_head(x3d_up_l1)

        res["ssc_logit"] = ssc_logit_full.reshape(w, l, h, self.class_num).permute(3,0,1,2).unsqueeze(0)

        return res

class OccupancyHeader(nn.Module):
    def __init__(
        self,
        class_num,
        feature,
        num_occ_fcs=2,
    ):
        super(OccupancyHeader, self).__init__()
        self.feature = feature
        self.class_num = class_num
        # mlp_head = []
        # for _ in range(num_layers):
        #     mlp_head.append(nn.Linear(self.feature, self.feature))
        #     mlp_head.append(nn.LayerNorm(self.feature))
        #     mlp_head.append(nn.ReLU(inplace=True))
        # mlp_head.append(nn.Linear(self.feature, self.class_num))
        # self.mlp_head = nn.Sequential(*mlp_head)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.feature),
            nn.Linear(self.feature, self.class_num),
        )

        self.up_scale_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)


    def forward(self, input_dict):
        res = {}

        x3d_l1 = input_dict["x3d"] # [1, 64, 128, 128, 16]

        _, feat_dim, w, l, h  = x3d_l1.shape

        x3d_l1 = x3d_l1.squeeze().permute(1,2,3,0).reshape(-1, feat_dim)

        ssc_logit_full = self.mlp_head(x3d_l1)

        res["ssc_logit"] = ssc_logit_full.reshape(w, l, h, self.class_num).permute(3,0,1,2).unsqueeze(0)

        return res

    
# class SegmentationHead(nn.Module):
#   '''
#   3D Segmentation heads to retrieve semantic segmentation at each scale.
#   Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
#   '''
#   def __init__(self, inplanes, planes, nbr_classes, dilations_conv_list):
#         super().__init__()

#         # First convolution
#         self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1)

#         # ASPP Block
#         self.conv_list = dilations_conv_list
#         self.conv1 = nn.ModuleList(
#         [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
#         self.bn1 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
#         self.conv2 = nn.ModuleList(
#         [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
#         self.bn2 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
#         self.relu = nn.ReLU(inplace=True)

#         # Convolution for output
#         self.conv_classes = nn.Conv3d(planes, nbr_classes, kernel_size=3, padding=1, stride=1)

#   def forward(self, x_in):

#     # Dimension exapension
#     x_in = x_in[:, None, :, :, :]

#     # Convolution to go from inplanes to planes features...
#     x_in = self.relu(self.conv0(x_in))

#     y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
#     for i in range(1, len(self.conv_list)):
#       y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
#     x_in = self.relu(y + x_in)  # modified

#     x_in = self.conv_classes(x_in)

#     return x_in
    
class SegmentationHead(nn.Module):
    '''
    3D Segmentation heads to retrieve semantic segmentation at each scale.
    Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
    '''
    def __init__(self, inplanes, planes, nbr_classes, dilations_conv_list=[1, 2, 3]):
        super().__init__()

        # self.up_scale_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        # First convolution
        self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1)

        # ASPP Block
        self.conv_list = dilations_conv_list
        self.conv1 = nn.ModuleList(
            [nn.Conv3d(inplanes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
        self.bn1 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
        self.conv2 = nn.ModuleList(
            [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
        self.bn2 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
        self.relu = nn.ReLU(inplace=True)

        # Convolution for output
        self.conv_classes = nn.Conv3d(planes, nbr_classes, kernel_size=3, padding=1, stride=1)

    def forward(self, input_dict):
        res = {}
        x3d_l1 = input_dict["x3d"] # [1, 64, 128, 128, 16]
        # x3d_l1 = self.up_scale_2(x3d_l1) # [1, dim, 128, 128, 16] -> [1, dim, 256, 256, 32]
        # _, feat_dim, w, l, h  = x3d_l1.shape
        x3d_l1 = x3d_l1.permute(0, 1, 4, 2, 3) # [1, dim, 32, 256, 256]
        # Dimension exapension
        # x_in = x_in[:, None, :, :, :]

        # Convolution to go from inplanes to planes features...
        x3d_l1 = self.relu(self.conv0(x3d_l1))

        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x3d_l1)))))
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x3d_l1)))))
        x3d_l1 = self.relu(y + x3d_l1)  # modified

        ssc_logit_full = self.conv_classes(x3d_l1)
        res["ssc_logit"] = ssc_logit_full.permute(0, 1, 3, 4, 2)
        # res["ssc_logit"] = ssc_logit_full

        return res
    

class LMSCHead(nn.Module):
  '''
  3D Segmentation heads to retrieve semantic segmentation at each scale.
  Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
  '''
  def __init__(self, inplanes, planes, nbr_classes, dilations_conv_list):
    super().__init__()

    # First convolution
    self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1)

    # ASPP Block
    self.conv_list = dilations_conv_list
    self.conv1 = nn.ModuleList(
      [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
    self.bn1 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
    self.conv2 = nn.ModuleList(
      [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
    self.bn2 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
    self.relu = nn.ReLU(inplace=True)

    # Convolution for output
    self.conv_classes = nn.Conv3d(planes, nbr_classes, kernel_size=3, padding=1, stride=1)

  def forward(self, x_in):

    # Dimension exapension
    x_in = x_in[:, None, :, :, :]

    # Convolution to go from inplanes to planes features...
    x_in = self.relu(self.conv0(x_in))

    y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
    for i in range(1, len(self.conv_list)):
      y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
    x_in = self.relu(y + x_in)  # modified

    x_in = self.conv_classes(x_in)

    return x_in
    
class MultiObjectSegmentationHead(nn.Module):
    '''
    3D Segmentation heads to retrieve semantic segmentation at each scale.
    Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
    '''
    def __init__(self, inplanes, planes, nbr_classes):
        super().__init__()

        self.up_scale_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        # First convolution
        self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1)

        # ASPP Block
        # self.conv_list = dilations_conv_list
        # self.conv1 = nn.ModuleList(
        #     [nn.Conv3d(inplanes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
        # self.bn1 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
        # self.conv2 = nn.ModuleList(
        #     [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
        # self.bn2 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
        self.relu = nn.ReLU(inplace=True)

        # Convolution for output
        self.conv_classes = nn.Conv3d(planes, nbr_classes, kernel_size=3, padding=1, stride=1)

    def forward(self, input_dict):
        res = {}
        x3d_l1 = input_dict["x3d"] # [1, 64, 128, 128, 16]
        x3d_l1 = self.up_scale_2(x3d_l1) # [1, dim, 128, 128, 16] -> [1, dim, 256, 256, 32]
        _, feat_dim, w, l, h  = x3d_l1.shape
        x3d_l1 = x3d_l1.permute(0, 1, 4, 2, 3) # [1, dim, 32, 256, 256]
        # Dimension exapension
        # x_in = x_in[:, None, :, :, :]

        # Convolution to go from inplanes to planes features...
        x3d_l1 = self.relu(self.conv0(x3d_l1))

        # y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x3d_l1)))))
        # for i in range(1, len(self.conv_list)):
        #     y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x3d_l1)))))
        # x3d_l1 = self.relu(y + x3d_l1)  # modified

        ssc_logit_full = self.conv_classes(x3d_l1)
        res["ssc_logit"] = ssc_logit_full.permute(0, 1, 3, 4, 2)

        return res