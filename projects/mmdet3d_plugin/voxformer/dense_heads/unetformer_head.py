# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.cnn import xavier_init
from projects.mmdet3d_plugin.voxformer.utils.header import SegmentationHead, Header
from projects.mmdet3d_plugin.voxformer.utils.ssc_loss import sem_scal_loss, KL_sep, geo_scal_loss, CE_ssc_loss, BCE_ssc_loss
from projects.mmdet3d_plugin.voxformer.utils.lovasz_softmax import lovasz_softmax
from projects.mmdet3d_plugin.models.utils.bricks import run_time
# from .sparse_unet3d_head import SparseUNet3D
from .unet3d_head import Conv3DBlock, weight_xavier_init
from mmdet3d.models.builder import build_loss

@HEADS.register_module()
class UNetFormerHead(nn.Module):
    def __init__(
        self,
        *args,
        bev_h,
        bev_w,
        bev_z,
        cross_transformer,
        self_transformer,
        positional_encoding,
        embed_dims,
        focal_loss=None,
        lovasz_loss=False,
        CE_ssc_loss=False,
        geo_scal_loss=False,
        sem_scal_loss=False,
        dist_loss = False,
        save_flag = False,
        save_dir=False,
        data_type='semantickitti',
        **kwargs
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w 
        self.bev_z = bev_z
        self.real_w = 51.2
        self.real_h = 51.2
        self.embed_dims = embed_dims
        self.bev_embed = nn.Embedding((self.bev_h) * (self.bev_w) * (self.bev_z), self.embed_dims)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.cross_transformer = build_transformer(cross_transformer)
        self.self_transformer = build_transformer(self_transformer)
        if focal_loss is not None:
            self.focal_loss = build_loss(focal_loss)
        else:
            self.focal_loss = None
        self.lovasz_loss = lovasz_loss
        self.sem_scal_loss = sem_scal_loss
        self.geo_scal_loss = geo_scal_loss
        self.CE_ssc_loss = CE_ssc_loss
        self.dist_loss = dist_loss
        self.save_flag = save_flag
        self.save_dir = save_dir
        
        assert data_type in ("semantickitti", "kitti-360")
        if data_type == "semantickitti":
            self.n_classes = 20
            self.class_names =  [ "empty", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist", "motorcyclist", "road", 
                                "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign",]
            self.class_weights = torch.from_numpy(np.array([0.446, 0.603, 0.852, 0.856, 0.747, 0.734, 0.801, 0.796, 0.818, 0.557, 
                                                            0.653, 0.568, 0.683, 0.560, 0.603, 0.530, 0.688, 0.574, 0.716, 0.786]))
        if data_type == "kitti-360":
            self.n_classes = 19
            self.class_names =  ['empty', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'road', 'parking', 'sidewalk', 
                                'other-ground', 'building', 'fence', 'vegetation', 'terrain', 'pole', 'traffic-sign', 'other-structure', 'other-object']
            self.class_weights = torch.from_numpy(np.array([0.464, 0.595, 0.865, 0.871, 0.717, 0.657, 0.852, 0.541, 0.602,
                                                            0.567, 0.607, 0.540, 0.636, 0.513, 0.564, 0.701, 0.774, 0.580, 0.690]))
        # self.occ_class_weights = torch.from_numpy(np.array([0.446, 0.505]))
        self.header = Header(self.n_classes, feature=self.embed_dims)
        # self.occ_header = nn.Sequential(
        #     nn.Linear(self.embed_dims, self.embed_dims),
        #     nn.LayerNorm(self.embed_dims),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.embed_dims, 2),
        # )
        self.loss_voxel_lovasz_weight = 1.0
        
        # self.voxel_layer = Conv3DBlock(1, self.embed_dims)
        # self.voxel_layer.apply(weight_xavier_init)
        # self.bev_embed.weight.data.uniform_(-1, 1)
        
    def forward(self, mlvl_feats, img_metas, voxel_feats, target):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            img_metas: Meta information such as camera intrinsics.
            target: Semantic completion ground truth. 
        Returns:
            ssc_logit (Tensor): Outputs from the segmentation head.
        """

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        device = mlvl_feats[0].device

        # Generate bev postional embeddings for cross and self attention
        bev_pos_cross_attn = self.positional_encoding(torch.zeros((bs, 512, 512), device=device).to(dtype)).to(dtype) # [1, dim, 128*4, 128*4]
        bev_pos_self_attn = self.positional_encoding(torch.zeros((bs, 512, 512), device=device).to(dtype)).to(dtype) # [1, dim, 128*4, 128*4]

        # Load query proposals
        bev_queries = self.bev_embed.weight.to(dtype) #[128*128*16, dim]
        if voxel_feats is not None:
            voxel_feats_flatten = voxel_feats.flatten(2).permute(0, 2, 1)
            bev_queries = bev_queries + voxel_feats_flatten.reshape(-1, self.embed_dims)
        # bev_queries = bev_queries + self.voxel_layer(voxel_feats_flatten).reshape(-1, self.embed_dims)
        _, ref_3d = self.get_ref_3d()

        # Compute seed features of query proposals by deformable cross attention
        seed_feats = self.cross_transformer.get_vox_features(
            mlvl_feats,
            bev_queries, 
            self.bev_h,
            self.bev_w,
            self.bev_z,
            ref_3d=ref_3d,
            vox_coords=None,
            unmasked_idx=None,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos_cross_attn,
            img_metas=img_metas,
            prev_bev=None,
        )
        # seed_feats = seed_feats + bev_queries.unsqueeze(0)
        # occ_feats = seed_feats.reshape(bs, self.bev_h, \
        #     self.bev_w, self.bev_z, self.embed_dims).contiguous()
        
        ssc_feats = self.self_transformer.diffuse_vox_features(
            mlvl_feats,
            seed_feats.reshape(-1, self.embed_dims).contiguous(),
            self.bev_h,
            self.bev_w,
            self.bev_z,
            ref_3d=ref_3d,
            vox_coords=None,
            unmasked_idx=None,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos_self_attn,
            img_metas=img_metas,
            prev_bev=None,
        )

        ssc_feats = ssc_feats.reshape(self.bev_h, self.bev_w, self.bev_z, self.embed_dims)
        input_dict = {
            "x3d": ssc_feats.permute(3, 0, 1, 2).unsqueeze(0),
        }
        out = self.header(input_dict)
        # occ_out = self.occ_header(occ_feats).permute(0, 4, 1, 2, 3)
        # out.update(
        #     dict(occ_logit=occ_out)
        # )

        return out 
    
    # def loss_occ(self, out_dict, target):
    #     occ_pred = out_dict["occ_logit"]
    #     target_1_2 = F.interpolate(target.unsqueeze(1), scale_factor=0.5, mode='trilinear', align_corners=True).squeeze(1)
    #     ones = torch.ones_like(target_1_2).to(target_1_2.device)
    #     occ_target = torch.where(torch.logical_or(target_1_2==255, target_1_2==0), target_1_2, ones) # [1, 128, 128, 16]
    #     class_weight = self.occ_class_weights.type_as(target_1_2)
    #     loss_occ = BCE_ssc_loss(occ_pred, occ_target, class_weight, alpha=0.54)
    #     return loss_occ
    
    def get_occ_predictions(self, out_dict):
        ssc_pred = out_dict["ssc_logit"]
        y_pred = ssc_pred.detach().cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)  
        y_pred_bin = self.pack(y_pred)
        y_pred_bin.tofile('./pretrain_test.bin')
        return y_pred

    def step(self, out_dict, teacher_dict, target, img_metas, step_type):
        """Training/validation function.
        Args:
            out_dict (dict[Tensor]): Segmentation output.
            img_metas: Meta information such as camera intrinsics.
            target: Semantic completion ground truth. 
            step_type: Train or test.
        Returns:
            loss or predictions
        """

        ssc_pred = out_dict["ssc_logit"]
    
        if step_type== "train":
            loss_dict = dict()

            if self.focal_loss is not None:
                valid_target = target[target!=255]
                valid_pred = ssc_pred.permute(0, 2, 3, 4, 1)[target!=255]
                num_classes = ssc_pred.shape[1]

                num_pos_occ = torch.sum(valid_target < num_classes)
                occ_avg_factor = num_pos_occ * 1.0

                loss_ssc = self.focal_loss(valid_pred, valid_target.view(-1).long(), avg_factor=occ_avg_factor)
                # loss_ssc = self.focal_loss(valid_pred, valid_target.view(-1).long(), avg_factor=None)
                loss_dict['loss_ssc'] = loss_ssc
                
            if self.CE_ssc_loss:
                class_weight = self.class_weights.type_as(target)
                loss_ssc = CE_ssc_loss(ssc_pred, target, class_weight)
                loss_dict['loss_ssc'] = loss_ssc
                
            if self.lovasz_loss:
                loss_lovasz_softmax = self.loss_voxel_lovasz_weight * lovasz_softmax(torch.softmax(ssc_pred, dim=1), target, ignore=255)
                loss_dict['loss_lovasz_softmax'] = loss_lovasz_softmax
                
            if self.sem_scal_loss:
                loss_sem_scal = sem_scal_loss(ssc_pred, target)
                loss_dict['loss_sem_scal'] = loss_sem_scal

            if self.geo_scal_loss:
                loss_geo_scal = geo_scal_loss(ssc_pred, target)
                loss_dict['loss_geo_scal'] = loss_geo_scal

            

            return loss_dict

        elif step_type== "val" or "test":
            if target is not None:
                y_true = target.cpu().numpy()
            else:
                y_true = None
            y_pred = ssc_pred.detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)

            result = dict()
            result['y_pred'] = y_pred
            result['y_true'] = y_true

            if self.save_flag:
                self.save_pred(img_metas, y_pred)

            return result

    def training_step(self, out_dict, target, img_metas, teacher_dict=None):
        """Training step.
        """
        return self.step(out_dict, teacher_dict, target, img_metas, "train")

    def validation_step(self, out_dict, target, img_metas):
        """Validation step.
        """
        return self.step(out_dict, None, target, img_metas, "val")

    def get_ref_3d(self):
        """Get reference points in 3D.
        Args:
            self.real_h, self.bev_h
        Returns:
            vox_coords (Array): Voxel indices
            ref_3d (Array): 3D reference points
        """
        scene_size = (51.2, 51.2, 6.4)
        vox_origin = np.array([0, -25.6, -2])
        voxel_size = self.real_h / self.bev_h

        vol_bnds = np.zeros((3,2))
        vol_bnds[:,0] = vox_origin
        vol_bnds[:,1] = vox_origin + np.array(scene_size)

        # Compute the voxels index in lidar cooridnates
        vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
        idx = np.array([range(vol_dim[0]*vol_dim[1]*vol_dim[2])])
        xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing='ij')
        vox_coords = np.concatenate([xv.reshape(1,-1), yv.reshape(1,-1), zv.reshape(1,-1), idx], axis=0).astype(int).T

        # Normalize the voxels centroids in lidar cooridnates
        ref_3d = np.concatenate([(xv.reshape(1,-1)+0.5)/self.bev_h, (yv.reshape(1,-1)+0.5)/self.bev_w, (zv.reshape(1,-1)+0.5)/self.bev_z,], axis=0).astype(np.float64).T 

        return vox_coords, ref_3d

    def save_pred(self, img_metas, y_pred):
        """Save predictions for evaluations and visualizations.

        learning_map_inv: inverse of previous map
        
        0: 0    # "unlabeled/ignored"  # 1: 10   # "car"        # 2: 11   # "bicycle"       # 3: 15   # "motorcycle"     # 4: 18   # "truck" 
        5: 20   # "other-vehicle"      # 6: 30   # "person"     # 7: 31   # "bicyclist"     # 8: 32   # "motorcyclist"   # 9: 40   # "road"   
        10: 44  # "parking"            # 11: 48  # "sidewalk"   # 12: 49  # "other-ground"  # 13: 50  # "building"       # 14: 51  # "fence"          
        15: 70  # "vegetation"         # 16: 71  # "trunk"      # 17: 72  # "terrain"       # 18: 80  # "pole"           # 19: 81  # "traffic-sign"
        """

        y_pred[y_pred==10] = 44
        y_pred[y_pred==11] = 48
        y_pred[y_pred==12] = 49
        y_pred[y_pred==13] = 50
        y_pred[y_pred==14] = 51
        y_pred[y_pred==15] = 70
        y_pred[y_pred==16] = 71
        y_pred[y_pred==17] = 72
        y_pred[y_pred==18] = 80
        y_pred[y_pred==19] = 81
        y_pred[y_pred==1] = 10
        y_pred[y_pred==2] = 11
        y_pred[y_pred==3] = 15
        y_pred[y_pred==4] = 18
        y_pred[y_pred==5] = 20
        y_pred[y_pred==6] = 30
        y_pred[y_pred==7] = 31
        y_pred[y_pred==8] = 32
        y_pred[y_pred==9] = 40

        # save predictions
        pred_folder = os.path.join("./unetformer", "sequences", img_metas[0]['sequence_id'], "predictions") 
        if not os.path.exists(pred_folder):
            os.makedirs(pred_folder)
        y_pred_bin = y_pred.astype(np.uint16)
        y_pred_bin.tofile(os.path.join(pred_folder, img_metas[0]['frame_id'] + ".label"))

    def pack(self, array):
        """ convert a boolean array into a bitwise array. """
        array = array.reshape((-1))

        #compressing bit flags.
        # yapf: disable
        compressed = array[::8] << 7 | array[1::8] << 6  | array[2::8] << 5 | array[3::8] << 4 | array[4::8] << 3 | array[5::8] << 2 | array[6::8] << 1 | array[7::8]
        # yapf: enable

        return np.array(compressed, dtype=np.uint8)
    
class ConvResidualBlock(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels):
        super(ConvResidualBlock, self).__init__()
        
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        
        self.conv_l1 = nn.Sequential(
            nn.Conv3d(self.in_channels+self.hid_channels, self.hid_channels, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
        )
        
        self.conv_l2 = nn.Sequential(
            nn.Conv3d(self.hid_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(self.out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.conv_l1.apply(weight_xavier_init)
        self.conv_l2.apply(weight_xavier_init)
        
    def forward(self, input, residual):
        out = self.conv_l1(torch.cat([input, residual], dim=1))
        out = self.conv_l2(out)
        return out