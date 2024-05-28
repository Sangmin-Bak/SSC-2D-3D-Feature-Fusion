"""
3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
Paper URL: https://arxiv.org/abs/1606.06650
Author: Amir Aghdam
"""
import math
import numpy as np
from torch import nn
# from torch.nn.init import xavier_uniform_, uniform_, _calculate_fan_in_and_fan_out
from mmcv.cnn import xavier_init
# from torchsummary import summary
import torch
import time
from mmdet.models import HEADS
from mmdet.models import build_loss
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.transformer import build_positional_encoding
from projects.mmdet3d_plugin.voxformer.utils.ssc_loss import sem_scal_loss, geo_scal_loss, BCE_ssc_loss

def weight_xavier_init(submodule):
    if isinstance(submodule, nn.Conv3d) or isinstance(submodule, nn.ConvTranspose3d) or isinstance(submodule, nn.Linear):
        xavier_init(submodule, distribution='uniform', bias=0.)
    elif isinstance(submodule, nn.BatchNorm3d):
        submodule.weight.data.fill_(1.0)
        submodule.bias.data.zero_()

class Conv3DBlock(nn.Module):
    """
    The basic block for double 3x3x3 convolutions in the analysis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> desired number of output channels
    :param bottleneck -> specifies the bottlneck block
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    """
    
    def __init__(self, in_channels, out_channels, bottleneck = False):
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels= in_channels, out_channels=out_channels//2, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(num_features=out_channels//2),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels= out_channels//2, out_channels=out_channels, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)

        self._init_weights()
            
    def _init_weights(self):
        self.conv1.apply(weight_xavier_init)
        self.conv2.apply(weight_xavier_init)
            
    def forward(self, input):
        res = self.conv1(input)
        res = self.conv2(res)
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res




class UpConv3DBlock(nn.Module):
    """
    The basic block for upsampling followed by double 3x3x3 convolutions in the synthesis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> number of residual connections' channels to be concatenated
    :param last_layer -> specifies the last output layer
    :param num_classes -> specifies the number of output channels for dispirate classes
    -- forward()
    :param input -> input Tensor
    :param residual -> residual connection to be concatenated with input
    :return -> Tensor
    """

    def __init__(self, in_channels, res_channels=0, last_layer=False, num_classes=None):
        super(UpConv3DBlock, self).__init__()
        assert (last_layer==False and num_classes==None) or (last_layer==True and num_classes!=None), 'Invalid arguments'
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2, 2), stride=2),
            nn.BatchNorm3d(num_features=in_channels),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels+res_channels, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(num_features=in_channels//2),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(num_features=in_channels//2),
            nn.ReLU(inplace=True),
        )
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=in_channels//2, out_channels=num_classes, kernel_size=(1,1,1))
        else: 
            self.conv3 = None

        self._init_weights()

    def _init_weights(self):
        self.upconv1.apply(weight_xavier_init)
        self.conv1.apply(weight_xavier_init)
        self.conv2.apply(weight_xavier_init)
        if self.last_layer:
            xavier_init(self.conv3, distribution='uniform', bias=0.)
            
    def forward(self, input, residual=None):
        out = self.upconv1(input)
        if residual!=None: out = torch.cat((out, residual), 1)
        out = self.conv1(out)
        out = self.conv2(out)
        if self.last_layer: out = self.conv3(out)
        return out
        

@HEADS.register_module()
class UNet3DHead(nn.Module):
    """
    The 3D UNet model
    -- __init__()
    :param in_channels -> number of input channels
    :param num_classes -> specifies the number of output channels or masks for different classes
    :param level_channels -> the number of channels at each level (count top-down)
    :param bottleneck_channel -> the number of bottleneck channels 
    :param device -> the device on which to run the model
    -- forward()
    :param input -> input Tensor
    :return -> Tensor
    """
    
    def __init__(
            self, 
            in_channels, 
            num_classes=2, 
            level_channels=[64, 128, 256], 
            bottleneck_channel=512, 
            scale="1_1",
            bev_h=128,
            bev_w=128,
            bev_z=16,
            BCE_occ_loss=False,
            sem_scal_loss=False,
            geo_scal_loss=False,
            alpha=0.54,
            train_cfg=None,
            test_cfg=None):
        assert scale in ["1_1", "1_2", "1_4"]
        super(UNet3DHead, self).__init__()
        level_1_chnls, level_2_chnls, level_3_chnls = level_channels[0], level_channels[1], level_channels[2]
        self.a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_1_chnls)
        self.a_block2 = Conv3DBlock(in_channels=level_1_chnls, out_channels=level_2_chnls)
        self.a_block3 = Conv3DBlock(in_channels=level_2_chnls, out_channels=level_3_chnls)
        self.bottleNeck = Conv3DBlock(in_channels=level_3_chnls, out_channels=bottleneck_channel, bottleneck= True)
        self.s_block3 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_3_chnls)
        # if scale == "1_2":
        #     self.s_block2 = UpConv3DBlock(in_channels=level_3_chnls, res_channels=level_2_chnls, num_classes=num_classes, last_layer=True)
        # if scale == "1_1":
        self.s_block2 = UpConv3DBlock(in_channels=level_3_chnls, res_channels=level_2_chnls)
        self.s_block1 = UpConv3DBlock(in_channels=level_2_chnls, res_channels=level_1_chnls)
        # self.s_block1 = UpConv3DBlock(in_channels=level_2_chnls, res_channels=level_1_chnls, num_classes=num_classes, last_layer=True)
        self.fcn = nn.Conv3d(level_1_chnls, num_classes, kernel_size=1, stride=1)
        self.scale = scale
        self.bev_h = bev_h 
        self.bev_w = bev_w 
        self.bev_z = bev_z 
        self.BCE_occ_loss=BCE_occ_loss,
        self.sem_scal_loss=sem_scal_loss,
        self.geo_scal_loss=geo_scal_loss,
        
        self.class_frequencies_level1 =  np.array([5.41773033e09, 4.03113667e08])
        self.class_weights_level_1 = torch.from_numpy(
            1 / np.log(self.class_frequencies_level1 + 0.001)
        )
        self.alpha = alpha

    
    def forward(self, input, return_occ_pred=True):
        #Analysis path forward feed
        out, residual_level1 = self.a_block1(input)
        out, residual_level2 = self.a_block2(out)
        out, residual_level3 = self.a_block3(out)
        out, _ = self.bottleNeck(out)

        #Synthesis path forward feed
        if self.scale == "1_4":
            out = self.s_block3(out, residual_level3)
            return out
        if self.scale == "1_2":
            out = self.s_block3(out, residual_level3)
            out = self.s_block2(out, residual_level2)
            return out
        if self.scale == "1_1":
            out = self.s_block3(out, residual_level3)
            out = self.s_block2(out, residual_level2)
            out = self.s_block1(out, residual_level1)
            if return_occ_pred: 
                out = self.fcn(out)
            return out
    
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
    
    def step(self, ssc_pred, target, img_metas, step_type):
        """Training/validation function.
        Args:
            pred (dict[Tensor]): Segmentation output.
            img_metas: Meta information such as camera intrinsics.
            target: Semantic completion ground truth. 
            step_type: Train or test.
        Returns:
            loss or predictions
        """

        if step_type== "train":
            loss_dict = dict()

            # if self.BCE_occ_loss:
            class_weights_level_1 = self.class_weights_level_1.type_as(target)
            loss_occ = BCE_ssc_loss(ssc_pred, target, class_weights_level_1, self.alpha)
            loss_dict['loss_occ'] = loss_occ

            # if self.sem_scal_loss:
            loss_sem_scal = sem_scal_loss(ssc_pred, target)
            # loss_dict['loss_sem_scal'] = (1. - scale) * loss_sem_scal
            loss_dict['loss_sem_scal'] = loss_sem_scal

            # if self.geo_scal_loss:
            loss_geo_scal = geo_scal_loss(ssc_pred, target)
            # loss_dict['loss_geo_scal'] = (1. - scale) * loss_geo_scal
            loss_dict['loss_geo_scal'] = loss_geo_scal

            return loss_dict

        elif step_type== "val" or "test":
            # if img_metas[0]['sequence_id'] == '02' and img_metas[0]['frame_id'] == '004515':
            #     print(img_metas[0])
            if target is not None:
                y_true = target.cpu().numpy()
            else:
                y_true = None
            y_pred = ssc_pred.detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)

            result = dict()
            result['y_pred'] = y_pred
            result['y_true'] = y_true

            # y_pred_bin = self.pack(y_pred)
            # y_pred_bin.tofile("pretrain_temp.bin")


            # if self.save_flag:
            #     self.save_occ_pred(img_metas, y_pred)

            return result

    def training_step(self, pred, target, img_metas):
        """Training step.
        """
        return self.step(pred, target, img_metas, "train")

    def validation_step(self, pred, target, img_metas):
        """Validation step.
        """
        return self.step(pred, target, img_metas, "val")

    

# @HEADS.register_module()
# class UNet3DHead(nn.Module):
#     """
#     The 3D UNet model
#     -- __init__()
#     :param in_channels -> number of input channels
#     :param num_classes -> specifies the number of output channels or masks for different classes
#     :param level_channels -> the number of channels at each level (count top-down)
#     :param bottleneck_channel -> the number of bottleneck channels 
#     :param device -> the device on which to run the model
#     -- forward()
#     :param input -> input Tensor
#     :return -> Tensor
#     """
    
#     def __init__(
#             self, 
#             in_channels, 
#             num_classes=2, 
#             level_channels=[64, 128, 256], 
#             bottleneck_channel=512, 
#             scale="1_1",
#             bev_h=128,
#             bev_w=128,
#             bev_z=16,
#             BCE_occ_loss=False,
#             sem_scal_loss=False,
#             geo_scal_loss=False,
#             alpha=0.54,
#             train_cfg=None,
#             test_cfg=None):
#         super(UNet3DHead, self).__init__()
#         assert scale in ["1_1", "1_2"]
#         level_1_chnls, level_2_chnls, level_3_chnls = level_channels[0], level_channels[1], level_channels[2]
#         self.a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_1_chnls)
#         self.a_block2 = Conv3DBlock(in_channels=level_1_chnls, out_channels=level_2_chnls)
#         self.a_block3 = Conv3DBlock(in_channels=level_2_chnls, out_channels=level_3_chnls)
#         self.bottleNeck = Conv3DBlock(in_channels=level_3_chnls, out_channels=bottleneck_channel, bottleneck= True)
#         self.s_block3 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_3_chnls)
#         # if scale == "1_2":
#         #     self.s_block2 = UpConv3DBlock(in_channels=level_3_chnls, res_channels=level_2_chnls, num_classes=num_classes, last_layer=True)
#         # if scale == "1_1":
#         self.s_block2 = UpConv3DBlock(in_channels=level_3_chnls, res_channels=level_2_chnls)
#         # self.s_block1 = UpConv3DBlock(in_channels=level_2_chnls, res_channels=level_1_chnls)
#         self.s_block1 = UpConv3DBlock(in_channels=level_2_chnls, res_channels=level_1_chnls, last_layer=True, num_classes=2)
#         # self.s_block1 = UpConv3DBlock(in_channels=level_2_chnls, res_channels=level_1_chnls, num_classes=num_classes, last_layer=True)
#         self.scale = scale
#         self.bev_h = bev_h 
#         self.bev_w = bev_w 
#         self.bev_z = bev_z 
#         self.BCE_occ_loss=BCE_occ_loss,
#         self.sem_scal_loss=sem_scal_loss,
#         self.geo_scal_loss=geo_scal_loss,
        
#         self.class_frequencies_level1 =  np.array([5.41773033e09, 4.03113667e08])
#         self.class_weights_level_1 = torch.from_numpy(
#             1 / np.log(self.class_frequencies_level1 + 0.001)
#         )
#         self.alpha = alpha

    
#     def forward(self, img_metas, targets):
#         device = targets.device
#         input =  img_metas[0]['pseudo'].reshape(self.bev_h*2, self.bev_w*2, self.bev_z*2)
#         input = torch.from_numpy(input).unsqueeze(0).unsqueeze(0).to(device)
#         #Analysis path forward feed
#         out, residual_level1 = self.a_block1(input)
#         out, residual_level2 = self.a_block2(out)
#         out, residual_level3 = self.a_block3(out)
#         out, _ = self.bottleNeck(out)

#         #Synthesis path forward feed
#         if self.scale == "1_4":
#             out = self.s_block3(out, residual_level3)
#         if self.scale == "1_2":
#             out = self.s_block3(out, residual_level3)
#             out = self.s_block2(out, residual_level2)
#             return out
#         if self.scale == "1_1":
#             out = self.s_block3(out, residual_level3)
#             out = self.s_block2(out, residual_level2)
#             out = self.s_block1(out, residual_level1)
#         return out
    
#     def step(self, occ_pred, target, img_metas, step_type):
#         """Training/validation function.
#         Args:
#             pred (dict[Tensor]): Segmentation output.
#             img_metas: Meta information such as camera intrinsics.
#             target: Semantic completion ground truth. 
#             step_type: Train or test.
#         Returns:
#             loss or predictions
#         """

#         if step_type== "train":
#             loss_dict = dict()

#             # if self.BCE_occ_loss:
#             class_weights_level_1 = self.class_weights_level_1.type_as(target)
#             loss_occ = BCE_ssc_loss(occ_pred, target, class_weights_level_1, self.alpha)
#             # loss_dict['loss_occ'] = loss_occ

#             # if self.sem_scal_loss:
#             #     loss_sem_scal = sem_scal_loss(occ_pred, target)
#             #     # loss_dict['loss_sem_scal'] = (1. - scale) * loss_sem_scal
#             #     loss_dict['loss_sem_scal'] = loss_sem_scal

#             # if self.geo_scal_loss:
#             #     loss_geo_scal = geo_scal_loss(occ_pred, target)
#             #     # loss_dict['loss_geo_scal'] = (1. - scale) * loss_geo_scal
#             #     loss_dict['loss_geo_scal'] = loss_geo_scal

#             return loss_occ

#         elif step_type== "val" or "test":
#             # if img_metas[0]['sequence_id'] == '02' and img_metas[0]['frame_id'] == '004515':
#             #     print(img_metas[0])
#             if target is not None:
#                 y_true = target.cpu().numpy()
#             else:
#                 y_true = None
#             y_pred = ssc_pred.detach().cpu().numpy()
#             y_pred = np.argmax(y_pred, axis=1)

#             result = dict()
#             result['y_pred'] = y_pred
#             result['y_true'] = y_true

#             # y_pred_bin = self.pack(y_pred)
#             # y_pred_bin.tofile("pretrain_temp.bin")


#             # if self.save_flag:
#             #     self.save_occ_pred(img_metas, y_pred)

#             return result

#     def training_step(self, pred, target, img_metas):
#         """Training step.
#         """
#         return self.step(pred, target, img_metas, "train")

#     def validation_step(self, pred, target, img_metas):
#         """Validation step.
#         """
#         return self.step(pred, target, img_metas, "val")



# if __name__ == '__main__':
#     #Configurations according to the Xenopus kidney dataset
#     model = UNet3D(in_channels=3, num_classes=1)
#     start_time = time.time()
#     summary(model=model, input_size=(3, 16, 128, 128), batch_size=-1, device="cpu")
#     print("--- %s seconds ---" % (time.time() - start_time))