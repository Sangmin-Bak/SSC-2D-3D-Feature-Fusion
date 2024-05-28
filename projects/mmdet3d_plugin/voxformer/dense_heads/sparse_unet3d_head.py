# """
# 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
# Paper URL: https://arxiv.org/abs/1606.06650
# Author: Amir Aghdam
# """

# import numpy as np
# from torch import nn
# import torch
# import spconv.pytorch as spconv

# class SparseConv3DBlock(nn.Module):
#     """
#     The basic block for double 3x3x3 convolutions in the analysis path
#     -- __init__()
#     :param in_channels -> number of input channels
#     :param out_channels -> desired number of output channels
#     :param bottleneck -> specifies the bottlneck block
#     -- forward()
#     :param input -> input Tensor to be convolved
#     :return -> Tensor
#     """

#     def __init__(self, in_channels, out_channels, bottleneck = False) -> None:
#         super(SparseConv3DBlock, self).__init__()
#         self.conv1 = spconv.SparseSequential(
#             spconv.SparseConv3d(in_channels= in_channels, out_channels=out_channels//2, kernel_size=3, padding=1),
#             nn.BatchNorm1d(num_features=out_channels//2),
#             nn.LeakyReLU(),
#         )
#         self.conv1[0].reset_parameters()
        
#         self.conv2 = spconv.SparseSequential(
#             spconv.SparseConv3d(in_channels= out_channels//2, out_channels=out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm1d(num_features=out_channels),
#             nn.LeakyReLU(),
#         )
#         self.conv2[0].reset_parameters()

#         self.bottleneck = bottleneck
#         if not bottleneck:
#             self.pooling = spconv.SparseMaxPool3d(kernel_size=2, stride=2)
        
    
#     def forward(self, input):
#         res = self.conv1(input)
#         res = self.conv2(res)
#         out = None
#         if not self.bottleneck:
#             out = self.pooling(res)
#         else:
#             out = res
#         return out, res

# class SparseUpConv3DBlock(nn.Module):
#     """
#     The basic block for upsampling followed by double 3x3x3 convolutions in the synthesis path
#     -- __init__()
#     :param in_channels -> number of input channels
#     :param out_channels -> number of residual connections' channels to be concatenated
#     :param last_layer -> specifies the last output layer
#     :param num_classes -> specifies the number of output channels for dispirate classes
#     -- forward()
#     :param input -> input Tensor
#     :param residual -> residual connection to be concatenated with input
#     :return -> Tensor
#     """

#     def __init__(self, in_channels, res_channels=0, num_voxels=None, last_layer=False, num_classes=None) -> None:
#         super(SparseUpConv3DBlock, self).__init__()
#         assert (last_layer==False and num_classes==None) or (last_layer==True and num_classes!=None), 'Invalid arguments'
#         self.upconv1 = spconv.SparseSequential(
#             spconv.SparseConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2),
#             nn.BatchNorm1d(num_features=in_channels),
#             nn.LeakyReLU(),
#         )
#         self.upconv1[0].reset_parameters()

#         self.conv1 = spconv.SparseSequential(
#             spconv.SparseConv3d(in_channels=in_channels+res_channels, out_channels=in_channels//2, kernel_size=3, padding=1),
#             nn.BatchNorm1d(num_features=in_channels//2),
#             nn.LeakyReLU(),
#         )
#         self.conv1[0].reset_parameters()

#         self.conv2 = spconv.SparseSequential(
#             spconv.SparseConv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=3, padding=1),
#             nn.BatchNorm1d(num_features=in_channels//2),
#             nn.LeakyReLU(),
#         )
#         self.conv2[0].reset_parameters()

#         self.last_layer = last_layer
#         if last_layer:
#             self.conv3 = spconv.SparseConv3d(in_channels=in_channels//2, out_channels=num_classes, kernel_size=1)
#             self.conv3.reset_parameters()
            
#         self.num_voxels = num_voxels
            
        
#     def forward(self, input, residual=None):
#         out = self.upconv1(input)
#         if residual!=None: 
#             # out_dense = out.dense()
#             # residual_dense = residual.dense()
#             # out_features = out._features
#             # residual_features = residual._features
#             # zeros = torch.zeros((out_features.shape[0]-residual_features.shape[0], residual_features.shape[1])).to(residual_features.device)
#             # residual_features = torch.cat([residual_features, zeros], dim=0)
#             # out._features = torch.cat([out_features, residual_features], dim=1)
#             out_torch = torch.cat([out.dense(), residual.dense()], dim=1).permute(0, 2, 3, 4, 1)
#             out = spconv.SparseConvTensor.from_dense(out_torch)
#             out = self.conv1(out)
#             out = self.conv2(out)
#         else:
#             out = self.conv1(out)
#             out = self.conv2(out)
#         if self.last_layer: out = self.conv3(out)
#         return out
    
# class SparseUNet3D(nn.Module):
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
#             BCE_occ_loss=True,
#             sem_scal_loss=True,
#             geo_scal_loss=True) -> None:
#         assert scale in ["1_1", "1_2"]
#         super(SparseUNet3D, self).__init__()
#         level_1_chnls, level_2_chnls, level_3_chnls = level_channels[0], level_channels[1], level_channels[2]
#         self.a_block1 = SparseConv3DBlock(in_channels=in_channels, out_channels=level_1_chnls)
#         self.a_block2 = SparseConv3DBlock(in_channels=level_1_chnls, out_channels=level_2_chnls)
#         self.a_block3 = SparseConv3DBlock(in_channels=level_2_chnls, out_channels=level_3_chnls)
#         self.bottleNeck = SparseConv3DBlock(in_channels=level_3_chnls, out_channels=bottleneck_channel, bottleneck= True)
#         self.s_block3 = SparseUpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_3_chnls)
#         # if scale == "1_2":
#         #     self.s_block2 = UpConv3DBlock(in_channels=level_3_chnls, res_channels=level_2_chnls, num_classes=num_classes, last_layer=True)
#         # if scale == "1_1":
#         self.s_block2 = SparseUpConv3DBlock(in_channels=level_3_chnls, res_channels=level_2_chnls)
#         self.s_block1 = SparseUpConv3DBlock(in_channels=level_2_chnls, res_channels=level_1_chnls)
#         # self.s_block1 = UpConv3DBlock(in_channels=level_2_chnls, res_channels=level_1_chnls, num_classes=num_classes, last_layer=True)
#         # self.class_frequencies_level1 =  np.array([5.41773033e09, 4.03113667e08])
#         # self.class_weights = torch.from_numpy(
#         #     1 / np.log(self.class_frequencies_level1 + 0.001)
#         # )
#         self.scale = scale
#         self.bev_h = bev_h 
#         self.bev_w = bev_w 
#         self.bev_z = bev_z 
#         self.BCE_occ_loss=BCE_occ_loss,
#         self.sem_scal_loss=sem_scal_loss,
#         self.geo_scal_loss=geo_scal_loss,

    
#     def forward(self, img_metas, targets):
#         device = targets.device
#         input =  img_metas[0]['pseudo'].reshape(self.bev_h*2, self.bev_w*2, self.bev_z*2)
#         input = torch.from_numpy(input).unsqueeze(0).to(device)
#         # indices = indices.type(torch.int32)
#         # input = spconv.SparseConvTensor(input, indices, spatial_shape, batch_size=1)
#         input = spconv.SparseConvTensor.from_dense(input.unsqueeze(-1))
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
#         if self.scale == "1_1":
#             out = self.s_block3(out, residual_level3)
#             out = self.s_block2(out, residual_level2)
#             out = self.s_block1(out, residual_level1)
            
#         return out.dense()