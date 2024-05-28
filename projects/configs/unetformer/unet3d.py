work_dir = 'result/unet3d_occ'
_base_ = [
    '../_base_/default_runtime.py'
]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

_num_layers_cross_ = 3
_num_points_cross_ = 8
_num_layers_self_ = 2
_num_points_self_ = 8
_dim_ = 128
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1

_labels_tag_ = 'labels'
_num_cams_ = 1
_temporal_ = []
point_cloud_range = [0, -25.6, -2.0, 51.2, 25.6, 4.4]
voxel_size = [0.2, 0.2, 0.2]
# voxel_level_dims = [32, 64, 128]
voxel_level_dims = [64, 128, 256]

_sem_scal_loss_ = True
_geo_scal_loss_ = True
_depthmodel_= 'msnet3d'
_nsweep_ = 10
_query_tag_ = 'query_iou5203_pre7712_rec6153'

model = dict(
   type='UNet3D',
   pretrained=dict(img='ckpts/resnet50-19c8e357.pth'),
#    pretrained=dict(img='torchvision://resnet50'),
   img_backbone=None,
   img_neck=None,
   occ_head=dict(
       type='UNet3DHead',
       in_channels=1, 
       num_classes=2,
       level_channels=voxel_level_dims, 
       bottleneck_channel=voxel_level_dims[-1]*2, 
       scale="1_1",
       BCE_occ_loss=True,
       sem_scal_loss=True,
       geo_scal_loss=True,
    #    focal_loss=dict(
    #         type='FocalLoss',
    #         use_sigmoid=True,
    #         gamma=2.0,
    #         alpha=0.25,
    #         loss_weight=1.0),
       alpha=0.54,
   ),
   train_cfg=dict(pts=dict(
       grid_size=[512, 512, 1],
       voxel_size=voxel_size,
       point_cloud_range=point_cloud_range,
       out_size_factor=4)))


dataset_type = 'SemanticKittiDatasetStage1'
data_root = './kitti/'
file_client_args = dict(backend='disk')

data = dict(
   samples_per_gpu=1,
   workers_per_gpu=4,
   train=dict(
       type=dataset_type,
       split = "train",
       test_mode=False,
       data_root=data_root,
       preprocess_root=data_root + 'dataset',
    #    eval_range = 51.2,
       depthmodel=_depthmodel_,
       nsweep=_nsweep_,
    #    temporal = _temporal_,
    #    labels_tag = _labels_tag_,
    #    query_tag = _query_tag_),
   ),
   val=dict(
       type=dataset_type,
       split = "val",
       test_mode=True,
       data_root=data_root,
       preprocess_root=data_root + 'dataset',
    #    eval_range = 51.2,
       depthmodel=_depthmodel_,
       nsweep=_nsweep_,
    #    temporal = _temporal_,
    #    labels_tag = _labels_tag_,
    #    query_tag = _query_tag_),
   ),
   test=dict(
       type=dataset_type,
       split = "val",
       test_mode=True,
       data_root=data_root,
       preprocess_root=data_root + 'dataset',
    #    eval_range = 51.2,
       depthmodel=_depthmodel_,
       nsweep=_nsweep_,
    #    temporal = _temporal_,
    #    labels_tag = _labels_tag_,
    #    query_tag = _query_tag_),
   ),
   shuffler_sampler=dict(type='DistributedGroupSampler'),
   nonshuffler_sampler=dict(type='DistributedSampler')
)
optimizer = dict(
   type='AdamW',
   lr=2e-4,
   weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
   policy='CosineAnnealing',
   warmup='linear',
   warmup_iters=500,
   warmup_ratio=1.0 / 3,
   min_lr_ratio=1e-3)
total_epochs = 20
evaluation = dict(interval=1)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
log_config = dict(
   interval=50,
   hooks=[
       dict(type='TextLoggerHook'),
       dict(type='TensorboardLoggerHook')
   ])

# checkpoint_config = None
checkpoint_config = dict(interval=1)