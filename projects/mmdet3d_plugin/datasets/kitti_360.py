# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

import os
from os import path as osp
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import glob
# import random
import copy
import mmcv
import torch
from torch.utils.data import Dataset
import numpy as np
from numpy.linalg import inv
from torchvision import transforms
from mmdet.datasets import DATASETS
from mmcv.parallel import DataContainer as DC
from projects.mmdet3d_plugin.voxformer.utils.ssc_metric import SSCMetrics

SPLITS = {
    'train':
    ('2013_05_28_drive_0004_sync', '2013_05_28_drive_0000_sync', '2013_05_28_drive_0010_sync',
     '2013_05_28_drive_0002_sync', '2013_05_28_drive_0003_sync', '2013_05_28_drive_0005_sync',
     '2013_05_28_drive_0007_sync'),
    'val': ('2013_05_28_drive_0006_sync', ),
    'test': ('2013_05_28_drive_0009_sync', ),
}

KITTI_360_CLASS_FREQ = torch.tensor([
    2264087502, 20098728, 104972, 96297, 1149426, 4051087, 125103, 105540713, 16292249, 45297267,
    14454132, 110397082, 6766219, 295883213, 50037503, 1561069, 406330, 30516166, 1950115
])

@DATASETS.register_module()
class KITTI360(Dataset):
    META_INFO = {
        'class_weights':
        1 / torch.log(KITTI_360_CLASS_FREQ + 1e-6),
        'class_names':
        ('empty', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'road',
         'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'terrain',
         'pole', 'traffic-sign', 'other-structure', 'other-object')
    }
    
    def __init__(
        self,
        split,
        test_mode,
        data_root,
        preprocess_root,
        temporal = [],
        eval_range = 51.2,
        depthmodel="msnet3d",
        nsweep=10,
        labels_tag = 'labels',
        query_tag = 'query_iou5203_pre7712_rec6153',
        color_jitter=None,
    ):
        super().__init__()
        
        self.data_root = data_root
        self.label_root = os.path.join(preprocess_root, labels_tag)
        self.query_tag = query_tag
        self.nsweep=str(nsweep)
        self.depthmodel = depthmodel
        self.eval_range = eval_range
        splits = {
            "train": ['2013_05_28_drive_0004_sync', '2013_05_28_drive_0000_sync', '2013_05_28_drive_0010_sync',
                      '2013_05_28_drive_0002_sync', '2013_05_28_drive_0003_sync', '2013_05_28_drive_0005_sync',
                    '2013_05_28_drive_0007_sync'],
            "val": ['2013_05_28_drive_0006_sync'],
            "test": ['2013_05_28_drive_0009_sync'],
        }
        self.split = split 
        self.sequences = splits[split]
        self.n_classes = 19
        self.class_names =  ['empty', 'car', 'bicycle', 'motorcycle', 'truck', 
                             'other-vehicle', 'person', 'road', 'parking', 'sidewalk', 
                             'other-ground', 'building', 'fence', 'vegetation', 'terrain',
                             'pole', 'traffic-sign', 'other-structure', 'other-object']
        self.metrics = SSCMetrics(self.n_classes)
        self.scene_size = (51.2, 51.2, 6.4)
        self.vox_origin = np.array([0, -25.6, -2])
        self.voxel_size = 0.2  # 0.2m

        self.img_W = 1408
        self.img_H = 376

        self.poses=self.load_poses()
        self.target_frames = temporal
        self.load_scans()
        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )
        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.test_mode = test_mode
        self.set_group_flag()
        

    def __getitem__(self, index):
        
        return self.prepare_data(index)

    def __len__(self):
        return len(self.scans)

    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out["P2"] = calib_all["P2"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
        return calib_out

    @staticmethod
    def parse_poses(filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses

    def load_poses(self):
        """ read poses for each sequence

            Returns
            -------
            dict
                pose dict for different sequences.
        """
        pose_dict = dict()
        for sequence in self.sequences:
            pose_path = os.path.join(self.data_root, "data_2d_raw", sequence, "poses.txt")
            # calib = self.read_calib(
            #     os.path.join(self.data_root, "dataset", "sequences", sequence, "calib.txt")
            # )
            calib = self.read_calib()
            pose_dict[sequence] = self.parse_poses(pose_path, calib)
        return pose_dict

    def load_scans(self):
        """ read each scan

            Returns
            -------
            list
                list of each single scan.
        """
        self.scans = []
        calib = self.read_calib()
        for sequence in self.sequences:
            # calib = self.read_calib(
            #     os.path.join(self.data_root, "data_2d_raw", sequence, "cam0_to_world.txt")
            # )
            P = calib["P2"]
            T_velo_2_cam = calib["Tr"]
            proj_matrix = P @ T_velo_2_cam

            # glob_path = os.path.join(
            #     self.data_root, "dataset", "sequences_" + self.depthmodel + "_sweep"+ self.nsweep, sequence, "queries", "*." + self.query_tag
            # )
            glob_path = os.path.join(
                self.data_root, "data_3d_voxel", "sequences_" + self.depthmodel + "_sweep"+ self.nsweep, sequence, "voxels", "*.pseudo"
            )
            
            for proposal_path in glob.glob(glob_path):

                self.scans.append(
                    {
                        "sequence": sequence,
                        "pose": self.poses[sequence],
                        "P": P,
                        "T_velo_2_cam": T_velo_2_cam,
                        "proj_matrix": proj_matrix,
                        "proposal_path": proposal_path,
                        # "pseudo_proposal_path": pseudo_proposal_path,
                    }
                )

    def set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def prepare_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        data_queue = []
        example = self.get_data_info(index)

        data_queue.insert(0, example)

        return self.union2one(data_queue)

    def union2one(self, queue):
        """
        convert sample queue into one single sample.
        """
        imgs_list = [each['img'] for each in queue]
        voxel_list = [each['voxel'] for each in queue]
        metas_map = {}
        
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas']

        queue[-1]['img'] = DC(torch.stack(imgs_list),
                              cpu_only=False, stack=True)
        queue[-1]['voxel'] = DC(torch.stack(voxel_list),
                              cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines.
        """
        scan = self.scans[index]

        proposal_path = scan["proposal_path"]
        # lidar_proposal_path = scan["lidar_proposal_path"]
        # pseudo_proposal_path = scan["pseudo_proposal_path"]

        sequence = scan["sequence"]
        filename = os.path.basename(proposal_path)
        frame_id = os.path.splitext(filename)[0]

        meta_dict = self.get_meta_info(scan, sequence, frame_id, proposal_path)
        img = self.get_input_info(sequence, frame_id)
        # voxel = self.get_voxel_info(sequence, frame_id)
        voxel = self.get_voxel_info(proposal_path)
        # depth = self.get_depth_info(sequence, frame_id)
        target = self.get_gt_info(sequence, frame_id)

        data_info = dict(
            img_metas = meta_dict,
            img = img,
            voxel = voxel,
            target = target
        )
        return data_info

    def get_meta_info(self, scan, sequence, frame_id, proposal_path):
        """Get meta info according to the given index.

        Args:
            scan (dict): scan information,
            sequence (str): sequence id,
            frame_id (str): frame id,
            proposal_path (str): proposal path.

        Returns:
            dict: Meta information that will be passed to the data \
                preprocessing pipelines.
        """
        rgb_path = os.path.join(
            self.data_root, "data_2d_raw", sequence, "image_00/data_rect", frame_id + ".png"
        )

        # for multiple images
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        image_paths = []

        # transform points from lidar to camera coordinate
        lidar2cam_rt = scan["T_velo_2_cam"]
        # camera intrisic
        P = scan["P"]
        cam_k = P[0:3, 0:3]
        intrinsic = cam_k
        viewpad = np.eye(4)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        # transform 3d point in lidar coordinate to 2D image (projection matrix)
        lidar2img_rt = (viewpad @ lidar2cam_rt)

        lidar2img_rts.append(lidar2img_rt)
        lidar2cam_rts.append(lidar2cam_rt)
        cam_intrinsics.append(intrinsic)
        image_paths.append(rgb_path)

        # for reference img
        seq_len = len(self.poses[sequence])
        for i in self.target_frames:
            id = int(frame_id)

            if id + i < 0 or id + i > seq_len-1:
                target_id = frame_id
            else:
                target_id = str(id + i).zfill(6)

            rgb_path = os.path.join(
                self.data_root, "data_2d_raw", sequence, "image_00/data_rect", frame_id + ".png"
            )

            pose_list = self.poses[sequence]

            ref = pose_list[int(frame_id)] # reference frame with GT semantic voxel
            target = pose_list[int(target_id)]
            ref2target = np.matmul(inv(target), ref) # both for lidar

            target2cam = scan["T_velo_2_cam"] # lidar to camera
            ref2cam = target2cam @ ref2target

            lidar2cam_rt  = ref2cam
            lidar2img_rt = (viewpad @ lidar2cam_rt)

            lidar2img_rts.append(lidar2img_rt)
            lidar2cam_rts.append(lidar2cam_rt)
            cam_intrinsics.append(intrinsic)
            image_paths.append(rgb_path)

        # proposal_bin = self.read_occupancy_SemKITTI(proposal_path)
        # lidar_proposal_bin = self.read_occupancy_SemKITTI(proposal_path)
        # pseudo_proposal_bin = self.read_occupancy_SemKITTI(pseudo_proposal_path)

        meta_dict = dict(
            sequence_id = sequence,
            frame_id = frame_id,
            proposal_path = proposal_path,
            # proposal=proposal_bin,
            # pseudo=pseudo_proposal_bin,
            img_filename=image_paths,
            lidar2img = lidar2img_rts,
            lidar2cam=lidar2cam_rts,
            cam_intrinsic=cam_intrinsics,
            img_shape = [(self.img_H,self.img_W)]
        )

        return meta_dict

    def get_input_info(self, sequence, frame_id):
        """Get the image of the specific frame in a sequence.

        Args:
            sequence (str): sequence id,
            frame_id (str): frame id.

        Returns:
            torch.tensor: Img.
        """
        seq_len = len(self.poses[sequence])
        image_list = []

        rgb_path = os.path.join(
            self.data_root, "data_2d_raw", sequence, "image_00/data_rect", frame_id + ".png"
        )
        img = Image.open(rgb_path).convert("RGB")
        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)
        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0
        img = img[:self.img_H, :self.img_W, :]  # crop image
        image_list.append(self.normalize_rgb(img))

        # reference frame
        for i in self.target_frames:
            id = int(frame_id)

            if id + i < 0 or id + i > seq_len-1:
                target_id = frame_id
            else:
                target_id = str(id + i).zfill(6)

            rgb_path = os.path.join(
                self.data_root, "data_2d_raw", sequence, "image_00/data_rect", frame_id + ".png"
            )
            img = Image.open(rgb_path).convert("RGB")
            # Image augmentation
            if self.color_jitter is not None:
                img = self.color_jitter(img)
            # PIL to numpy
            img = np.array(img, dtype=np.float32, copy=False) / 255.0
            img = img[:self.img_H, :self.img_W, :]  # crop image

            image_list.append(self.normalize_rgb(img))

        image_tensor = torch.stack(image_list, dim=0) #[N, 3, 370, 1220]

        return image_tensor
    
    def get_voxel_info(self, proposal_path):
        """Get the image of the specific frame in a sequence.

        Args:
            sequence (str): sequence id,
            frame_id (str): frame id.

        Returns:
            torch.tensor: Img.
        """
        # seq_len = len(self.poses[sequence])
        # image_list = []

        # voxel_path = os.path.join(
        #     self.data_root, "dataset", "sequences_msnet3d_sweep10", sequence, "voxels", frame_id + ".pseudo"
        # )
        # voxel = self.read_occupancy_SemKITTI(voxel_path)
        voxel = self.read_occupancy_SemKITTI(proposal_path)
        voxel_tensor = torch.from_numpy(voxel)
        voxel_tensor = voxel_tensor.reshape(256, 256, 32)

        return voxel_tensor
    
    def get_depth_info(self, sequence, frame_id):
        """Get the image of the specific frame in a sequence.

        Args:
            sequence (str): sequence id,
            frame_id (str): frame id.

        Returns:
            torch.tensor: Img.
        """
        seq_len = len(self.poses[sequence])
        image_list, depth_list = [], []

        # load depth image
        
        depth_path = os.path.join(
            self.data_root, "depth", "sequences", sequence, frame_id + ".npy"
        )
        depth = np.load(depth_path)
        depth = np.stack([depth for _ in range(3)], axis=0)
        if self.color_jitter is not None:
            depth = self.color_jitter(depth)
        depth = depth[:, :self.img_H, :self.img_W]  # crop image
        depth_list.append(torch.from_numpy(depth))

        # reference frame
        for i in self.target_frames:
            id = int(frame_id)

            if id + i < 0 or id + i > seq_len-1:
                target_id = frame_id
            else:
                target_id = str(id + i).zfill(6)

            depth_path = os.path.join(
            self.data_root, "depth", "sequences", sequence, target_id + ".npy"
            )
            depth = np.load(depth_path)
            depth = np.stack([depth for _ in range(3)], axis=-1)
            if self.color_jitter is not None:
                depth = self.color_jitter(depth)
            depth = depth[:self.img_H, :self.img_W, :]  # crop image
            depth_list.append(torch.from_numpy(depth))

        depth_tensor = torch.stack(depth_list, dim=0) #[N, 3, 370, 1220]

        return depth_tensor

    def get_gt_info(self, sequence, frame_id):
        """Get the ground truth.

        Args:
            sequence (str): sequence id,
            frame_id (str): frame id.

        Returns:
            array: target. 
        """
        # load full-range groundtruth
        target_1_path = os.path.join(self.label_root, sequence, frame_id + "_1_1.npy")
        # target_1_path = os.path.join(self.data_root, "dataset", "sequences", sequence, "voxels", frame_id + ".bin")
        target = np.load(target_1_path)
        # target = self.read_occupancy_SemKITTI(target_1_path).reshape(256, 256, 32)
        
        # short-range groundtruth
        if self.eval_range == 25.6:
            target[128:, :, :] = 255
            target[:, :64, :] = 255
            target[:, 192:, :] = 255

        elif self.eval_range == 12.8:
            target[64:, :, :] = 255
            target[:, :96, :] = 255
            target[:, 160:, :] = 255

        return target
    
    def read_SemKITTI(self, path, dtype, do_unpack):
        bin = np.fromfile(path, dtype=dtype)  # Flattened array
        if do_unpack:
            bin = self.unpack(bin)
        return bin

    def read_occupancy_SemKITTI(self, path):
        occupancy = self.read_SemKITTI(path, dtype=np.uint8, do_unpack=True).astype(np.float32)
        return occupancy

    def unpack(self, compressed):
        ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
        uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
        uncompressed[::8] = compressed[:] >> 7 & 1
        uncompressed[1::8] = compressed[:] >> 6 & 1
        uncompressed[2::8] = compressed[:] >> 5 & 1
        uncompressed[3::8] = compressed[:] >> 4 & 1
        uncompressed[4::8] = compressed[:] >> 3 & 1
        uncompressed[5::8] = compressed[:] >> 2 & 1
        uncompressed[6::8] = compressed[:] >> 1 & 1
        uncompressed[7::8] = compressed[:] & 1

        return uncompressed

    def pack(self, array):
        """ convert a boolean array into a bitwise array. """
        array = array.reshape((-1))
        compressed = array[::8] << 7 | array[1::8] << 6  | array[2::8] << 5 | array[3::8] << 4 | array[4::8] << 3 | array[5::8] << 2 | array[6::8] << 1 | array[7::8]
        return np.array(compressed, dtype=np.uint8)

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_name='ssc',
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in SemanticKITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        detail = dict()

        for result in results:
            self.metrics.add_batch(result['y_pred'], result['y_true'])
        metric_prefix = f'{result_name}_SemanticKITTI'

        stats = self.metrics.get_stats()
        for i, class_name in enumerate(self.class_names):
            detail["{}/SemIoU_{}".format(metric_prefix, class_name)] = stats["iou_ssc"][i]

        detail["{}/mIoU".format(metric_prefix)] = stats["iou_ssc_mean"]
        detail["{}/IoU".format(metric_prefix)] = stats["iou"]
        detail["{}/Precision".format(metric_prefix)] = stats["precision"]
        detail["{}/Recall".format(metric_prefix)] = stats["recall"]
        self.metrics.reset()

        return detail

    @staticmethod
    def read_calib():
        P = np.array([
            552.554261,
            0.000000,
            682.049453,
            0.000000,
            0.000000,
            552.554261,
            238.769549,
            0.000000,
            0.000000,
            0.000000,
            1.000000,
            0.000000,
        ]).reshape(3, 4)

        cam2velo = np.array([
            0.04307104361,
            -0.08829286498,
            0.995162929,
            0.8043914418,
            -0.999004371,
            0.007784614041,
            0.04392796942,
            0.2993489574,
            -0.01162548558,
            -0.9960641394,
            -0.08786966659,
            -0.1770225824,
        ]).reshape(3, 4)
        C2V = np.concatenate([cam2velo, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0)
        V2C = np.linalg.inv(C2V)
        V2C = V2C[:3, :]

        out = {}
        out['P2'] = P
        out['Tr'] = np.identity(4)
        out['Tr'][:3, :4] = V2C
        return out