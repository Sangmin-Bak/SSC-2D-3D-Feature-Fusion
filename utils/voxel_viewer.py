import time
import os, sys
import glfw
import numpy as np
import torch
import torch.nn.functional as F

import OpenGL
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.GL import *

import imgui
from imgui.integrations.glfw import GlfwRenderer

OpenGL.ERROR_ON_COPY = True
OpenGL.ERROR_CHECKING = True

import argparse
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append('/home/sangmin/workspace/semantic-kitti-api')

import auxiliary.glow as glow
from auxiliary.camera import Camera


from visualize_voxels import Window, unpack

def get_parser():
    parser = argparse.ArgumentParser("./visualize_voxels.py")
    parser.add_argument(
        '--dataset',
        '-d',
        type=str,
        default='./kitti/dataset',
        # required=True,
        help='Dataset to visualize. No Default',
    )

    parser.add_argument(
        '--sequence',
        '-s',
        type=str,
        default="08",
        required=False,
        help='Sequence to visualize. Defaults to %(default)s',
    )
    
    parser.add_argument(
        '--voxel_dir',
        type=str,
        default='./kitti/preprocess/sequences_msnet3d_sweep1',
        required=False,
    )
    
    parser.add_argument(
        '--query_dir',
        type=str,
        default='sequences_msnet3d_sweep10',
        required=False,
    )
    
    parser.add_argument(
        '--label_dir',
        type=str,
        default='labels',
        required=False,
    )
    
    parser.add_argument(
        '--result_dir',
        type=str,
        default='predictions',
        required=False,
    )
    
    parser.add_argument(
        '--outputs',
        type=str,
        default='./voxel_query_vis',
        required=False,
    )
    
    parser.add_argument(
        '--qpn_labels',
        type=str,
        default='./kitti/dataset/labels',
        required=False,
    )
    # parser.add_argument('--config', type=str, default='./projects/configs/voxformer/voxformer-T.py', help='test config file path')
    # parser.add_argument('--checkpoint', type=str, default='./result/voxformer-T/voxformer-T/miou13.35_iou44.15_epoch_12.pth', help='checkpoint file')
    
    return parser

def get_ref_3d():
    """Get reference points in 3D.
    Args:
        self.real_h, self.bev_h
    Returns:
        vox_coords (Array): Voxel indices
        ref_3d (Array): 3D reference points
    """
    bev_h, bev_w, bev_z = 128, 128, 16
    scene_size = (51.2, 51.2, 6.4)
    vox_origin = np.array([0, -25.6, -2])
    voxel_size = 51.2 / bev_h

    vol_bnds = np.zeros((3,2))
    vol_bnds[:,0] = vox_origin
    vol_bnds[:,1] = vox_origin + np.array(scene_size)

    # Compute the voxels index in lidar cooridnates
    vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
    idx = np.array([range(vol_dim[0]*vol_dim[1]*vol_dim[2])])
    xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing='ij')
    vox_coords = np.concatenate([xv.reshape(1,-1), yv.reshape(1,-1), zv.reshape(1,-1), idx], axis=0).astype(int).T

    # Normalize the voxels centroids in lidar cooridnates
    ref_3d = np.concatenate([(xv.reshape(1,-1)+0.5)/bev_h, (yv.reshape(1,-1)+0.5)/bev_w, (zv.reshape(1,-1)+0.5)/bev_z,], axis=0).astype(np.float64).T 

    return vox_coords, ref_3d

def pred_label(data):
    data[data==10] = 44
    data[data==11] = 48
    data[data==12] = 49
    data[data==13] = 50
    data[data==14] = 51
    data[data==15] = 70
    data[data==16] = 71
    data[data==17] = 72
    data[data==18] = 80
    data[data==19] = 81
    data[data==1] = 10
    data[data==2] = 11
    data[data==3] = 15
    data[data==4] = 18
    data[data==5] = 20
    data[data==6] = 30
    data[data==7] = 31
    data[data==8] = 32
    data[data==9] = 40
    data[data==255] = 0
    return data
    
class VoxelViewer(Window):
    def __init__(self):
        super(VoxelViewer, self).__init__()
        
        # self.cam.lookAt(0.0, 90.0, 50.0, 0.0, 25.0, -20.0)
        # self.cam.lookAt(0.0, 50.0, 50.0, 0.0, 0.0, -20.0)
        # self.program["use_label_colors"] = False
        self.predict_dirs = './voxformer/sequences/08/predictions'
        
    def open_directory(self, voxel_dir, sequences_dir, gt_dir, result_dir, qpn_label_directory):
        """ open given sequences directory and get filenames of relevant files. """
        self.subdirs = [subdir for subdir in ["voxels", "queries"] if os.path.exists(os.path.join(sequences_dir, subdir))]

        if len(self.subdirs) == 0: raise RuntimeError("Neither 'voxels' nor 'predictions' found in " + sequences_dir)

        self.availableData = {}
        self.data = {}
        # self.subdirs = []
        self.bev_embed = torch.nn.Embedding((128) * (128) * (16), 128) 
        
        for subdir in self.subdirs:
            self.availableData[subdir] = []
            self.data[subdir] = {}
            complete_path = os.path.join(sequences_dir, subdir)
            files = os.listdir(complete_path)

            # data = sorted([os.path.join(complete_path, f) for f in files if f.endswith(".pseudo")]) 
            # if len(data) > 0:
            #     self.availableData[subdir].append("input")
            #     self.data[subdir]["input"] = data
            #     self.num_scans = len(data)
                # self.subdirs.append("input")

            data = sorted([os.path.join(complete_path, f) for f in files if f.endswith(".query_iou5203_pre7712_rec6153")])
            # data = sorted([os.path.join(complete_path, f) for f in files if f.endswith(".query_iou5203_pre7712_rec6153")])
            if len(data) > 0:
                self.availableData[subdir].append("query")
                self.data[subdir]["query"] = data
                self.num_scans = len(data)
                # self.subdirs.append("queries")
                
        # complete_path = os.path.join(voxel_dir)
        # files = os.listdir(complete_path)
        # data = sorted([os.path.join(complete_path, f) for f in files if f.endswith(".pseudo")])

        # if len(data) > 0:
        #     self.availableData['voxels'].append('input')
        #     self.data['voxels']['input'] = data
        #     self.num_scans = len(data)
        
        files = os.listdir(gt_dir)
        data = sorted([os.path.join(gt_dir, f) for f in files if f.endswith(".npy")])
        data = [file for i, file in enumerate(data) if i % 2 == 0]

        if len(data) > 0:
            self.availableData['voxels'].append('labels')
            self.data['voxels']['labels'] = data
            self.num_scans = len(data)
        
        complete_path = os.path.join(result_dir)
        files = os.listdir(complete_path)
        data = sorted([os.path.join(complete_path, f) for f in files if f.endswith(".label")])

        if len(data) > 0:
            self.availableData['voxels'].append('pred')
            self.data['voxels']['pred'] = data
            self.num_scans = len(data)
            
        complete_path = os.path.join(qpn_label_directory)
        files = os.listdir(complete_path)
        data = sorted([os.path.join(complete_path, f) for f in files if f.endswith(".npy")])
        data = [data[i] for i in range(len(data)) if (i+1) % 2 == 0]

        if len(data) > 0:
            self.availableData['voxels'].append('qpn_label')
            self.data['voxels']['qpn_label'] = data
            self.num_scans = len(data)    
        
        # self.availableData['updated query'] = './voxformer_query_vis/updated_queries.npy'
        # self.availableData['voxel feature'] = './voxformer_query_vis/voxel_features.npy'

        self.current_subdir = 0
        self.current_data = self.availableData[self.subdirs[self.current_subdir]][0]

        self.currentTimestep = 0
        self.sliderValue = 0

        self.lastChange = None
        self.lastUpdate = time.time()
        self.button_backward_hold = False
        self.button_forward_hold = False

        # todo: modify based on available stuff.
        self.showQuery = (self.current_data == "query")
        self.showInput = (self.current_data == "input")
        self.showLabels = (self.current_data == "labels")
        self.showPredict = (self.current_data == "pred")
        self.showQPNLabels = (self.current_data == "qpn_label")
        # self.showUpdatedQuery = False
        # self.showVoxelFeature = False
        # self.showCrossAttnWeights = False
        
    def setQueryData(self, data_name, t):
        # update buffer content with given data identified by data_name.
        subdir = self.subdirs[self.current_subdir]

        if len(self.data[subdir][data_name]) < t: return False

        # Note: uint with np.uint32 did not work as expected! (with instances and uint32 this causes problems!)
        if data_name == "query":
            buffer_data = unpack(np.fromfile(self.data[subdir][data_name][t], dtype=np.uint8)).astype(np.float32)
        else:
            return False
        
        self.label_vbo.assign(buffer_data)

        return True    
    
    def setCurrentBufferData(self, data_name, t, subdir_idx=0):
        # update buffer content with given data identified by data_name.
        if subdir_idx is not None: 
            subdir = self.subdirs[subdir_idx]

            if len(self.data[subdir][data_name]) < t: return False

        # Note: uint with np.uint32 did not work as expected! (with instances and uint32 this causes problems!)
        bev_queries = self.bev_embed.weight.unsqueeze(1).detach().numpy()
        vox_coors, ref_3d = get_ref_3d()
        if data_name == "query":
            data = np.fromfile(self.data[subdir][data_name][t], dtype=np.uint8)
            # buffer_data = pred_label(buffer_data)
            data = unpack(data).astype(np.float32)
            unmasked_idx = np.asarray(np.where(data.reshape(-1)>0)).astype(np.int32)
            data[unmasked_idx[0]] = 10
            buffer_data = data
            # coors = vox_coors[unmasked_idx[0], 3]
            # unmasked_bev_queries = bev_queries[coors, :, :]tkd
            # unmasked_ref_3d = ref_3d[vox_coors[unmasked_idx[0], 3], :]
            # buffer_data = unmasked_bev_queries
        elif data_name == "labels":
            buffer_data = np.load(self.data[subdir][data_name][t]).reshape(-1).astype(np.float32)
            buffer_data = pred_label(buffer_data)
        elif data_name == "pred":
            # if len(self.data[subdir]["labels"]) < t: return False
            buffer_data = np.fromfile(self.data[subdir][data_name][t], dtype=np.uint16).astype(np.float32)
            # buffer_data = pred_label(pred_label)
        elif data_name == "qpn_label":
            buffer_data = np.load(self.data[subdir][data_name][t]).astype(np.float32)
            buffer_data = pred_label(buffer_data)
        else:   # input voxel
            buffer_data = unpack(np.fromfile(self.data[subdir][data_name][t], dtype=np.uint8)).astype(np.float32)
            # buffer_data = F.interpolate(
            #     input=torch.from_numpy(buffer_data).view(1, 1, 256, 256, 32),
            #     size=(128, 128, 16),
            #     mode='trilinear',
            #     align_corners=True,
            # ).view(-1).numpy()
        self.label_vbo.assign(buffer_data)

        return True
        
    def run(self):
        # Loop until the user closes the window
        while not glfw.window_should_close(self.window):
        # Poll for and process events
            glfw.poll_events()

            # build gui.
            self.impl.process_inputs()

            w, h = glfw.get_window_size(self.window)
            glViewport(0, 0, w, h)

            imgui.new_frame()

            timeline_height = 35
            imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 0)
            imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 0)

            imgui.set_next_window_position(0, h - timeline_height - 10)
            imgui.set_next_window_size(w, timeline_height)

            imgui.begin("Timeline", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_SCROLLBAR)
            imgui.columns(1)
            imgui.same_line(0, 0)
            imgui.push_item_width(-50)
            changed, value = imgui.slider_int("", self.sliderValue, 0, self.num_scans - 1)
            if changed: self.sliderValue = value
            if self.sliderValue != self.currentTimestep:
                self.currentTimestep = self.sliderValue

            imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 3)

            play_delay = 1
            refresh_rate = 0.05

            current_time = time.time()

            imgui.same_line(spacing=5)
            changed = imgui.button("<", 20)
            if self.currentTimestep > 0:
                # just a click
                if changed:
                    self.currentTimestep = self.sliderValue = self.currentTimestep - 1
                    self.lastUpdate = current_time

                # button pressed.
                if imgui.is_item_active() and not self.button_backward_hold:
                    self.hold_start = current_time
                    self.button_backward_hold = True

                if not imgui.is_item_active() and self.button_backward_hold:
                    self.button_backward_hold = False

                # start playback when button pressed long enough
                if self.button_backward_hold and ((current_time - self.hold_start) > play_delay):
                    if (current_time - self.lastUpdate) > refresh_rate:
                        self.currentTimestep = self.sliderValue = self.currentTimestep - 1
                        self.lastUpdate = current_time

            imgui.same_line(spacing=2)
            changed = imgui.button(">", 20)

            if self.currentTimestep < self.num_scans - 1:
                # just a click
                if changed:
                    self.currentTimestep = self.sliderValue = self.currentTimestep + 1
                    self.lastUpdate = current_time

                # button pressed.
                if imgui.is_item_active() and not self.button_forward_hold:
                    self.hold_start = current_time
                    self.button_forward_hold = True

                if not imgui.is_item_active() and self.button_forward_hold:
                    self.button_forward_hold = False

                # start playback when button pressed long enough
                if self.button_forward_hold and ((current_time - self.hold_start) > play_delay):
                    if (current_time - self.lastUpdate) > refresh_rate:
                        self.currentTimestep = self.sliderValue = self.currentTimestep + 1
                        self.lastUpdate = current_time

            imgui.pop_style_var(3)
            imgui.end()

            imgui.set_next_window_position(20, 20, imgui.FIRST_USE_EVER)
            imgui.set_next_window_size(200, 150, imgui.FIRST_USE_EVER)
            imgui.begin("Show Data")

            if len(self.subdirs) > 1:
                for i, subdir in enumerate(self.subdirs):
                    changed, value = imgui.checkbox(subdir, self.current_subdir == i)
                    if i < len(self.subdirs) - 1: imgui.same_line()
                    if changed and value: self.current_subdir = i

            subdir = self.subdirs[self.current_subdir]

            data_available = "input" in self.availableData[subdir]
            if data_available:
                imgui.push_style_var(imgui.STYLE_ALPHA, 1.0)
            else:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)

            changed, value = imgui.checkbox("input", self.showInput)
            if changed and value and data_available:
                self.showInput = True
                self.showLabels = False
                self.showQuery = False
                self.showPredict = False
                self.showQPNLabels = False

            imgui.pop_style_var()
            
            data_available = "query" in self.availableData["queries"]
            if data_available:
                imgui.push_style_var(imgui.STYLE_ALPHA, 1.0)
            else:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)

            changed, value = imgui.checkbox("query", self.showQuery)
            if changed and value and data_available:
                self.showInput = False
                self.showLabels = False
                self.showQuery = True
                self.showPredict = False
                self.showQPNLabels = False
                # self.showCrossAttnWeights = False
                
            imgui.pop_style_var()
            
            data_available = "labels" in self.availableData[subdir]
            if data_available:
                imgui.push_style_var(imgui.STYLE_ALPHA, 1.0)
            else:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)

            changed, value = imgui.checkbox("labels", self.showLabels)
            if changed and value and data_available:
                self.showInput = False
                self.showLabels = True
                self.showQuery = False
                self.showPredict = False
                self.showQPNLabels = False
                
            imgui.pop_style_var()
            
            data_available = "pred" in self.availableData[subdir]
            if data_available:
                imgui.push_style_var(imgui.STYLE_ALPHA, 1.0)
            else:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)
                
            changed, value = imgui.checkbox("predictions", self.showPredict)
            if changed and value and data_available:
                self.showInput = False
                self.showLabels = False
                self.showQuery = False
                self.showPredict = True
                self.showQPNLabels = False
                
            imgui.pop_style_var()
                
            data_available = "qpn_label" in self.availableData[subdir]
            if data_available:
                imgui.push_style_var(imgui.STYLE_ALPHA, 1.0)
            else:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.3)
                
            changed, value = imgui.checkbox("qpn_label", self.showQPNLabels)
            if changed and value and data_available:
                self.showInput = False
                self.showLabels = False
                self.showQuery = False
                self.showPredict = False
                self.showQPNLabels = True

            imgui.pop_style_var()

            imgui.end()

            # imgui.show_demo_window()

            # showData = ["input"]
            showData = []
            if self.showInput: showData.append("input")
            if self.showQuery: showData.append("query")
            if self.showPredict: showData.append("pred")
            if self.showQPNLabels: showData.append("qpn_label")

            mvp = self.projection_ @ self.cam.matrix @ self.conversion_

            glClearColor(1.0, 1.0, 1.0, 0.0)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glBindVertexArray(self.vao)

            self.program.bind()
            self.program["mvp"] = mvp.transpose()
            self.program["view_mat"] = (self.cam.matrix @ self.conversion_).transpose()
            self.program["lightPos"] = glow.vec3(10, 10, 10)
            self.program["voxel_scale"] = 0.8
            self.program["voxel_alpha"] = 0.1
            self.program["use_label_colors"] = True

            self.label_colors.bind(0)

            if self.showLabels:
                self.program['voxel_dims'] = glow.ivec3(256, 256, 32)
                self.setCurrentBufferData("labels", self.currentTimestep)
                glDrawArraysInstanced(GL_TRIANGLES, 0, 36, self.num_instances)
                
            self.program["use_label_colors"] = False
            self.program["voxel_color"] = glow.vec3(0.3, 0.3, 0.3)

            self.program["voxel_alpha"] = 0.8
            
            # if self.showQuery:
            #     self.setCurrentBufferData("query", self.currentTimestep, subdir_idx=1)
            #     glDrawArraysInstanced(GL_TRIANGLES, 0, 36, self.num_instances)
            # if self.showUpdatedQuery:
            #     self.setCurrentBufferData("updated query", self.currentTimestep, subdir_idx=None)
            #     self.program["voxel_color"] = glow.vec3(0.3, 1.0, 0.5)
            #     glDrawArraysInstanced(GL_TRIANGLES, 0, 36, self.num_instances)
            # if self.showVoxelFeature:
            #     self.setCurrentBufferData("voxel feature", self.currentTimestep, subdir_idx=None)
            #     self.program["voxel_color"] = glow.vec3(1.0, 0.3, 0.5)
            #     glDrawArraysInstanced(GL_TRIANGLES, 0, 36, self.num_instances)
            # if self.showCrossAttnWeights:
            #     self.setCurrentBufferData("cross-attention weights", self.currentTimestep, subdir_idx=None)
            #     self.program["voxel_color"] = glow.vec3(1.0, 1.0, 0.0)
            #     glDrawArraysInstanced(GL_TRIANGLES, 0, 36, self.num_instances)

            for data_name in showData:
                self.program["voxel_scale"] = 0.5
                # self.program["voxel_scale"] = 0.8
                if data_name == "input": 
                    self.program['voxel_dims'] = glow.ivec3(256, 256, 32)
                    self.program["voxel_scale"] = 0.8
                    self.setCurrentBufferData(data_name, self.currentTimestep, subdir_idx=0)
                elif data_name == "query": 
                    self.program['voxel_dims'] = glow.ivec3(128, 128, 16)
                    self.program["voxel_scale"] = 0.8
                    self.setCurrentBufferData(data_name, self.currentTimestep, subdir_idx=1)
                elif data_name == "pred": 
                    self.program["use_label_colors"] = True
                    self.program['voxel_dims'] = glow.ivec3(256, 256, 32)
                    self.program["voxel_scale"] = 0.8
                    self.setCurrentBufferData(data_name, self.currentTimestep, subdir_idx=0)
                elif data_name == "qpn_label": 
                    self.program["use_label_colors"] = True
                    self.program['voxel_dims'] = glow.ivec3(128, 128, 16)
                    self.program["voxel_scale"] = 0.8
                    self.setCurrentBufferData(data_name, self.currentTimestep, subdir_idx=0)
                    
                glDrawArraysInstanced(GL_TRIANGLES, 0, 36, self.num_instances)

            self.program.release()
            self.label_colors.release(0)

            glBindVertexArray(self.vao_no_points)

            self.prgDrawPose.bind()
            self.prgDrawPose["mvp"] = mvp.transpose()
            self.prgDrawPose["pose"] = np.identity(4, dtype=np.float32)
            self.prgDrawPose["size"] = 1.0

            glDrawArrays(GL_POINTS, 0, 1)
            self.prgDrawPose.release()

            glBindVertexArray(0)

            # draw gui ontop.
            imgui.render()
            self.impl.render(imgui.get_draw_data())

            # Swap front and back buffers
            glfw.swap_buffers(self.window)
        
        
if __name__ == '__main__':
    
    parser = get_parser()

    FLAGS, unparsed = parser.parse_known_args()

    voxel_directory = os.path.join(FLAGS.voxel_dir, FLAGS.sequence, "voxels")
    sequence_directory = os.path.join(FLAGS.dataset, FLAGS.query_dir, FLAGS.sequence)
    label_directory = os.path.join(FLAGS.dataset, FLAGS.label_dir, FLAGS.sequence)
    # result_directory = os.path.join(FLAGS.dataset, "sequences", FLAGS.sequence, FLAGS.result_dir)
    result_directory = FLAGS.result_dir
    qpn_label_directory = os.path.join(FLAGS.qpn_labels, FLAGS.sequence)

    window = VoxelViewer()
    window.open_directory(voxel_directory, sequence_directory, label_directory, result_directory, qpn_label_directory)

    window.run()

    glfw.terminate()    