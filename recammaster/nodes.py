import numpy as np
import os, io
import torch
from PIL import Image
import math, time
script_directory = os.path.dirname(os.path.abspath(__file__))

class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)

def parse_matrix(matrix_str):
    rows = matrix_str.strip().split('] [')
    matrix = []
    for row in rows:
        row = row.replace('[', '').replace(']', '')
        matrix.append(list(map(float, row.split())))
    return np.array(matrix)

class WanVideoReCamMasterDefaultCamera:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "camera_type": ([
                "pan_right", 
                "pan_left",
                "tilt_up",
                "tilt_down",
                "zoom_in",
                "zoom_out",
                "translate_up",
                "translate_down",
                "arc_left",
                "arc_right",
                ], {"default": "pan_right", "tooltip": "Camera type to use"}),
            "latents": ("LATENT", {"tooltip": "source video"}),
        },
        }

    RETURN_TYPES = ("CAMERAPOSES",)
    RETURN_NAMES = ("camera_poses",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "https://github.com/KwaiVGI/ReCamMaster"

    def process(self, camera_type, latents):
        import json
        
        camera_data_path = os.path.join(script_directory, "recam_extrinsics.json")
        with open(camera_data_path, 'r') as file:
            cam_data = json.load(file)
        
        samples = latents["samples"].squeeze(0)
        C, T, H, W = samples.shape
        num_frames = (T - 1) * 4 + 1

        camera_type_map = {
            "pan_right": 1,
            "pan_left": 2,
            "tilt_up": 3,
            "tilt_down": 4,
            "zoom_in": 5,
            "zoom_out": 6,
            "translate_up": 7,
            "translate_down": 8,
            "arc_left": 9,
            "arc_right": 10,
        }

        cam_idx = list(range(num_frames))[::4]
        traj = [parse_matrix(cam_data[f"frame{idx}"][f"cam{int(camera_type_map[camera_type]):02d}"]) for idx in cam_idx]
        traj = np.stack(traj).transpose(0, 2, 1)
       
        return (traj,)
        
class WanVideoReCamMasterGenerateOrbitCamera:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 1000, "step": 1, "tooltip": "Number of frames to generate"}),
            "degrees": ("INT", {"default": 90, "min": -180, "max": 180, "step": 1, "tooltip": "Degrees to orbit"}),
        },
        }

    RETURN_TYPES = ("CAMERAPOSES",)
    RETURN_NAMES = ("camera_poses",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "https://github.com/KwaiVGI/ReCamMaster"

    def process(self, degrees, num_frames):

        def generate_orbit(num_frames=num_frames, degrees=degrees):
            camera_data = []
            center = np.array([3390, 1380, 240])  # Center point of orbit
            
            for i in range(num_frames):
                # Calculate angle from 0 to specified degrees
                angle = i * degrees / (num_frames - 1)
                angle_rad = np.radians(angle)
                
                # Calculate position - circular path around center
                x = center[0] - np.cos(angle_rad)
                y = center[1] - np.sin(angle_rad)
                z = center[2]
                
                # Calculate direction from camera to center point
                camera_pos = np.array([x, y, z])
                dir_to_center = center - camera_pos
                
                # Calculate the angle needed to face the center
                look_angle = np.arctan2(dir_to_center[1], dir_to_center[0])
                
                # Rotation matrix for facing the center (corrected)
                cos_look = np.cos(look_angle)
                sin_look = np.sin(look_angle)
                
                # Create transformation matrix directly
                transform = np.array([
                    [cos_look, -sin_look, 0, x],
                    [sin_look, cos_look, 0, y],
                    [0, 0, 1, z],
                    [0, 0, 0, 1]
                ])
                
                camera_data.append(transform)
                
            return camera_data
        
        # Generate orbit data
        camera_transforms = generate_orbit(num_frames=num_frames, degrees=degrees)
        
        traj = camera_transforms[::4]
        traj = np.stack(traj)
       
        return (traj,)
class WanVideoReCamMasterCameraEmbed:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "camera_poses": ("CAMERAPOSES",),
            "latents": ("LATENT", {"tooltip": "source video"}),
        },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", "CAMERAPOSES",)
    RETURN_NAMES = ("camera_embeds", "camera_poses",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "https://github.com/KwaiVGI/ReCamMaster"

    def process(self, camera_poses, latents):
        from einops import rearrange
        samples = latents["samples"].squeeze(0)
        C, T, H, W = samples.shape
        num_frames = (T - 1) * 4 + 1
        
        
        c2ws = []
        for c2w in camera_poses:
            c2w = c2w[:, [1, 2, 0, 3]]
            c2w[:3, 1] *= -1.
            c2w[:3, 3] /= 100
            c2ws.append(c2w)
        tgt_cam_params = [Camera(cam_param) for cam_param in c2ws]
        relative_poses = []
        for i in range(len(tgt_cam_params)):
            relative_pose = self.get_relative_pose([tgt_cam_params[0], tgt_cam_params[i]])
            relative_poses.append(torch.as_tensor(relative_pose)[:,:3,:][1])
        pose_embedding = torch.stack(relative_poses, dim=0)  # 21x3x4
        pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')

        seq_len = math.ceil((H * W) / 4 * ((num_frames - 1) // 4 + 1))
      
        embeds = {
            "max_seq_len": seq_len,
            "target_shape": samples.shape,
            "num_frames": num_frames,
            "recammaster": {
                "camera_embed": pose_embedding,
                "source_latents": samples
            }
        }

        return (embeds, camera_poses,)
    
    def get_relative_pose(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]

        cam_to_origin = 0
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -cam_to_origin],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

def get_c2w(w2cs, transform_matrix, relative_c2w=True):
    if relative_c2w:
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ np.linalg.inv(w2c) for w2c in w2cs[1:]]
    else:
        ret_poses = [np.linalg.inv(w2c) for w2c in w2cs]
    ret_poses = [transform_matrix @ x for x in ret_poses]
    return np.array(ret_poses, dtype=np.float32)

class ReCamMasterPoseVisualizer:
                
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "camera_poses": ("CAMERAPOSES",),
            "base_xval": ("FLOAT", {"default": 0.2,"min": 0, "max": 100, "step": 0.01}),
            "zval": ("FLOAT", {"default": 0.3,"min": 0, "max": 100, "step": 0.01}),
            "scale": ("FLOAT", {"default": 1.0,"min": 0.01, "max": 10.0, "step": 0.01}),
            "arrow_length": ("FLOAT", {"default": 1,"min": 0, "max": 100, "step": 0.01}),
            },
            }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "plot"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = """
Visualizes the camera poses, from Animatediff-Evolved CameraCtrl Pose  
or a .txt file with RealEstate camera intrinsics and coordinates, in a 3D plot. 
"""
        
    def plot(self, camera_poses, scale, base_xval, zval, arrow_length):
        import matplotlib as mpl
        mpl.use('Agg')
        
        import matplotlib.pyplot as plt
        from torchvision.transforms import ToTensor

        x_min = -2.0 * scale
        x_max = 2.0 * scale
        y_min = -2.0 * scale
        y_max = 2.0 * scale
        z_min = -2.0 * scale
        z_max = 2.0 * scale
        plt.rcParams['text.color'] = '#999999'
        self.fig = plt.figure(figsize=(18, 7))
        self.fig.patch.set_facecolor('#353535')
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_facecolor('#353535') # Set the background color here
        self.ax.grid(color='#999999', linestyle='-', linewidth=0.5)
        self.plotly_data = None  # plotly data traces
        self.ax.set_aspect("auto")
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_zlim(z_min, z_max)
        self.ax.set_xlabel('x', color='#999999')
        self.ax.set_ylabel('y', color='#999999')
        self.ax.set_zlabel('z', color='#999999')
        for text in self.ax.get_xticklabels() + self.ax.get_yticklabels() + self.ax.get_zticklabels():
            text.set_color('#999999')
        print('initialize camera pose visualizer')

        total_frames = len(camera_poses)

        w2cs = []
        for cam in camera_poses:
            if cam.shape[0] == 3:
                cam = np.vstack((cam, np.array([[0, 0, 0, 1]])))
            cam = cam[:, [1, 2, 0, 3]]
            cam[:3, 1] *= -1.
            w2cs.append(np.linalg.inv(cam))
        transform_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        c2ws = get_c2w(w2cs, transform_matrix, True)
        scale = max(max(abs(c2w[:3, 3])) for c2w in c2ws)
        if scale > 1e-3:  # otherwise, pan or tilt
            for c2w in c2ws:
                c2w[:3, 3] /= scale

        for frame_idx, c2w in enumerate(c2ws):
            self.extrinsic2pyramid(c2w, frame_idx / total_frames, hw_ratio=1, base_xval=base_xval, zval=(zval))
          
            if arrow_length > 0:
                pos = c2w[:3, 3]
                forward = c2w[:3, 2]
                arrow_start = pos + forward * base_xval
                arrow_length = arrow_length
                self.ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2], 
                    forward[0], forward[1], forward[2], 
                    color='black', length=arrow_length, arrow_length_ratio=0.1)

        # Create the colorbar
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=total_frames)
        colorbar = self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=self.ax, orientation='vertical')

        # Change the colorbar label
        colorbar.set_label('Frame', color='#999999') # Change the label and its color

        # Change the tick colors
        colorbar.ax.yaxis.set_tick_params(colors='#999999') # Change the tick color

        # Change the tick frequency
        # Assuming you want to set the ticks at every 10th frame
        ticks = np.arange(0, total_frames, 10)
        colorbar.ax.yaxis.set_ticks(ticks)
        
        plt.title('')
        plt.draw()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = Image.open(buf)
        tensor_img = ToTensor()(img)
        buf.close()
        tensor_img = tensor_img.permute(1, 2, 0).unsqueeze(0)
        return (tensor_img,)

    def extrinsic2pyramid(self, extrinsic, color_map='red', hw_ratio=9/16, base_xval=1, zval=3):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import matplotlib.pyplot as plt
        vertex_std = np.array([[0, 0, 0, 1],
                               [base_xval, -base_xval * hw_ratio, zval, 1],
                               [base_xval, base_xval * hw_ratio, zval, 1],
                               [-base_xval, base_xval * hw_ratio, zval, 1],
                               [-base_xval, -base_xval * hw_ratio, zval, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]

        color = color_map if isinstance(color_map, str) else plt.cm.rainbow(color_map)

        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))

    def customize_legend(self, list_label):
        from matplotlib.patches import Patch
        import matplotlib.pyplot as plt
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

NODE_CLASS_MAPPINGS = {
    "WanVideoReCamMasterCameraEmbed": WanVideoReCamMasterCameraEmbed,
    "ReCamMasterPoseVisualizer": ReCamMasterPoseVisualizer,
    "WanVideoReCamMasterGenerateOrbitCamera": WanVideoReCamMasterGenerateOrbitCamera,
    "WanVideoReCamMasterDefaultCamera": WanVideoReCamMasterDefaultCamera,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoReCamMasterCameraEmbed": "WanVideo ReCamMaster Camera Embed",
    "ReCamMasterPoseVisualizer": "ReCamMaster Pose Visualizer",
    "WanVideoReCamMasterGenerateOrbitCamera": "WanVideo ReCamMaster Generate Orbit Camera",
    "WanVideoReCamMasterDefaultCamera": "WanVideo ReCamMaster Default Camera",
    }
