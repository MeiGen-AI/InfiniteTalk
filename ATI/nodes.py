import json
from .motion import process_tracks
import numpy as np
from typing import List, Tuple
import torch
FIXED_LENGTH = 121
def pad_pts(tr):
    """Convert list of {x,y} to (FIXED_LENGTH,1,3) array, padding/truncating."""
    pts = np.array([[p['x'], p['y'], 1] for p in tr], dtype=np.float32)
    n = pts.shape[0]
    if n < FIXED_LENGTH:
        pad = np.zeros((FIXED_LENGTH - n, 3), dtype=np.float32)
        pts = np.vstack((pts, pad))
    else:
        pts = pts[:FIXED_LENGTH]
    return pts.reshape(FIXED_LENGTH, 1, 3)

def age_to_bgr(ratio: float) -> Tuple[int,int,int]:
    """
    Map ratio∈[0,1] through: 0→blue, 1/3→green, 2/3→yellow, 1→red.
    Returns (B,G,R) for OpenCV.
    """
    if ratio <= 1/3:
        # blue→green
        t = ratio / (1/3)
        b = int(255 * (1 - t))
        g = int(255 * t)
        r = 0
    elif ratio <= 2/3:
        # green→yellow
        t = (ratio - 1/3) / (1/3)
        b = 0
        g = 255
        r = int(255 * t)
    else:
        # yellow→red
        t = (ratio - 2/3) / (1/3)
        b = 0
        g = int(255 * (1 - t))
        r = 255
    return (r, g, b)

def paint_point_track(
    frames: np.ndarray,
    point_tracks: np.ndarray,
    visibles: np.ndarray,
    min_radius: int = 1,
    max_radius: int = 6,
    max_retain: int = 50
) -> np.ndarray:
    """
    Draws every past point of each track on each frame, with radius and color
    interpolated by the point's age (old→small to new→large).

    Args:
      frames:      [F, H, W, 3] uint8 RGB
      point_tracks:[N, F, 2] float32  – (x,y) in pixel coords
      visibles:    [N, F] bool        – visibility mask
      min_radius:  radius for the very first point (oldest)
      max_radius:  radius for the current point (newest)

    Returns:
      video: [F, H, W, 3] uint8 RGB
    """
    import cv2
    num_points, num_frames = point_tracks.shape[:2]
    H, W = frames.shape[1:3]

    video = frames.copy()

    for t in range(num_frames):
        # start from the original frame
        frame = video[t].copy()

        for i in range(num_points):
            # draw every past step τ = 0..t
            for τ in range(t + 1):
                if not visibles[i, τ]:
                    continue

                if t - τ > max_retain:
                    continue

                # sub-pixel offset + clamp
                x, y = point_tracks[i, τ] + 0.5
                xi = int(np.clip(x, 0, W - 1))
                yi = int(np.clip(y, 0, H - 1))

                # age‐ratio in [0,1]
                if num_frames > 1:
                    ratio = 1 - float(t - τ) / max_retain
                else:
                    ratio = 1.0

                # interpolated radius
                radius = int(round(min_radius + (max_radius - min_radius) * ratio))

                # OpenCV draws in BGR order:
                color_rgb = age_to_bgr(ratio)

                # filled circle
                cv2.circle(frame, (xi, yi), radius, color_rgb, thickness=-1)

        video[t] = frame

    return video

def parse_json_tracks(tracks):
    tracks_data = []
    try:
        # If tracks is a string, try to parse it as JSON
        if isinstance(tracks, str):
            parsed = json.loads(tracks.replace("'", '"'))
            tracks_data.extend(parsed)
        else:
            # If tracks is a list of strings, parse each one
            for track_str in tracks:
                parsed = json.loads(track_str.replace("'", '"'))
                tracks_data.append(parsed)
        
        # Check if we have a single track (dict with x,y) or a list of tracks
        if tracks_data and isinstance(tracks_data[0], dict) and 'x' in tracks_data[0]:
            # Single track detected, wrap it in a list
            tracks_data = [tracks_data]
        elif tracks_data and isinstance(tracks_data[0], list) and tracks_data[0] and isinstance(tracks_data[0][0], dict) and 'x' in tracks_data[0][0]:
            # Already a list of tracks, nothing to do
            pass
        else:
            # Unexpected format
            print(f"Warning: Unexpected track format: {type(tracks_data[0])}")
            
    except json.JSONDecodeError as e:
        print(f"Error parsing tracks JSON: {e}")
        tracks_data = []

    return tracks_data

class WanVideoATITracks:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("WANVIDEOMODEL", ),
            "tracks": ("STRING",),
            "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 29048, "step": 8, "tooltip": "Height of the image to encode"}),
            "temperature": ("FLOAT", {"default": 220.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
            "topk": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the steps to apply ATI"}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the steps to apply ATI"}),
        },
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patchmodel"
    CATEGORY = "WanVideoWrapper"

    def patchmodel(self, model, tracks, width, height, temperature, topk, start_percent, end_percent):
        tracks_data = parse_json_tracks(tracks)
        arrs = []
        for track in tracks_data:
            pts = pad_pts(track)
            arrs.append(pts)

        tracks_np = np.stack(arrs, axis=0)

        processed_tracks = process_tracks(tracks_np, (width, height))

        patcher = model.clone()
        patcher.model_options["transformer_options"]["ati_tracks"] = processed_tracks.unsqueeze(0)
        patcher.model_options["transformer_options"]["ati_temperature"] = temperature
        patcher.model_options["transformer_options"]["ati_topk"] = topk
        patcher.model_options["transformer_options"]["ati_start_percent"] = start_percent
        patcher.model_options["transformer_options"]["ati_end_percent"] = end_percent
        
        return (patcher,)
    
class WanVideoATITracksVisualize:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "images": ("IMAGE",),
            "tracks": ("STRING",),
            "min_radius": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1, "tooltip": "radius for the very first point (oldest)"}),
            "max_radius": ("INT", {"default": 6, "min": 0, "max": 100, "step": 1, "tooltip": "radius for the current point (newest)"}),
            "max_retain": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1, "tooltip": "Maximum number of points to retain"}),
        },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "patchmodel"
    CATEGORY = "WanVideoWrapper"

    def patchmodel(self, images, tracks, min_radius, max_radius, max_retain):
        tracks_data = parse_json_tracks(tracks)
        arrs = []
        for track in tracks_data:
            pts = pad_pts(track)
            arrs.append(pts)

        tracks_np = np.stack(arrs, axis=0)
        track = np.repeat(tracks_np, 2, axis=1)[:, ::3]
        points = track[:, :, 0, :2].astype(np.float32)
        visibles = track[:, :, 0, 2].astype(np.float32)

        if images.shape[0] < points.shape[1]:
            repeat_count = (points.shape[1] + images.shape[0] - 1) // images.shape[0]
            images = images.repeat(repeat_count, 1, 1, 1)
            images = images[:points.shape[1]]
        elif images.shape[0] > points.shape[1]:
            images = images[:points.shape[1]]

        video_viz = paint_point_track(images.cpu().numpy(), points, visibles, min_radius, max_radius, max_retain)
        video_viz = torch.from_numpy(video_viz).float()
        
        return (video_viz,)

from comfy import utils
import types
from .motion_patch import patch_motion

class WanConcatCondPatch:
    def __init__(self, tracks, temperature, topk):
        self.tracks = tracks
        self.temperature = temperature
        self.topk = topk
        
    def __get__(self, obj, objtype=None):
        # Create bound method with stored parameters
        def wrapped_concat_cond(self_module, *args, **kwargs):
            return modified_concat_cond(self_module, self.tracks, self.temperature, self.topk, *args, **kwargs)
        return types.MethodType(wrapped_concat_cond, obj)
    
def modified_concat_cond(self, tracks, temperature, topk, **kwargs):
    noise = kwargs.get("noise", None)
    extra_channels = self.diffusion_model.patch_embedding.weight.shape[1] - noise.shape[1]
    if extra_channels == 0:
        return None

    image = kwargs.get("concat_latent_image", None)
    device = kwargs["device"]

    if image is None:
        shape_image = list(noise.shape)
        shape_image[1] = extra_channels
        image = torch.zeros(shape_image, dtype=noise.dtype, layout=noise.layout, device=noise.device)
    else:
        image = utils.common_upscale(image.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")
        for i in range(0, image.shape[1], 16):
            image[:, i: i + 16] = self.process_latent_in(image[:, i: i + 16])
        image = utils.resize_to_batch_size(image, noise.shape[0])

    if not self.image_to_video or extra_channels == image.shape[1]:
        return image

    if image.shape[1] > (extra_channels - 4):
        image = image[:, :(extra_channels - 4)]

    mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
    if mask is None:
        mask = torch.zeros_like(noise)[:, :4]
    else:
        if mask.shape[1] != 4:
            mask = torch.mean(mask, dim=1, keepdim=True)
        mask = 1.0 - mask
        mask = utils.common_upscale(mask.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")
        if mask.shape[-3] < noise.shape[-3]:
            mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, noise.shape[-3] - mask.shape[-3]), mode='constant', value=0)
        if mask.shape[1] == 1:
            mask = mask.repeat(1, 4, 1, 1, 1)
        mask = utils.resize_to_batch_size(mask, noise.shape[0])

    image_cond = torch.cat((mask, image), dim=1)
    image_cond_ati = patch_motion(tracks.to(image_cond.device, image_cond.dtype), image_cond[0], 
                                  temperature=temperature, topk=topk)

    return image_cond_ati.unsqueeze(0)

class WanVideoATI_comfy:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL", ),
            "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 29048, "step": 8, "tooltip": "Height of the image to encode"}),
            "tracks": ("STRING",),
            "temperature": ("FLOAT", {"default": 220.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
            "topk": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "patchcond"
    CATEGORY = "WanVideoWrapper"

    def patchcond(self, model, tracks, width, height, temperature, topk):
        
        tracks_data = parse_json_tracks(tracks)
        arrs = []
        for track in tracks_data:
            pts = pad_pts(track)
            arrs.append(pts)

        tracks_np = np.stack(arrs, axis=0)

        processed_tracks = process_tracks(tracks_np, (width, height))
    
        model_clone = model.clone()
        model_clone.add_object_patch(
            "concat_cond", 
            WanConcatCondPatch(
                processed_tracks.unsqueeze(0), temperature, topk
                ).__get__(model.model, model.model.__class__)
            )

        return (model_clone,)
        
NODE_CLASS_MAPPINGS = {
    "WanVideoATITracks": WanVideoATITracks,
    "WanVideoATITracksVisualize": WanVideoATITracksVisualize,
    "WanVideoATI_comfy": WanVideoATI_comfy,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoATITracks": "WanVideo ATI Tracks",
    "WanVideoATITracksVisualize": "WanVideo ATI Tracks Visualize",
    "WanVideoATI_comfy": "WanVideo ATI Comfy",
    }
