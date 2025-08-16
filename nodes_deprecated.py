import torch
import numpy as np
import os
from comfy.clip_vision import clip_preprocess, ClipVisionModel

from comfy import model_management as mm
from comfy.utils import common_upscale
from comfy.clip_vision import clip_preprocess, ClipVisionModel

script_directory = os.path.dirname(os.path.abspath(__file__))
VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)

from .utils import add_noise_to_reference_video

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()
# only kept for backwards compatibility, use WanVideoImageToVideoEncode instead
class WanVideoImageClipEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip_vision": ("CLIP_VISION",),
            "image": ("IMAGE", {"tooltip": "Image to encode"}),
            "vae": ("WANVAE",),
            "generation_width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "generation_height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            },
            "optional": {
                "force_offload": ("BOOLEAN", {"default": True}),
                "noise_aug_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Strength of noise augmentation, helpful for I2V where some noise can add motion and give sharper results"}),
                "latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional latent multiplier, helpful for I2V where lower values allow for more motion"}),
                "clip_embed_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional clip embed multiplier"}),
                "adjust_resolution": ("BOOLEAN", {"default": True, "tooltip": "Performs the same resolution adjustment as in the original code"}),

            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DEPRECATED = True

    def process(self, clip_vision, vae, image, num_frames, generation_width, generation_height, force_offload=True, noise_aug_strength=0.0, 
                latent_strength=1.0, clip_embed_strength=1.0, adjust_resolution=True):

        self.image_mean = [0.48145466, 0.4578275, 0.40821073]
        self.image_std = [0.26862954, 0.26130258, 0.27577711]
    
        H, W = image.shape[1], image.shape[2]
        max_area = generation_width * generation_height

        
        print(clip_vision)
        clip_vision.model.to(device)
        if isinstance(clip_vision, ClipVisionModel):
            clip_context = clip_vision.encode_image(image).last_hidden_state.to(device)
        else:
            pixel_values = clip_preprocess(image.to(device), size=224, mean=self.image_mean, std=self.image_std, crop=True).float()
            clip_context = clip_vision.visual(pixel_values)
        if clip_embed_strength != 1.0:
            clip_context *= clip_embed_strength
        
        if force_offload:
            clip_vision.model.to(offload_device)
            mm.soft_empty_cache()

        if adjust_resolution:
            aspect_ratio = H / W
            lat_h = round(
            np.sqrt(max_area * aspect_ratio) // VAE_STRIDE[1] //
            PATCH_SIZE[1] * PATCH_SIZE[1])
            lat_w = round(
                np.sqrt(max_area / aspect_ratio) // VAE_STRIDE[2] //
                PATCH_SIZE[2] * PATCH_SIZE[2])
            h = lat_h * VAE_STRIDE[1]
            w = lat_w * VAE_STRIDE[2]
        else:
            h = generation_height
            w = generation_width
            lat_h = h // 8
            lat_w = w // 8

        # Step 1: Create initial mask with ones for first frame, zeros for others
        mask = torch.ones(1, num_frames, lat_h, lat_w, device=device)
        mask[:, 1:] = 0

        # Step 2: Repeat first frame 4 times and concatenate with remaining frames
        first_frame_repeated = torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1)
        mask = torch.concat([first_frame_repeated, mask[:, 1:]], dim=1)

        # Step 3: Reshape mask into groups of 4 frames
        mask = mask.view(1, mask.shape[1] // 4, 4, lat_h, lat_w)

        # Step 4: Transpose dimensions and select first batch
        mask = mask.transpose(1, 2)[0]

        # Calculate maximum sequence length
        frames_per_stride = (num_frames - 1) // VAE_STRIDE[0] + 1
        patches_per_frame = lat_h * lat_w // (PATCH_SIZE[1] * PATCH_SIZE[2])
        max_seq_len = frames_per_stride * patches_per_frame

        vae.to(device)

        # Step 1: Resize and rearrange the input image dimensions
        #resized_image = image.permute(0, 3, 1, 2)  # Rearrange dimensions to (B, C, H, W)
        #resized_image = torch.nn.functional.interpolate(resized_image, size=(h, w), mode='bicubic')
        resized_image = common_upscale(image.movedim(-1, 1), w, h, "lanczos", "disabled")
        resized_image = resized_image.transpose(0, 1)  # Transpose to match required format
        resized_image = resized_image * 2 - 1

        if noise_aug_strength > 0.0:
            resized_image = add_noise_to_reference_video(resized_image, ratio=noise_aug_strength)
        
        # Step 2: Create zero padding frames
        zero_frames = torch.zeros(3, num_frames-1, h, w, device=device)

        # Step 3: Concatenate image with zero frames
        concatenated = torch.concat([resized_image.to(device), zero_frames, resized_image.to(device)], dim=1).to(device = device, dtype = vae.dtype)
        concatenated *= latent_strength
        y = vae.encode([concatenated], device)[0]

        y = torch.concat([mask, y])

        vae.model.clear_cache()
        vae.to(offload_device)

        image_embeds = {
            "image_embeds": y,
            "clip_context": clip_context,
            "max_seq_len": max_seq_len,
            "num_frames": num_frames,
            "lat_h": lat_h,
            "lat_w": lat_w,
        }

        return (image_embeds,)
    
NODE_CLASS_MAPPINGS = {
    "WanVideoImageClipEncode": WanVideoImageClipEncode,#deprecated
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoImageClipEncode": "WanVideo ImageClip Encode (Deprecated)",
    }