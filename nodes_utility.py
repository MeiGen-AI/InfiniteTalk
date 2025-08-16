import torch
import numpy as np
from comfy.utils import common_upscale

VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)

class WanVideoImageResizeToClosest:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "Image to resize"}),
            "generation_width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "generation_height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "aspect_ratio_preservation": (["keep_input", "stretch_to_new", "crop_to_new"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", )
    RETURN_NAMES = ("image","width","height",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Resizes image to the closest supported resolution based on aspect ratio and max pixels, according to the original code"

    def process(self, image, generation_width, generation_height, aspect_ratio_preservation ):
    
        H, W = image.shape[1], image.shape[2]
        max_area = generation_width * generation_height

        crop = "disabled"

        if aspect_ratio_preservation == "keep_input":
            aspect_ratio = H / W
        elif aspect_ratio_preservation == "stretch_to_new" or aspect_ratio_preservation == "crop_to_new":
            aspect_ratio = generation_height / generation_width
            if aspect_ratio_preservation == "crop_to_new":
                crop = "center"
                
        lat_h = round(
        np.sqrt(max_area * aspect_ratio) // VAE_STRIDE[1] //
        PATCH_SIZE[1] * PATCH_SIZE[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // VAE_STRIDE[2] //
            PATCH_SIZE[2] * PATCH_SIZE[2])
        h = lat_h * VAE_STRIDE[1]
        w = lat_w * VAE_STRIDE[2]

        resized_image = common_upscale(image.movedim(-1, 1), w, h, "lanczos", crop).movedim(1, -1)

        return (resized_image, w, h)

class ExtractStartFramesForContinuations:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_video_frames": ("IMAGE", {"tooltip": "Input video frames to extract the start frames from."}),
                "num_frames": ("INT", {"default": 10, "min": 1, "max": 1024, "step": 1, "tooltip": "Number of frames to get from the start of the video."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("start_frames",)
    FUNCTION = "get_start_frames"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Extracts the first N frames from a video sequence for continuations."

    def get_start_frames(self, input_video_frames, num_frames):
        if input_video_frames is None or input_video_frames.shape[0] == 0:
            log.warning("Input video frames are empty. Returning an empty tensor.")
            if input_video_frames is not None:
                return (torch.empty((0,) + input_video_frames.shape[1:], dtype=input_video_frames.dtype),)
            else:
                # Return a tensor with 4 dimensions, as expected for an IMAGE type.
                return (torch.empty((0, 64, 64, 3), dtype=torch.float32),)

        total_frames = input_video_frames.shape[0]
        num_to_get = min(num_frames, total_frames)

        if num_to_get < num_frames:
            log.warning(f"Requested {num_frames} frames, but input video only has {total_frames} frames. Returning first {num_to_get} frames.")

        start_frames = input_video_frames[:num_to_get]

        return (start_frames.cpu().float(),)

class WanVideoVACEStartToEndFrame:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "empty_frame_level": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "White level of empty frame to use"}),
            },
            "optional": {
                "start_image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "control_images": ("IMAGE",),
                "inpaint_mask": ("MASK", {"tooltip": "Inpaint mask to use for the empty frames"}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "tooltip": "Index to start from"}),
                "end_index": ("INT", {"default": -1, "min": -10000, "max": 10000, "step": 1, "tooltip": "Index to end at"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("images", "masks",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Helper node to create start/end frame batch and masks for VACE"

    def process(self, num_frames, empty_frame_level, start_image=None, end_image=None, control_images=None, inpaint_mask=None, start_index=0, end_index=-1):
        
        B, H, W, C = start_image.shape if start_image is not None else end_image.shape
        device = start_image.device if start_image is not None else end_image.device

        # Convert negative end_index to positive
        if end_index < 0:
            end_index = num_frames + end_index
        
        # Create output batch with empty frames
        out_batch = torch.ones((num_frames, H, W, 3), device=device) * empty_frame_level
        
        # Create mask tensor with proper dimensions
        masks = torch.ones((num_frames, H, W), device=device)
        
        # Pre-process all images at once to avoid redundant work
        if end_image is not None and (end_image.shape[1] != H or end_image.shape[2] != W):
            end_image = common_upscale(end_image.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(1, -1)
        
        if control_images is not None and (control_images.shape[1] != H or control_images.shape[2] != W):
            control_images = common_upscale(control_images.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(1, -1)
        
        # Place start image at start_index
        if start_image is not None:
            frames_to_copy = min(start_image.shape[0], num_frames - start_index)
            if frames_to_copy > 0:
                out_batch[start_index:start_index + frames_to_copy] = start_image[:frames_to_copy]
                masks[start_index:start_index + frames_to_copy] = 0
        
        # Place end image at end_index
        if end_image is not None:
            # Calculate where to start placing end images
            end_start = end_index - end_image.shape[0] + 1
            if end_start < 0:  # Handle case where end images won't all fit
                end_image = end_image[abs(end_start):]
                end_start = 0
                
            frames_to_copy = min(end_image.shape[0], num_frames - end_start)
            if frames_to_copy > 0:
                out_batch[end_start:end_start + frames_to_copy] = end_image[:frames_to_copy]
                masks[end_start:end_start + frames_to_copy] = 0
        
        # Apply control images to remaining frames that don't have start or end images
        if control_images is not None:
            # Create a mask of frames that are still empty (mask == 1)
            empty_frames = masks.sum(dim=(1, 2)) > 0.5 * H * W
            
            if empty_frames.any():
                # Only apply control images where they exist
                control_length = control_images.shape[0]
                for frame_idx in range(num_frames):
                    if empty_frames[frame_idx] and frame_idx < control_length:
                        out_batch[frame_idx] = control_images[frame_idx]
        
        # Apply inpaint mask if provided
        if inpaint_mask is not None:
            inpaint_mask = common_upscale(inpaint_mask.unsqueeze(1), W, H, "nearest-exact", "disabled").squeeze(1).to(device)
            
            # Handle different mask lengths efficiently
            if inpaint_mask.shape[0] > num_frames:
                inpaint_mask = inpaint_mask[:num_frames]
            elif inpaint_mask.shape[0] < num_frames:
                repeat_factor = (num_frames + inpaint_mask.shape[0] - 1) // inpaint_mask.shape[0]  # Ceiling division
                inpaint_mask = inpaint_mask.repeat(repeat_factor, 1, 1)[:num_frames]

            # Apply mask in one operation
            masks = inpaint_mask * masks

        return (out_batch.cpu().float(), masks.cpu().float())


class CreateCFGScheduleFloatList:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "steps": ("INT", {"default": 30, "min": 2, "max": 1000, "step": 1, "tooltip": "Number of steps to schedule cfg for"} ),
            "cfg_scale_start": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.01, "round": 0.01, "tooltip": "CFG scale to use for the steps"}),
            "cfg_scale_end": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.01, "round": 0.01, "tooltip": "CFG scale to use for the steps"}),
            "interpolation": (["linear", "ease_in", "ease_out"], {"default": "linear", "tooltip": "Interpolation method to use for the cfg scale"}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01,"tooltip": "Start percent of the steps to apply cfg"}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01,"tooltip": "End percent of the steps to apply cfg"}),
            }
        }

    RETURN_TYPES = ("FLOAT", )
    RETURN_NAMES = ("float_list",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Helper node to generate a list of floats that can be used to schedule cfg scale for the steps, outside the set range cfg is set to 1.0"

    def process(self, steps, cfg_scale_start, cfg_scale_end, interpolation, start_percent, end_percent):
        
        # Create a list of floats for the cfg schedule
        cfg_list = [1.0] * steps
        start_idx = min(int(steps * start_percent), steps - 1)
        end_idx = min(int(steps * end_percent), steps - 1)
        
        for i in range(start_idx, end_idx + 1):
            if i >= steps:
                break
                
            if end_idx == start_idx:
                t = 0
            else:
                t = (i - start_idx) / (end_idx - start_idx)
            
            if interpolation == "linear":
                factor = t
            elif interpolation == "ease_in":
                factor = t * t
            elif interpolation == "ease_out":
                factor = t * (2 - t)
            
            cfg_list[i] = round(cfg_scale_start + factor * (cfg_scale_end - cfg_scale_start), 2)
        
        # If start_percent > 0, always include the first step
        if start_percent > 0:
            cfg_list[0] = 1.0

        return (cfg_list,)
    
NODE_CLASS_MAPPINGS = {
    "WanVideoImageResizeToClosest": WanVideoImageResizeToClosest,
    "WanVideoVACEStartToEndFrame": WanVideoVACEStartToEndFrame,
    "ExtractStartFramesForContinuations": ExtractStartFramesForContinuations,
    "CreateCFGScheduleFloatList": CreateCFGScheduleFloatList
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoImageResizeToClosest": "WanVideo Image Resize To Closest",
    "WanVideoVACEStartToEndFrame": "WanVideo VACE Start To End Frame",
    "ExtractStartFramesForContinuations": "Extract Start Frames For Continuations",
    "CreateCFGScheduleFloatList": "Create CFG Schedule Float List"
    }