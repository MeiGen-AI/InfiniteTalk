import os, gc, math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import inspect

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from .wanvideo.modules.model import rope_params

from .wanvideo.schedulers import get_scheduler, get_sampling_sigmas, retrieve_timesteps, scheduler_list

from .multitalk.multitalk import timestep_transform, add_noise
from .utils import log, print_memory, apply_lora, clip_encode_image_tiled, fourier_filter, is_image_black, add_noise_to_reference_video, optimized_scale, find_closest_valid_dim
from .cache_methods.cache_methods import cache_report
from .enhance_a_video.globals import set_enhance_weight, set_num_frames
from .taehv import TAEHV

from einops import rearrange

from comfy import model_management as mm
from comfy.utils import ProgressBar, common_upscale
from comfy.clip_vision import clip_preprocess, ClipVisionModel
from comfy.cli_args import args, LatentPreviewMethod

script_directory = os.path.dirname(os.path.abspath(__file__))

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)

class WanVideoEnhanceAVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "weight": ("FLOAT", {"default": 2.0, "min": 0, "max": 100, "step": 0.01, "tooltip": "The feta Weight of the Enhance-A-Video"}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percentage of the steps to apply Enhance-A-Video"}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percentage of the steps to apply Enhance-A-Video"}),
            },
        }
    RETURN_TYPES = ("FETAARGS",)
    RETURN_NAMES = ("feta_args",)
    FUNCTION = "setargs"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "https://github.com/NUS-HPC-AI-Lab/Enhance-A-Video"

    def setargs(self, **kwargs):
        return (kwargs, )

class WanVideoSetBlockSwap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL", ),
                "block_swap_args": ("BLOCKSWAPARGS", ),
               }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"

    def loadmodel(self, model, block_swap_args):

        patcher = model.clone()
        if 'transformer_options' not in patcher.model_options:
            patcher.model_options['transformer_options'] = {}
        patcher.model_options["transformer_options"]["block_swap_args"] = block_swap_args     

        return (patcher,)

class WanVideoSetRadialAttention:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL", ),
                "dense_attention_mode": ([
                    "sdpa",
                    "flash_attn_2",
                    "flash_attn_3",
                    "sageattn",
                    "sparse_sage_attention",
                    ], {"default": "sageattn", "tooltip": "The attention mode for dense attention"}),
                "dense_blocks": ("INT",  {"default": 1, "min": 0, "max": 40, "step": 1, "tooltip": "Number of blocks to apply normal attention to"}),
                "dense_vace_blocks": ("INT",  {"default": 1, "min": 0, "max": 15, "step": 1, "tooltip": "Number of vace blocks to apply normal attention to"}),
                "dense_timesteps": ("INT",  {"default": 2, "min": 0, "max": 100, "step": 1, "tooltip": "The step to start applying sparse attention"}),
                "decay_factor": ("FLOAT",  {"default": 0.2, "min": 0, "max": 1, "step": 0.01, "tooltip": "Controls how quickly the attention window shrinks as the distance between frames increases in the sparse attention mask."}),
                "block_size":([128, 64], {"default": 128, "tooltip": "Radial attention block size, larger blocks are faster but restricts usable dimensions more."}),
               }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Sets radial attention parameters, dense attention refers to normal attention"

    def loadmodel(self, model, dense_attention_mode, dense_blocks, dense_vace_blocks, dense_timesteps, decay_factor, block_size):
        if "radial" not in model.model.diffusion_model.attention_mode:
            raise Exception("Enable radial attention first in the model loader.")
            
        patcher = model.clone()
        if 'transformer_options' not in patcher.model_options:
            patcher.model_options['transformer_options'] = {}

        patcher.model_options["transformer_options"]["dense_attention_mode"] = dense_attention_mode
        patcher.model_options["transformer_options"]["dense_blocks"] = dense_blocks
        patcher.model_options["transformer_options"]["dense_vace_blocks"] = dense_vace_blocks
        patcher.model_options["transformer_options"]["dense_timesteps"] = dense_timesteps
        patcher.model_options["transformer_options"]["decay_factor"] = decay_factor
        patcher.model_options["transformer_options"]["block_size"] = block_size

        return (patcher,)

class WanVideoBlockList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "blocks": ("STRING",  {"default": "1", "multiline":True}),
               }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("block_list", )
    FUNCTION = "create_list"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Comma separated list of blocks to apply block swap to, can also use ranges like '0-5' or '0,2,3-5' etc., can be connected to the dense_blocks input of 'WanVideoSetRadialAttention' node"

    def create_list(self, blocks):
        block_list = []
        for line in blocks.splitlines():
            for part in line.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    try:
                        start, end = map(int, part.split("-", 1))
                        block_list.extend(range(start, end + 1))
                    except Exception:
                        raise ValueError(f"Invalid range: '{part}'")
                else:
                    try:
                        block_list.append(int(part))
                    except Exception:
                        raise ValueError(f"Invalid integer: '{part}'")
        return (block_list,)

#region TextEncode
class WanVideoTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "t5": ("WANTEXTENCODER",),
            "positive_prompt": ("STRING", {"default": "", "multiline": True} ),
            "negative_prompt": ("STRING", {"default": "", "multiline": True} ),
            },
            "optional": {
                "force_offload": ("BOOLEAN", {"default": True}),
                "model_to_offload": ("WANVIDEOMODEL", {"tooltip": "Model to move to offload_device before encoding"}),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", )
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Encodes text prompts into text embeddings. For rudimentary prompt travel you can input multiple prompts separated by '|', they will be equally spread over the video length"

    def process(self, t5, positive_prompt, negative_prompt,force_offload=True, model_to_offload=None):
        if model_to_offload is not None:
            log.info(f"Moving video model to {offload_device}")
            model_to_offload.model.to(offload_device)
            mm.soft_empty_cache()

        encoder = t5["model"]
        dtype = t5["dtype"]

        # Split positive prompts and process each with weights
        positive_prompts_raw = [p.strip() for p in positive_prompt.split('|')]
        positive_prompts = []
        all_weights = []
        
        for p in positive_prompts_raw:
            cleaned_prompt, weights = self.parse_prompt_weights(p)
            positive_prompts.append(cleaned_prompt)
            all_weights.append(weights)
        
        encoder.model.to(device)
       
        with torch.autocast(device_type=mm.get_autocast_device(device), dtype=dtype, enabled=True):
            context = encoder(positive_prompts, device)
            context_null = encoder([negative_prompt], device)

            # Apply weights to embeddings if any were extracted
            for i, weights in enumerate(all_weights):
                for text, weight in weights.items():
                    log.info(f"Applying weight {weight} to prompt: {text}")
                    if len(weights) > 0:
                        context[i] = context[i] * weight

        if force_offload:
            encoder.model.to(offload_device)
            mm.soft_empty_cache()

        prompt_embeds_dict = {
                "prompt_embeds": context,
                "negative_prompt_embeds": context_null,
            }
        return (prompt_embeds_dict,)
    
    def parse_prompt_weights(self, prompt):
        """Extract text and weights from prompts with (text:weight) format"""
        import re
        
        # Parse all instances of (text:weight) in the prompt
        pattern = r'\((.*?):([\d\.]+)\)'
        matches = re.findall(pattern, prompt)
        
        # Replace each match with just the text part
        cleaned_prompt = prompt
        weights = {}
        
        for match in matches:
            text, weight = match
            orig_text = f"({text}:{weight})"
            cleaned_prompt = cleaned_prompt.replace(orig_text, text)
            weights[text] = float(weight)
            
        return cleaned_prompt, weights
    
class WanVideoTextEncodeSingle:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "t5": ("WANTEXTENCODER",),
            "prompt": ("STRING", {"default": "", "multiline": True} ),
            },
            "optional": {
                "force_offload": ("BOOLEAN", {"default": True}),
                "model_to_offload": ("WANVIDEOMODEL", {"tooltip": "Model to move to offload_device before encoding"}),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", )
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Encodes text prompt into text embedding."

    def process(self, t5, prompt, force_offload=True, model_to_offload=None):
        if model_to_offload is not None:
            log.info(f"Moving video model to {offload_device}")
            model_to_offload.model.to(offload_device)
            mm.soft_empty_cache()

        encoder = t5["model"]
        dtype = t5["dtype"]

        encoder.model.to(device)
       
        with torch.autocast(device_type=mm.get_autocast_device(device), dtype=dtype, enabled=True):
            encoded = encoder([prompt], device)

        if force_offload:
            encoder.model.to(offload_device)
            mm.soft_empty_cache()

        prompt_embeds_dict = {
                "prompt_embeds": encoded,
                "negative_prompt_embeds": None,
            }
        return (prompt_embeds_dict,)
    
class WanVideoApplyNAG:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "original_text_embeds": ("WANVIDEOTEXTEMBEDS",),
            "nag_text_embeds": ("WANVIDEOTEXTEMBEDS",),
            "nag_scale": ("FLOAT", {"default": 11.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            "nag_tau": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1}),
            "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", )
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Adds NAG prompt embeds to original prompt embeds: 'https://github.com/ChenDarYen/Normalized-Attention-Guidance'"

    def process(self, original_text_embeds, nag_text_embeds, nag_scale, nag_tau, nag_alpha):
        prompt_embeds_dict_copy = original_text_embeds.copy()
        prompt_embeds_dict_copy.update({
                "nag_prompt_embeds": nag_text_embeds["prompt_embeds"],
                "nag_params": {
                    "nag_scale": nag_scale,
                    "nag_tau": nag_tau,
                    "nag_alpha": nag_alpha,
                }
            })
        return (prompt_embeds_dict_copy,)
    
class WanVideoTextEmbedBridge:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "positive": ("CONDITIONING",),
            },
            "optional": {
                "negative": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", )
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Bridge between ComfyUI native text embedding and WanVideoWrapper text embedding"

    def process(self, positive, negative=None):
        prompt_embeds_dict = {
                "prompt_embeds": positive[0][0].to(device),
                "negative_prompt_embeds": negative[0][0].to(device) if negative is not None else None,
            }
        return (prompt_embeds_dict,)
    
#region clip vision
class WanVideoClipVisionEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip_vision": ("CLIP_VISION",),
            "image_1": ("IMAGE", {"tooltip": "Image to encode"}),
            "strength_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional clip embed multiplier"}), 
            "strength_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional clip embed multiplier"}),
            "crop": (["center", "disabled"], {"default": "center", "tooltip": "Crop image to 224x224 before encoding"}),
            "combine_embeds": (["average", "sum", "concat", "batch"], {"default": "average", "tooltip": "Method to combine multiple clip embeds"}),
            "force_offload": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image_2": ("IMAGE", ),
                "negative_image": ("IMAGE", {"tooltip": "image to use for uncond"}),
                "tiles": ("INT", {"default": 0, "min": 0, "max": 16, "step": 2, "tooltip": "Use matteo's tiled image encoding for improved accuracy"}),
                "ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Ratio of the tile average"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_CLIPEMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, clip_vision, image_1, strength_1, strength_2, force_offload, crop, combine_embeds, image_2=None, negative_image=None, tiles=0, ratio=1.0):
        image_mean = [0.48145466, 0.4578275, 0.40821073]
        image_std = [0.26862954, 0.26130258, 0.27577711]

        if image_2 is not None:
            image = torch.cat([image_1, image_2], dim=0)
        else:
            image = image_1

        clip_vision.model.to(device)
        
        negative_clip_embeds = None

        if tiles > 0:
            log.info("Using tiled image encoding")
            clip_embeds = clip_encode_image_tiled(clip_vision, image.to(device), tiles=tiles, ratio=ratio)
            if negative_image is not None:
                negative_clip_embeds = clip_encode_image_tiled(clip_vision, negative_image.to(device), tiles=tiles, ratio=ratio)
        else:
            if isinstance(clip_vision, ClipVisionModel):
                clip_embeds = clip_vision.encode_image(image).penultimate_hidden_states.to(device)
                if negative_image is not None:
                    negative_clip_embeds = clip_vision.encode_image(negative_image).penultimate_hidden_states.to(device)
            else:
                pixel_values = clip_preprocess(image.to(device), size=224, mean=image_mean, std=image_std, crop=(not crop == "disabled")).float()
                clip_embeds = clip_vision.visual(pixel_values)
                if negative_image is not None:
                    pixel_values = clip_preprocess(negative_image.to(device), size=224, mean=image_mean, std=image_std, crop=(not crop == "disabled")).float()
                    negative_clip_embeds = clip_vision.visual(pixel_values)
    
        log.info(f"Clip embeds shape: {clip_embeds.shape}, dtype: {clip_embeds.dtype}")

        weighted_embeds = []
        weighted_embeds.append(clip_embeds[0:1] * strength_1)

        # Handle all additional embeddings
        if clip_embeds.shape[0] > 1:
            weighted_embeds.append(clip_embeds[1:2] * strength_2)
            
            if clip_embeds.shape[0] > 2:
                for i in range(2, clip_embeds.shape[0]):
                    weighted_embeds.append(clip_embeds[i:i+1])  # Add as-is without strength modifier
            
            # Combine all weighted embeddings
            if combine_embeds == "average":
                clip_embeds = torch.mean(torch.stack(weighted_embeds), dim=0)
            elif combine_embeds == "sum":
                clip_embeds = torch.sum(torch.stack(weighted_embeds), dim=0)
            elif combine_embeds == "concat":
                clip_embeds = torch.cat(weighted_embeds, dim=1)
            elif combine_embeds == "batch":
                clip_embeds = torch.cat(weighted_embeds, dim=0)
        else:
            clip_embeds = weighted_embeds[0]
                

        log.info(f"Combined clip embeds shape: {clip_embeds.shape}")
        
        if force_offload:
            clip_vision.model.to(offload_device)
            mm.soft_empty_cache()

        clip_embeds_dict = {
            "clip_embeds": clip_embeds,
            "negative_clip_embeds": negative_clip_embeds
        }

        return (clip_embeds_dict,)
        
class WanVideoRealisDanceLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ref_latent": ("LATENT", {"tooltip": "Reference image to encode"}),
            "pose_cond_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the SMPL model"}),
            "pose_cond_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the SMPL model"}),
            },
            "optional": {
                "smpl_latent": ("LATENT", {"tooltip": "SMPL pose image to encode"}),
                "hamer_latent": ("LATENT", {"tooltip": "Hamer hand pose image to encode"}),
            },
        }

    RETURN_TYPES = ("ADD_COND_LATENTS",)
    RETURN_NAMES = ("add_cond_latents",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, ref_latent, pose_cond_start_percent, pose_cond_end_percent, hamer_latent=None, smpl_latent=None):
        if smpl_latent is None and hamer_latent is None:
            raise Exception("At least one of smpl_latent or hamer_latent must be provided")
        if smpl_latent is None:
            smpl = torch.zeros_like(hamer_latent["samples"])
        else:
            smpl = smpl_latent["samples"]
        if hamer_latent is None:
            hamer = torch.zeros_like(smpl_latent["samples"])
        else:
            hamer = hamer_latent["samples"]

        pose_latent = torch.cat((smpl, hamer), dim=1)
        
        add_cond_latents = {
            "ref_latent": ref_latent["samples"],
            "pose_latent": pose_latent,
            "pose_cond_start_percent": pose_cond_start_percent,
            "pose_cond_end_percent": pose_cond_end_percent,
        }

        return (add_cond_latents,)

class WanVideoImageToVideoEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae": ("WANVAE",),
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "noise_aug_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Strength of noise augmentation, helpful for I2V where some noise can add motion and give sharper results"}),
            "start_latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional latent multiplier, helpful for I2V where lower values allow for more motion"}),
            "end_latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional latent multiplier, helpful for I2V where lower values allow for more motion"}),
            "force_offload": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "clip_embeds": ("WANVIDIMAGE_CLIPEMBEDS", {"tooltip": "Clip vision encoded image"}),
                "start_image": ("IMAGE", {"tooltip": "Image to encode"}),
                "end_image": ("IMAGE", {"tooltip": "end frame"}),
                "control_embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "Control signal for the Fun -model"}),
                "fun_or_fl2v_model": ("BOOLEAN", {"default": True, "tooltip": "Enable when using official FLF2V or Fun model"}),
                "temporal_mask": ("MASK", {"tooltip": "mask"}),
                "extra_latents": ("LATENT", {"tooltip": "Extra latents to add to the input front, used for Skyreels A2 reference images"}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
                "add_cond_latents": ("ADD_COND_LATENTS", {"advanced": True, "tooltip": "Additional cond latents WIP"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, vae, width, height, num_frames, force_offload, noise_aug_strength, 
                start_latent_strength, end_latent_strength, start_image=None, end_image=None, control_embeds=None, fun_or_fl2v_model=False, 
                temporal_mask=None, extra_latents=None, clip_embeds=None, tiled_vae=False, add_cond_latents=None):

        H = height
        W = width
           
        lat_h = H // 8
        lat_w = W // 8
        
        num_frames = ((num_frames - 1) // 4) * 4 + 1
        two_ref_images = start_image is not None and end_image is not None

        base_frames = num_frames + (1 if two_ref_images and not fun_or_fl2v_model else 0)
        if temporal_mask is None:
            mask = torch.zeros(1, base_frames, lat_h, lat_w, device=device)
            if start_image is not None:
                mask[:, 0:start_image.shape[0]] = 1  # First frame
            if end_image is not None:
                mask[:, -end_image.shape[0]:] = 1  # End frame if exists
        else:
            mask = common_upscale(temporal_mask.unsqueeze(1).to(device), lat_w, lat_h, "nearest", "disabled").squeeze(1)
            if mask.shape[0] > base_frames:
                mask = mask[:base_frames]
            elif mask.shape[0] < base_frames:
                mask = torch.cat([mask, torch.zeros(base_frames - mask.shape[0], lat_h, lat_w, device=device)])
            mask = mask.unsqueeze(0).to(device)

        # Repeat first frame and optionally end frame
        start_mask_repeated = torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1) # T, C, H, W
        if end_image is not None and not fun_or_fl2v_model:
            end_mask_repeated = torch.repeat_interleave(mask[:, -1:], repeats=4, dim=1) # T, C, H, W
            mask = torch.cat([start_mask_repeated, mask[:, 1:-1], end_mask_repeated], dim=1)
        else:
            mask = torch.cat([start_mask_repeated, mask[:, 1:]], dim=1)

        # Reshape mask into groups of 4 frames
        mask = mask.view(1, mask.shape[1] // 4, 4, lat_h, lat_w) # 1, T, C, H, W
        mask = mask.movedim(1, 2)[0]# C, T, H, W

        # Resize and rearrange the input image dimensions
        if start_image is not None:
            resized_start_image = common_upscale(start_image.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            resized_start_image = resized_start_image * 2 - 1
            if noise_aug_strength > 0.0:
                resized_start_image = add_noise_to_reference_video(resized_start_image, ratio=noise_aug_strength)
        
        if end_image is not None:
            resized_end_image = common_upscale(end_image.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            resized_end_image = resized_end_image * 2 - 1
            if noise_aug_strength > 0.0:
                resized_end_image = add_noise_to_reference_video(resized_end_image, ratio=noise_aug_strength)
            
        # Concatenate image with zero frames and encode
        vae.to(device)

        if temporal_mask is None:
            if start_image is not None and end_image is None:
                zero_frames = torch.zeros(3, num_frames-start_image.shape[0], H, W, device=device)
                concatenated = torch.cat([resized_start_image.to(device), zero_frames], dim=1)
            elif start_image is None and end_image is not None:
                zero_frames = torch.zeros(3, num_frames-end_image.shape[0], H, W, device=device)
                concatenated = torch.cat([zero_frames, resized_end_image.to(device)], dim=1)
            elif start_image is None and end_image is None:
                concatenated = torch.zeros(3, num_frames, H, W, device=device)
            else:
                if fun_or_fl2v_model:
                    zero_frames = torch.zeros(3, num_frames-(start_image.shape[0]+end_image.shape[0]), H, W, device=device)
                else:
                    zero_frames = torch.zeros(3, num_frames-1, H, W, device=device)
                concatenated = torch.cat([resized_start_image.to(device), zero_frames, resized_end_image.to(device)], dim=1)
        else:
            temporal_mask = common_upscale(temporal_mask.unsqueeze(1), W, H, "nearest", "disabled").squeeze(1)
            concatenated = resized_start_image[:,:num_frames] * temporal_mask[:num_frames].unsqueeze(0)

        y = vae.encode([concatenated.to(device=device, dtype=vae.dtype)], device, end_=(end_image is not None and not fun_or_fl2v_model),tiled=tiled_vae)[0]
        has_ref = False
        if extra_latents is not None:
            samples = extra_latents["samples"].squeeze(0)
            y = torch.cat([samples, y], dim=1)
            mask = torch.cat([torch.ones_like(mask[:, 0:samples.shape[1]]), mask], dim=1)
            num_frames += samples.shape[1] * 4
            has_ref = True
        y[:, :1] *= start_latent_strength
        y[:, -1:] *= end_latent_strength
        if control_embeds is None:
            y = torch.cat([mask, y])
        else:
            if end_image is None:
                y[:, 1:] = 0
            elif start_image is None:
                y[:, -1:] = 0
            else:
                y[:, 1:-1] = 0 # doesn't seem to work anyway though...

        # Calculate maximum sequence length
        patches_per_frame = lat_h * lat_w // (PATCH_SIZE[1] * PATCH_SIZE[2])
        frames_per_stride = (num_frames - 1) // 4 + (2 if end_image is not None and not fun_or_fl2v_model else 1)
        max_seq_len = frames_per_stride * patches_per_frame

        if add_cond_latents is not None:
            add_cond_latents["ref_latent_neg"] = vae.encode(torch.zeros(1, 3, 1, H, W, device=device, dtype=vae.dtype), device)

        vae.model.clear_cache()
        if force_offload:
            vae.model.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()

        image_embeds = {
            "image_embeds": y,
            "clip_context": clip_embeds.get("clip_embeds", None) if clip_embeds is not None else None,
            "negative_clip_context": clip_embeds.get("negative_clip_embeds", None) if clip_embeds is not None else None,
            "max_seq_len": max_seq_len,
            "num_frames": num_frames,
            "lat_h": lat_h,
            "lat_w": lat_w,
            "control_embeds": control_embeds["control_embeds"] if control_embeds is not None else None,
            "end_image": resized_end_image if end_image is not None else None,
            "fun_or_fl2v_model": fun_or_fl2v_model,
            "has_ref": has_ref,
            "add_cond_latents": add_cond_latents
        }

        return (image_embeds,)
    
class WanVideoEmptyEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            },
            "optional": {
                "control_embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "control signal for the Fun -model"}),
                "extra_latents": ("LATENT", {"tooltip": "First latent to use for the Pusa -model"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, num_frames, width, height, control_embeds=None, extra_latents=None):
        target_shape = (16, (num_frames - 1) // VAE_STRIDE[0] + 1,
                        height // VAE_STRIDE[1],
                        width // VAE_STRIDE[2])
        
        embeds = {
            "target_shape": target_shape,
            "num_frames": num_frames,
            "control_embeds": control_embeds["control_embeds"] if control_embeds is not None else None,
            "extra_latents": extra_latents
        }
    
        return (embeds,)

class WanVideoMiniMaxRemoverEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "latents": ("LATENT", {"tooltip": "Encoded latents to use as control signals"}),
            "mask_latents": ("LATENT", {"tooltip": "Encoded latents to use as mask"}),
            },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, num_frames, width, height, latents, mask_latents):
        target_shape = (16, (num_frames - 1) // VAE_STRIDE[0] + 1,
                        height // VAE_STRIDE[1],
                        width // VAE_STRIDE[2])
        
        embeds = {
            "target_shape": target_shape,
            "num_frames": num_frames,
            "minimax_latents": latents["samples"].squeeze(0),
            "minimax_mask_latents": mask_latents["samples"].squeeze(0),
        }
    
        return (embeds,)
    
# region phantom
class WanVideoPhantomEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "phantom_latent_1": ("LATENT", {"tooltip": "reference latents for the phantom model"}),
            
            "phantom_cfg_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "CFG scale for the extra phantom cond pass"}),
            "phantom_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the phantom model"}),
            "phantom_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the phantom model"}),
            },
            "optional": {
                "phantom_latent_2": ("LATENT", {"tooltip": "reference latents for the phantom model"}),
                "phantom_latent_3": ("LATENT", {"tooltip": "reference latents for the phantom model"}),
                "phantom_latent_4": ("LATENT", {"tooltip": "reference latents for the phantom model"}),
                "vace_embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "VACE embeds"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, num_frames, phantom_cfg_scale, phantom_start_percent, phantom_end_percent, phantom_latent_1, phantom_latent_2=None, phantom_latent_3=None, phantom_latent_4=None, vace_embeds=None):
        samples = phantom_latent_1["samples"].squeeze(0)
        if phantom_latent_2 is not None:
            samples = torch.cat([samples, phantom_latent_2["samples"].squeeze(0)], dim=1)
        if phantom_latent_3 is not None:
            samples = torch.cat([samples, phantom_latent_3["samples"].squeeze(0)], dim=1)
        if phantom_latent_4 is not None:
            samples = torch.cat([samples, phantom_latent_4["samples"].squeeze(0)], dim=1)
        C, T, H, W = samples.shape

        log.info(f"Phantom latents shape: {samples.shape}")

        target_shape = (16, (num_frames - 1) // VAE_STRIDE[0] + 1 + T,
                        H * 8 // VAE_STRIDE[1],
                        W * 8 // VAE_STRIDE[2])
        
        embeds = {
            "target_shape": target_shape,
            "num_frames": num_frames,
            "phantom_latents": samples,
            "phantom_cfg_scale": phantom_cfg_scale,
            "phantom_start_percent": phantom_start_percent,
            "phantom_end_percent": phantom_end_percent,
        }
        if vace_embeds is not None:
            vace_input = {
                "vace_context": vace_embeds["vace_context"],
                "vace_scale": vace_embeds["vace_scale"],
                "has_ref": vace_embeds["has_ref"],
                "vace_start_percent": vace_embeds["vace_start_percent"],
                "vace_end_percent": vace_embeds["vace_end_percent"],
                "vace_seq_len": vace_embeds["vace_seq_len"],
                "additional_vace_inputs": vace_embeds["additional_vace_inputs"],
                }
            embeds.update(vace_input)
    
        return (embeds,)
    
class WanVideoControlEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "latents": ("LATENT", {"tooltip": "Encoded latents to use as control signals"}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the control signal"}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the control signal"}),
            },
            "optional": {
                "fun_ref_image": ("LATENT", {"tooltip": "Reference latent for the Fun 1.1 -model"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, latents, start_percent, end_percent, fun_ref_image=None):

        samples = latents["samples"].squeeze(0)
        C, T, H, W = samples.shape

        num_frames = (T - 1) * 4 + 1
        seq_len = math.ceil((H * W) / 4 * ((num_frames - 1) // 4 + 1))
      
        embeds = {
            "max_seq_len": seq_len,
            "target_shape": samples.shape,
            "num_frames": num_frames,
            "control_embeds": {
                "control_images": samples,
                "start_percent": start_percent,
                "end_percent": end_percent,
                "fun_ref_image": fun_ref_image["samples"][:,:, 0] if fun_ref_image is not None else None,
            }
        }
    
        return (embeds,)
    
class WanVideoSLG:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "blocks": ("STRING", {"default": "10", "tooltip": "Blocks to skip uncond on, separated by comma, index starts from 0"}),
            "start_percent": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the control signal"}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the control signal"}),
            },
        }

    RETURN_TYPES = ("SLGARGS", )
    RETURN_NAMES = ("slg_args",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Skips uncond on the selected blocks"

    def process(self, blocks, start_percent, end_percent):
        slg_block_list = [int(x.strip()) for x in blocks.split(",")]

        slg_args = {
            "blocks": slg_block_list,
            "start_percent": start_percent,
            "end_percent": end_percent,
        }
        return (slg_args,)

#region VACE
class WanVideoVACEEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae": ("WANVAE",),
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
            "vace_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the steps to apply VACE"}),
            "vace_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the steps to apply VACE"}),
            },
            "optional": {
                "input_frames": ("IMAGE",),
                "ref_images": ("IMAGE",),
                "input_masks": ("MASK",),
                "prev_vace_embeds": ("WANVIDIMAGE_EMBEDS",),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
            },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("vace_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, vae, width, height, num_frames, strength, vace_start_percent, vace_end_percent, input_frames=None, ref_images=None, input_masks=None, prev_vace_embeds=None, tiled_vae=False):
        vae = vae.to(device)

        width = (width // 16) * 16
        height = (height // 16) * 16

        target_shape = (16, (num_frames - 1) // VAE_STRIDE[0] + 1,
                        height // VAE_STRIDE[1],
                        width // VAE_STRIDE[2])
        # vace context encode
        if input_frames is None:
            input_frames = torch.zeros((1, 3, num_frames, height, width), device=device, dtype=vae.dtype)
        else:
            input_frames = input_frames[:num_frames]
            input_frames = common_upscale(input_frames.clone().movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)
            input_frames = input_frames.to(vae.dtype).to(device).unsqueeze(0).permute(0, 4, 1, 2, 3) # B, C, T, H, W
            input_frames = input_frames * 2 - 1
        if input_masks is None:
            input_masks = torch.ones_like(input_frames, device=device)
        else:
            print("input_masks shape", input_masks.shape)
            input_masks = input_masks[:num_frames]
            input_masks = common_upscale(input_masks.clone().unsqueeze(1), width, height, "nearest-exact", "disabled").squeeze(1)
            input_masks = input_masks.to(vae.dtype).to(device)
            input_masks = input_masks.unsqueeze(-1).unsqueeze(0).permute(0, 4, 1, 2, 3).repeat(1, 3, 1, 1, 1) # B, C, T, H, W

        if ref_images is not None:
            # Create padded image
            if ref_images.shape[0] > 1:
                ref_images = torch.cat([ref_images[i] for i in range(ref_images.shape[0])], dim=1).unsqueeze(0)
        
            B, H, W, C = ref_images.shape
            current_aspect = W / H
            target_aspect = width / height
            if current_aspect > target_aspect:
                # Image is wider than target, pad height
                new_h = int(W / target_aspect)
                pad_h = (new_h - H) // 2
                padded = torch.ones(ref_images.shape[0], new_h, W, ref_images.shape[3], device=ref_images.device, dtype=ref_images.dtype)
                padded[:, pad_h:pad_h+H, :, :] = ref_images
                ref_images = padded
            elif current_aspect < target_aspect:
                # Image is taller than target, pad width
                new_w = int(H * target_aspect)
                pad_w = (new_w - W) // 2
                padded = torch.ones(ref_images.shape[0], H, new_w, ref_images.shape[3], device=ref_images.device, dtype=ref_images.dtype)
                padded[:, :, pad_w:pad_w+W, :] = ref_images
                ref_images = padded
            ref_images = common_upscale(ref_images.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)
            
            ref_images = ref_images.to(vae.dtype).to(device).unsqueeze(0).permute(0, 4, 1, 2, 3).unsqueeze(0)
            ref_images = ref_images * 2 - 1
      
        z0 = self.vace_encode_frames(vae, input_frames, ref_images, masks=input_masks, tiled_vae=tiled_vae)
        vae.model.clear_cache()
        m0 = self.vace_encode_masks(input_masks, ref_images)
        z = self.vace_latent(z0, m0)

        vae.to(offload_device)

        vace_input = {
            "vace_context": z,
            "vace_scale": strength,
            "has_ref": ref_images is not None,
            "num_frames": num_frames,
            "target_shape": target_shape,
            "vace_start_percent": vace_start_percent,
            "vace_end_percent": vace_end_percent,
            "vace_seq_len": math.ceil((z[0].shape[2] * z[0].shape[3]) / 4 * z[0].shape[1]),
            "additional_vace_inputs": [],
        }

        if prev_vace_embeds is not None:
            if "additional_vace_inputs" in prev_vace_embeds and prev_vace_embeds["additional_vace_inputs"]:
                vace_input["additional_vace_inputs"] = prev_vace_embeds["additional_vace_inputs"].copy()
            vace_input["additional_vace_inputs"].append(prev_vace_embeds)
    
        return (vace_input,)
    def vace_encode_frames(self, vae, frames, ref_images, masks=None, tiled_vae=False):
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)

        if masks is None:
            latents = vae.encode(frames, device=device, tiled=tiled_vae)
        else:
            inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
            reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
            inactive = vae.encode(inactive, device=device, tiled=tiled_vae)
            reactive = vae.encode(reactive, device=device, tiled=tiled_vae)
            latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]
        vae.model.clear_cache()
        cat_latents = []
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                if masks is None:
                    ref_latent = vae.encode(refs, device=device, tiled=tiled_vae)
                else:
                    print("refs shape", refs.shape)#torch.Size([3, 1, 512, 512])
                    ref_latent = vae.encode(refs, device=device, tiled=tiled_vae)
                    ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]
                assert all([x.shape[1] == 1 for x in ref_latent])
                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
        return cat_latents

    def vace_encode_masks(self, masks, ref_images=None):
        if ref_images is None:
            ref_images = [None] * len(masks)
        else:
            assert len(masks) == len(ref_images)

        result_masks = []
        for mask, refs in zip(masks, ref_images):
            c, depth, height, width = mask.shape
            new_depth = int((depth + 3) // VAE_STRIDE[0])
            height = 2 * (int(height) // (VAE_STRIDE[1] * 2))
            width = 2 * (int(width) // (VAE_STRIDE[2] * 2))

            # reshape
            mask = mask[0, :, :, :]
            mask = mask.view(
                depth, height, VAE_STRIDE[1], width, VAE_STRIDE[1]
            )  # depth, height, 8, width, 8
            mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
            mask = mask.reshape(
                VAE_STRIDE[1] * VAE_STRIDE[2], depth, height, width
            )  # 8*8, depth, height, width

            # interpolation
            mask = F.interpolate(mask.unsqueeze(0), size=(new_depth, height, width), mode='nearest-exact').squeeze(0)

            if refs is not None:
                length = len(refs)
                mask_pad = torch.zeros_like(mask[:, :length, :, :])
                mask = torch.cat((mask_pad, mask), dim=1)
            result_masks.append(mask)
        return result_masks

    def vace_latent(self, z, m):
        return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]


#region context options
class WanVideoContextOptions:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "context_schedule": (["uniform_standard", "uniform_looped", "static_standard"],),
            "context_frames": ("INT", {"default": 81, "min": 2, "max": 1000, "step": 1, "tooltip": "Number of pixel frames in the context, NOTE: the latent space has 4 frames in 1"} ),
            "context_stride": ("INT", {"default": 4, "min": 4, "max": 100, "step": 1, "tooltip": "Context stride as pixel frames, NOTE: the latent space has 4 frames in 1"} ),
            "context_overlap": ("INT", {"default": 16, "min": 4, "max": 100, "step": 1, "tooltip": "Context overlap as pixel frames, NOTE: the latent space has 4 frames in 1"} ),
            "freenoise": ("BOOLEAN", {"default": True, "tooltip": "Shuffle the noise"}),
            "verbose": ("BOOLEAN", {"default": False, "tooltip": "Print debug output"}),
            },
            "optional": {
                "fuse_method": (["linear", "pyramid"], {"default": "linear", "tooltip": "Window weight function: linear=ramps at edges only, pyramid=triangular weights peaking in middle"}),
            }
        }

    RETURN_TYPES = ("WANVIDCONTEXT", )
    RETURN_NAMES = ("context_options",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Context options for WanVideo, allows splitting the video into context windows and attemps blending them for longer generations than the model and memory otherwise would allow."

    def process(self, context_schedule, context_frames, context_stride, context_overlap, freenoise, verbose, image_cond_start_step=6, image_cond_window_count=2, vae=None, fuse_method="linear"):
        context_options = {
            "context_schedule":context_schedule,
            "context_frames":context_frames,
            "context_stride":context_stride,
            "context_overlap":context_overlap,
            "freenoise":freenoise,
            "verbose":verbose,
            "fuse_method":fuse_method
        }

        return (context_options,)
    
    
class WanVideoFlowEdit:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "source_embeds": ("WANVIDEOTEXTEMBEDS", ),
                "skip_steps": ("INT", {"default": 4, "min": 0}),
                "drift_steps": ("INT", {"default": 0, "min": 0}),
                "drift_flow_shift": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 30.0, "step": 0.01}),
                "source_cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "drift_cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
            },
            "optional": {
                "source_image_embeds": ("WANVIDIMAGE_EMBEDS", ),
            }
        }

    RETURN_TYPES = ("FLOWEDITARGS", )
    RETURN_NAMES = ("flowedit_args",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Flowedit options for WanVideo"

    def process(self, **kwargs):
        return (kwargs,)
    
class WanVideoLoopArgs:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "shift_skip": ("INT", {"default": 6, "min": 0, "tooltip": "Skip step of latent shift"}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the looping effect"}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the looping effect"}),
            },
        }

    RETURN_TYPES = ("LOOPARGS", )
    RETURN_NAMES = ("loop_args",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Looping through latent shift as shown in https://github.com/YisuiTT/Mobius/"

    def process(self, **kwargs):
        return (kwargs,)

class WanVideoExperimentalArgs:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "video_attention_split_steps": ("STRING", {"default": "", "tooltip": "Steps to split self attention when using multiple prompts"}),
                "cfg_zero_star": ("BOOLEAN", {"default": False, "tooltip": "https://github.com/WeichenFan/CFG-Zero-star"}),
                "use_zero_init": ("BOOLEAN", {"default": False}),
                "zero_star_steps": ("INT", {"default": 0, "min": 0, "tooltip": "Steps to split self attention when using multiple prompts"}),
                "use_fresca": ("BOOLEAN", {"default": False, "tooltip": "https://github.com/WikiChao/FreSca"}),
                "fresca_scale_low": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "fresca_scale_high": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 10.0, "step": 0.01}),
                "fresca_freq_cutoff": ("INT", {"default": 20, "min": 0, "max": 10000, "step": 1}),
            },
        }

    RETURN_TYPES = ("EXPERIMENTALARGS", )
    RETURN_NAMES = ("exp_args",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Experimental stuff"
    EXPERIMENTAL = True

    def process(self, **kwargs):
        return (kwargs,)
    
class WanVideoFreeInitArgs:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "freeinit_num_iters": ("INT", {"default": 3, "min": 1, "max": 10, "tooltip": "Number of FreeInit iterations"}),
                "freeinit_method": (["butterworth", "ideal", "gaussian", "none"], {"default": "ideal", "tooltip": "Frequency filter type"}),
                "freeinit_n": ("INT", {"default": 4, "min": 1, "max": 10, "tooltip": "Butterworth filter order (only for butterworth)"}),
                "freeinit_d_s": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Spatial filter cutoff"}),
                "freeinit_d_t": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Temporal filter cutoff"}),
            },
        }

    RETURN_TYPES = ("FREEINITARGS", )
    RETURN_NAMES = ("freeinit_args",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "https://github.com/TianxingWu/FreeInit; FreeInit, a concise yet effective method to improve temporal consistency of videos generated by diffusion models"
    EXPERIMENTAL = True

    def process(self, **kwargs):
        return (kwargs,)
    
#region Sampler
class WanVideoSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                
                "image_embeds": ("WANVIDIMAGE_EMBEDS", ),
                "steps": ("INT", {"default": 30, "min": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Moves the model to the offload device after sampling"}),
                "scheduler": (scheduler_list, {"default": "uni_pc",}),
                "riflex_freq_index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": "Frequency index for RIFLEX, disabled when 0, default 6. Allows for new frames to be generated after without looping"}),
            },
            "optional": {
                "text_embeds": ("WANVIDEOTEXTEMBEDS", ),
                "samples": ("LATENT", {"tooltip": "init Latents to use for video2video process"} ),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feta_args": ("FETAARGS", ),
                "context_options": ("WANVIDCONTEXT", ),
                "cache_args": ("CACHEARGS", ),
                "flowedit_args": ("FLOWEDITARGS", ),
                "batched_cfg": ("BOOLEAN", {"default": False, "tooltip": "Batch cond and uncond for faster sampling, possibly faster on some hardware, uses more memory"}),
                "slg_args": ("SLGARGS", ),
                "rope_function": (["default", "comfy", "comfy_chunked"], {"default": "comfy", "tooltip": "Comfy's RoPE implementation doesn't use complex numbers and can thus be compiled, that should be a lot faster when using torch.compile. Chunked version has reduced peak VRAM usage when not using torch.compile"}),
                "loop_args": ("LOOPARGS", ),
                "experimental_args": ("EXPERIMENTALARGS", ),
                "sigmas": ("SIGMAS", ),
                "unianimate_poses": ("UNIANIMATE_POSE", ),
                "fantasytalking_embeds": ("FANTASYTALKING_EMBEDS", ),
                "uni3c_embeds": ("UNI3C_EMBEDS", ),
                "multitalk_embeds": ("MULTITALK_EMBEDS", ),
                "infinitetalk_embeds": ("MULTITALK_EMBEDS", ),
                "freeinit_args": ("FREEINITARGS", ),
            }
        }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, model, image_embeds, shift, steps, cfg, seed, scheduler, riflex_freq_index, text_embeds=None,
        force_offload=True, samples=None, feta_args=None, denoise_strength=1.0, context_options=None, 
        cache_args=None, teacache_args=None, flowedit_args=None, batched_cfg=False, slg_args=None, rope_function="default", loop_args=None, 
        experimental_args=None, sigmas=None, unianimate_poses=None, fantasytalking_embeds=None, uni3c_embeds=None, multitalk_embeds=None, infinitetalk_embeds=None, freeinit_args=None):
        
        patcher = model
        model = model.model
        transformer = model.diffusion_model
        dtype = model["dtype"]
        control_lora = model["control_lora"]

        multitalk_sampling = image_embeds.get("multitalk_sampling", False)
        if not multitalk_sampling and scheduler == "multitalk":
            raise Exception("multitalk scheduler is only for multitalk or infinitetalk sampling when using ImagetoVideoMultiTalk -node")

        transformer_options = patcher.model_options.get("transformer_options", None)
        
        steps = int(steps/denoise_strength)

        if text_embeds == None:
            text_embeds = {
                "prompt_embeds": [],
                "negative_prompt_embeds": [],
            }

        if isinstance(cfg, list):
            if steps != len(cfg):
                log.info(f"Received {len(cfg)} cfg values, but only {steps} steps. Setting step count to match.")
                steps = len(cfg)
        else:
            cfg = [cfg] * (steps +1)

        seed_g = torch.Generator(device=torch.device("cpu"))
        seed_g.manual_seed(seed)

        # Scheduler
        if scheduler != "multitalk":
            sample_scheduler, timesteps = get_scheduler(scheduler, steps, shift, device, transformer.dim, flowedit_args, denoise_strength, sigmas=sigmas)
        else:
            timesteps = torch.tensor([1000, 750, 500, 250], device=device)

        scheduler_step_args = {"generator": seed_g}
        step_sig = inspect.signature(sample_scheduler.step)
        for arg in list(scheduler_step_args.keys()):
            if arg not in step_sig.parameters:
                scheduler_step_args.pop(arg)
        
        if denoise_strength < 1.0:
            steps = int(steps * denoise_strength)
            timesteps = timesteps[-(steps + 1):] 
       
        control_latents = control_camera_latents = clip_fea = clip_fea_neg = end_image = recammaster = camera_embed = unianim_data = None
        vace_data = vace_context = vace_scale = None
        fun_or_fl2v_model = has_ref = drop_last = False
        phantom_latents = None
        fun_ref_image = None

        image_cond = image_embeds.get("image_embeds", None)
        ATI_tracks = None
        add_cond = attn_cond = attn_cond_neg = None
       
        if image_cond is not None:
            log.info(f"image_cond shape: {image_cond.shape}")
            #ATI tracks
            if transformer_options is not None:
                ATI_tracks = transformer_options.get("ati_tracks", None)
                if ATI_tracks is not None:
                    from .ATI.motion_patch import patch_motion
                    topk = transformer_options.get("ati_topk", 2)
                    temperature = transformer_options.get("ati_temperature", 220.0)
                    ati_start_percent = transformer_options.get("ati_start_percent", 0.0)
                    ati_end_percent = transformer_options.get("ati_end_percent", 1.0)
                    image_cond_ati = patch_motion(ATI_tracks.to(image_cond.device, image_cond.dtype), image_cond, topk=topk, temperature=temperature)
                    log.info(f"ATI tracks shape: {ATI_tracks.shape}")
            
            add_cond_latents = image_embeds.get("add_cond_latents", None)
            if add_cond_latents is not None:
                add_cond = add_cond_latents["pose_latent"]
                attn_cond = add_cond_latents["ref_latent"]
                attn_cond_neg = add_cond_latents["ref_latent_neg"]
                add_cond_start_percent = add_cond_latents["pose_cond_start_percent"]
                add_cond_end_percent = add_cond_latents["pose_cond_end_percent"]

            end_image = image_embeds.get("end_image", None)
            lat_h = image_embeds.get("lat_h", None)
            lat_w = image_embeds.get("lat_w", None)
            if lat_h is None or lat_w is None:
                raise ValueError("Clip encoded image embeds must be provided for I2V (Image to Video) model")
            fun_or_fl2v_model = image_embeds.get("fun_or_fl2v_model", False)
            noise = torch.randn(
                16,
                (image_embeds["num_frames"] - 1) // 4 + (2 if end_image is not None and not fun_or_fl2v_model else 1),
                lat_h,
                lat_w,
                dtype=torch.float32,
                generator=seed_g,
                device=torch.device("cpu"))
            seq_len = image_embeds["max_seq_len"]
            
            clip_fea = image_embeds.get("clip_context", None)
            if clip_fea is not None:
                clip_fea = clip_fea.to(dtype)
            clip_fea_neg = image_embeds.get("negative_clip_context", None)
            if clip_fea_neg is not None:
                clip_fea_neg = clip_fea_neg.to(dtype)

            control_embeds = image_embeds.get("control_embeds", None)
            if control_embeds is not None:
                if transformer.in_dim not in [48, 32]:
                    raise ValueError("Control signal only works with Fun-Control model")
                control_latents = control_embeds.get("control_images", None)
                control_camera_latents = control_embeds.get("control_camera_latents", None)
                control_camera_start_percent = control_embeds.get("control_camera_start_percent", 0.0)
                control_camera_end_percent = control_embeds.get("control_camera_end_percent", 1.0)
                control_start_percent = control_embeds.get("start_percent", 0.0)
                control_end_percent = control_embeds.get("end_percent", 1.0)
            drop_last = image_embeds.get("drop_last", False)
            has_ref = image_embeds.get("has_ref", False)
        else: #t2v
            target_shape = image_embeds.get("target_shape", None)
            if target_shape is None:
                raise ValueError("Empty image embeds must be provided for T2V (Text to Video")
            
            has_ref = image_embeds.get("has_ref", False)
            vace_context = image_embeds.get("vace_context", None)
            vace_scale = image_embeds.get("vace_scale", None)
            if not isinstance(vace_scale, list):
                vace_scale = [vace_scale] * (steps+1)
            vace_start_percent = image_embeds.get("vace_start_percent", 0.0)
            vace_end_percent = image_embeds.get("vace_end_percent", 1.0)
            vace_seqlen = image_embeds.get("vace_seq_len", None)

            vace_additional_embeds = image_embeds.get("additional_vace_inputs", [])
            if vace_context is not None:
                vace_data = [
                    {"context": vace_context, 
                     "scale": vace_scale, 
                     "start": vace_start_percent, 
                     "end": vace_end_percent,
                     "seq_len": vace_seqlen
                     }
                ]
                if len(vace_additional_embeds) > 0:
                    for i in range(len(vace_additional_embeds)):
                        if vace_additional_embeds[i].get("has_ref", False):
                            has_ref = True
                        vace_scale = vace_additional_embeds[i]["vace_scale"]
                        if not isinstance(vace_scale, list):
                            vace_scale = [vace_scale] * (steps+1)
                        vace_data.append({
                            "context": vace_additional_embeds[i]["vace_context"],
                            "scale": vace_scale,
                            "start": vace_additional_embeds[i]["vace_start_percent"],
                            "end": vace_additional_embeds[i]["vace_end_percent"],
                            "seq_len": vace_additional_embeds[i]["vace_seq_len"]
                        })

            noise = torch.randn(
                    target_shape[0],
                    target_shape[1] + 1 if has_ref else target_shape[1],
                    target_shape[2],
                    target_shape[3],
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                    generator=seed_g)
            
            seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * noise.shape[1])

            recammaster = image_embeds.get("recammaster", None)
            if recammaster is not None:
                camera_embed = recammaster.get("camera_embed", None)
                recam_latents = recammaster.get("source_latents", None)
                orig_noise_len = noise.shape[1]
                log.info(f"RecamMaster camera embed shape: {camera_embed.shape}")
                log.info(f"RecamMaster source video shape: {recam_latents.shape}")
                seq_len *= 2
            
            control_embeds = image_embeds.get("control_embeds", None)
            if control_embeds is not None:
                control_latents = control_embeds.get("control_images", None)
                if control_latents is not None:
                    control_latents = control_latents.to(device)
                control_camera_latents = control_embeds.get("control_camera_latents", None)
                control_camera_start_percent = control_embeds.get("control_camera_start_percent", 0.0)
                control_camera_end_percent = control_embeds.get("control_camera_end_percent", 1.0)
                if control_camera_latents is not None:
                    control_camera_latents = control_camera_latents.to(device)

                if control_lora:
                    image_cond = control_latents.to(device)
                    if not patcher.model.is_patched:
                        log.info("Re-loading control LoRA...")
                        patcher = apply_lora(patcher, device, device, low_mem_load=False)
                        patcher.model.is_patched = True
                else:
                    if transformer.in_dim not in [48, 32]:
                        raise ValueError("Control signal only works with Fun-Control model")
                    image_cond = torch.zeros_like(noise).to(device) #fun control
                    clip_fea = None
                    fun_ref_image = control_embeds.get("fun_ref_image", None)
                control_start_percent = control_embeds.get("start_percent", 0.0)
                control_end_percent = control_embeds.get("end_percent", 1.0)
            else:
                if transformer.in_dim == 36: #fun inp
                    mask_latents = torch.tile(
                        torch.zeros_like(noise[:1]), [4, 1, 1, 1]
                    )
                    masked_video_latents_input = torch.zeros_like(noise)
                    image_cond = torch.cat([mask_latents, masked_video_latents_input], dim=0).to(device)

            phantom_latents = image_embeds.get("phantom_latents", None)
            phantom_cfg_scale = image_embeds.get("phantom_cfg_scale", None)
            if not isinstance(phantom_cfg_scale, list):
                phantom_cfg_scale = [phantom_cfg_scale] * (steps +1)
            phantom_start_percent = image_embeds.get("phantom_start_percent", 0.0)
            phantom_end_percent = image_embeds.get("phantom_end_percent", 1.0)
            if phantom_latents is not None:
                phantom_latents = phantom_latents.to(device)

        latent_video_length = noise.shape[1]

        # Initialize FreeInit filter if enabled
        freq_filter = None
        if freeinit_args is not None:
            from .freeinit.freeinit_utils import get_freq_filter, freq_mix_3d
            filter_shape = list(noise.shape)  # [batch, C, T, H, W]
            freq_filter = get_freq_filter(
                filter_shape,
                device=device,
                filter_type=freeinit_args.get("freeinit_method", "butterworth"),
                n=freeinit_args.get("freeinit_n", 4) if freeinit_args.get("freeinit_method", "butterworth") == "butterworth" else None,
                d_s=freeinit_args.get("freeinit_s", 1.0),
                d_t=freeinit_args.get("freeinit_t", 1.0)
            )
            if samples is not None:
                saved_generator_state = samples.get("generator_state", None)
                if saved_generator_state is not None:
                    seed_g.set_state(saved_generator_state)
        
        # UniAnimate
        if unianimate_poses is not None:
            transformer.dwpose_embedding.to(device, model["dtype"])
            dwpose_data = unianimate_poses["pose"].to(device, model["dtype"])
            dwpose_data = torch.cat([dwpose_data[:,:,:1].repeat(1,1,3,1,1), dwpose_data], dim=2)
            dwpose_data = transformer.dwpose_embedding(dwpose_data)
            log.info(f"UniAnimate pose embed shape: {dwpose_data.shape}")
            if dwpose_data.shape[2] > latent_video_length:
                log.warning(f"UniAnimate pose embed length {dwpose_data.shape[2]} is longer than the video length {latent_video_length}, truncating")
                dwpose_data = dwpose_data[:,:, :latent_video_length]
            elif dwpose_data.shape[2] < latent_video_length:
                log.warning(f"UniAnimate pose embed length {dwpose_data.shape[2]} is shorter than the video length {latent_video_length}, padding with last pose")
                pad_len = latent_video_length - dwpose_data.shape[2]
                pad = dwpose_data[:,:,:1].repeat(1,1,pad_len,1,1)
                dwpose_data = torch.cat([dwpose_data, pad], dim=2)
            dwpose_data_flat = rearrange(dwpose_data, 'b c f h w -> b (f h w) c').contiguous()
            
            random_ref_dwpose_data = None
            if image_cond is not None:
                transformer.randomref_embedding_pose.to(device)
                random_ref_dwpose = unianimate_poses.get("ref", None)
                if random_ref_dwpose is not None:
                    random_ref_dwpose_data = transformer.randomref_embedding_pose(
                        random_ref_dwpose.to(device)
                        ).unsqueeze(2).to(model["dtype"]) # [1, 20, 104, 60]
                
            unianim_data = {
                "dwpose": dwpose_data_flat,
                "random_ref": random_ref_dwpose_data.squeeze(0) if random_ref_dwpose_data is not None else None,
                "strength": unianimate_poses["strength"],
                "start_percent": unianimate_poses["start_percent"],
                "end_percent": unianimate_poses["end_percent"]
            }

        # FantasyTalking
        audio_proj = multitalk_audio_embedding = infinitetalk_audio_embedding = None
        audio_scale = 1.0
        if fantasytalking_embeds is not None:
            audio_proj = fantasytalking_embeds["audio_proj"].to(device)
            audio_context_lens = fantasytalking_embeds["audio_context_lens"]
            audio_scale = fantasytalking_embeds["audio_scale"]
            audio_cfg_scale = fantasytalking_embeds["audio_cfg_scale"]
            if not isinstance(audio_cfg_scale, list):
                audio_cfg_scale = [audio_cfg_scale] * (steps +1)
            log.info(f"Audio proj shape: {audio_proj.shape}, audio context lens: {audio_context_lens}")
        elif multitalk_embeds is not None:
            # Handle single or multiple speaker embeddings
            audio_features_in = multitalk_embeds.get("audio_features", None)
            if audio_features_in is None:
                multitalk_audio_embedding = None
            else:
                if isinstance(audio_features_in, list):
                    multitalk_audio_embedding = [emb.to(device, dtype) for emb in audio_features_in]
                else:
                    # keep backward-compatibility with single tensor input
                    multitalk_audio_embedding = [audio_features_in.to(device, dtype)]

            audio_scale = multitalk_embeds.get("audio_scale", 1.0)
            audio_cfg_scale = multitalk_embeds.get("audio_cfg_scale", 1.0)
            ref_target_masks = multitalk_embeds.get("ref_target_masks", None)
            if not isinstance(audio_cfg_scale, list):
                audio_cfg_scale = [audio_cfg_scale] * (steps + 1)

            shapes = [tuple(e.shape) for e in multitalk_audio_embedding]
            log.info(f"Multitalk audio features shapes (per speaker): {shapes}")
        elif infinitetalk_embeds is not None:
            # Handle single or multiple speaker embeddings
            audio_features_in = infinitetalk_embeds.get("audio_features", None)
            if audio_features_in is None:
                multitalk_audio_embedding = None
            else:
                if isinstance(audio_features_in, list):
                    multitalk_audio_embedding = [emb.to(device, dtype) for emb in audio_features_in]
                else:
                    # keep backward-compatibility with single tensor input
                    multitalk_audio_embedding = [audio_features_in.to(device, dtype)]

            audio_scale = infinitetalk_embeds.get("audio_scale", 1.0)
            audio_cfg_scale = infinitetalk_embeds.get("audio_cfg_scale", 1.0)
            ref_target_masks = infinitetalk_embeds.get("ref_target_masks", None)
            if not isinstance(audio_cfg_scale, list):
                audio_cfg_scale = [audio_cfg_scale] * (steps + 1)

            shapes = [tuple(e.shape) for e in multitalk_audio_embedding]
            log.info(f"Infinitetalk audio features shapes (per speaker): {shapes}")
    
        # MiniMax Remover
        minimax_latents = minimax_mask_latents = None
        minimax_latents = image_embeds.get("minimax_latents", None)
        minimax_mask_latents = image_embeds.get("minimax_mask_latents", None)
        if minimax_latents is not None:
            log.info(f"minimax_latents: {minimax_latents.shape}")
            log.info(f"minimax_mask_latents: {minimax_mask_latents.shape}")
            minimax_latents = minimax_latents.to(device, dtype)
            minimax_mask_latents = minimax_mask_latents.to(device, dtype)

        # Context windows
        is_looped = False
        if context_options is not None:            
            context_schedule = context_options["context_schedule"]
            context_frames =  (context_options["context_frames"] - 1) // 4 + 1
            context_stride = context_options["context_stride"] // 4
            context_overlap = context_options["context_overlap"] // 4
            context_vae = context_options.get("vae", None)
            if context_vae is not None:
                context_vae.to(device)

            # Get total number of prompts
            num_prompts = len(text_embeds["prompt_embeds"])
            log.info(f"Number of prompts: {num_prompts}")
            # Calculate which section this context window belongs to
            section_size = latent_video_length / num_prompts
            log.info(f"Section size: {section_size}")
            is_looped = context_schedule == "uniform_looped"

            seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * context_frames)

            if context_options["freenoise"]:
                log.info("Applying FreeNoise")
                # code from AnimateDiff-Evolved by Kosinkadink (https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved)
                delta = context_frames - context_overlap
                for start_idx in range(0, latent_video_length-context_frames, delta):
                    place_idx = start_idx + context_frames
                    if place_idx >= latent_video_length:
                        break
                    end_idx = place_idx - 1

                    if end_idx + delta >= latent_video_length:
                        final_delta = latent_video_length - place_idx
                        list_idx = torch.tensor(list(range(start_idx,start_idx+final_delta)), device=torch.device("cpu"), dtype=torch.long)
                        list_idx = list_idx[torch.randperm(final_delta, generator=seed_g)]
                        noise[:, place_idx:place_idx + final_delta, :, :] = noise[:, list_idx, :, :]
                        break
                    list_idx = torch.tensor(list(range(start_idx,start_idx+delta)), device=torch.device("cpu"), dtype=torch.long)
                    list_idx = list_idx[torch.randperm(delta, generator=seed_g)]
                    noise[:, place_idx:place_idx + delta, :, :] = noise[:, list_idx, :, :]
            
            log.info(f"Context schedule enabled: {context_frames} frames, {context_stride} stride, {context_overlap} overlap")
            from .context_windows.context import get_context_scheduler, create_window_mask, WindowTracker
            self.window_tracker = WindowTracker(verbose=context_options["verbose"])
            context = get_context_scheduler(context_schedule)

        # vid2vid
        if samples is not None:
            input_samples = samples["samples"].squeeze(0).to(noise)
            if input_samples.shape[1] != noise.shape[1]:
                input_samples = torch.cat([input_samples[:, :1].repeat(1, noise.shape[1] - input_samples.shape[1], 1, 1), input_samples], dim=1)
            original_image = input_samples.to(device)
            if denoise_strength < 1.0:
                latent_timestep = timesteps[:1].to(noise)
                noise = noise * latent_timestep / 1000 + (1 - latent_timestep / 1000) * input_samples

            mask = samples.get("mask", None)
            if mask is not None:
                if mask.shape[2] != noise.shape[1]:
                    mask = torch.cat([torch.zeros(1, noise.shape[0], noise.shape[1] - mask.shape[2], noise.shape[2], noise.shape[3]), mask], dim=2)
        
        # extra latents (Pusa)
        if (extra_latents := image_embeds.get("extra_latents", None)) is not None:
            encoded_image_latents = extra_latents["samples"].squeeze(0).to(noise)
            if (empty_latent_indices := extra_latents.get("empty_latent_indices", None)) is not None and len(empty_latent_indices) > 0:
                noise_out = encoded_image_latents.clone()
                for idx in empty_latent_indices:
                    #print(f"Adding noise to Empty latent index: {idx}")
                    noise_out[:, idx] = noise[:, idx]
                noise = noise_out
            else:
                noise[:,0:encoded_image_latents.shape[1]] = encoded_image_latents

        latent = noise.to(device)

        #controlnet
        controlnet_latents = controlnet = None
        if transformer_options is not None:
            controlnet = transformer_options.get("controlnet", None)
            if controlnet is not None:
                self.controlnet = controlnet["controlnet"]
                controlnet_start = controlnet["controlnet_start"]
                controlnet_end = controlnet["controlnet_end"]
                controlnet_latents = controlnet["control_latents"]
                controlnet["controlnet_weight"] = controlnet["controlnet_strength"]
                controlnet["controlnet_stride"] = controlnet["control_stride"]

        #uni3c
        pcd_data = pcd_data_input = None
        if uni3c_embeds is not None:
            transformer.controlnet = uni3c_embeds["controlnet"]
            pcd_data = {
                "render_latent": uni3c_embeds["render_latent"].to(dtype),
                "render_mask": uni3c_embeds["render_mask"],
                "camera_embedding": uni3c_embeds["camera_embedding"],
                "controlnet_weight": uni3c_embeds["controlnet_weight"],
                "start": uni3c_embeds["start"],
                "end": uni3c_embeds["end"],
            }

        # Enhance-a-video (feta)
        if feta_args is not None and latent_video_length > 1:
            set_enhance_weight(feta_args["weight"])
            feta_start_percent = feta_args["start_percent"]
            feta_end_percent = feta_args["end_percent"]
            if context_options is not None:
                set_num_frames(context_frames)
            else:
                set_num_frames(latent_video_length)
            enhance_enabled = True
        else:
            feta_args = None
            enhance_enabled = False

        #region transformer settings
        #rope
        freqs = None
        transformer.rope_embedder.k = None
        transformer.rope_embedder.num_frames = None
        if "comfy" in rope_function:
            transformer.rope_embedder.k = riflex_freq_index
            transformer.rope_embedder.num_frames = latent_video_length
        else:
            d = transformer.dim // transformer.num_heads
            freqs = torch.cat([
                rope_params(1024, d - 4 * (d // 6), L_test=latent_video_length, k=riflex_freq_index),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
            dim=1)
        transformer.rope_func = rope_function
        for block in transformer.blocks:
            block.rope_func = rope_function
        if transformer.vace_layers is not None:
            for block in transformer.vace_blocks:
                block.rope_func = rope_function

        #blockswap init        
        if transformer_options is not None:
            block_swap_args = transformer_options.get("block_swap_args", None)

        if block_swap_args is not None:
            transformer.use_non_blocking = block_swap_args.get("use_non_blocking", True)
            for name, param in transformer.named_parameters():
                if "block" not in name:
                    param.data = param.data.to(device)
                if "control_adapter" in name:
                    param.data = param.data.to(device)
                elif block_swap_args["offload_txt_emb"] and "txt_emb" in name:
                    param.data = param.data.to(offload_device, non_blocking=transformer.use_non_blocking)
                elif block_swap_args["offload_img_emb"] and "img_emb" in name:
                    param.data = param.data.to(offload_device, non_blocking=transformer.use_non_blocking)

            transformer.block_swap(
                block_swap_args["blocks_to_swap"] - 1 ,
                block_swap_args["offload_txt_emb"],
                block_swap_args["offload_img_emb"],
                vace_blocks_to_swap = block_swap_args.get("vace_blocks_to_swap", None),
            )
        elif model["auto_cpu_offload"]:
            for module in transformer.modules():
                if hasattr(module, "offload"):
                    module.offload()
                if hasattr(module, "onload"):
                    module.onload()
        elif model["manual_offloading"]:
            transformer.to(device)

        # Initialize Cache if enabled
        transformer.enable_teacache = transformer.enable_magcache = transformer.enable_easycache = False
        cache_args = teacache_args if teacache_args is not None else cache_args #for backward compatibility on old workflows
        if cache_args is not None:            
           from .cache_methods.cache_methods import set_transformer_cache_method
           transformer = set_transformer_cache_method(transformer, timesteps, cache_args)

        # Initialize cache state
        self.cache_state = [None, None]
        if phantom_latents is not None:
            log.info(f"Phantom latents shape: {phantom_latents.shape}")
            self.cache_state = [None, None, None]
        self.cache_state_source = [None, None]
        self.cache_states_context = []

        # Skip layer guidance (SLG)
        if slg_args is not None:
            assert batched_cfg is not None, "Batched cfg is not supported with SLG"
            transformer.slg_blocks = slg_args["blocks"]
            transformer.slg_start_percent = slg_args["start_percent"]
            transformer.slg_end_percent = slg_args["end_percent"]
        else:
            transformer.slg_blocks = None

        # Radial attention setup
        if transformer.attention_mode == "radial_sage_attention":
            dense_timesteps = transformer_options.get("dense_timesteps", None)
            dense_blocks = transformer_options.get("dense_blocks", None)
            dense_vace_blocks = transformer_options.get("dense_vace_blocks", None)
            decay_factor = transformer_options.get("decay_factor", None)
            dense_attention_mode = transformer_options.get("dense_attention_mode", None)
            block_size = transformer_options.get("block_size", None)

            # Calculate closest valid latent sizes
            if latent.shape[2] % (block_size/8) != 0 or latent.shape[3] % (block_size/8) != 0:
                block_div = int(block_size // 8)
                closest_h = round(latent.shape[2] / block_div) * block_div
                closest_w = round(latent.shape[3] / block_div) * block_div
                raise Exception(
                    f"Radial attention mode only supports image size divisible by block size. "
                    f"Got {latent.shape[3] * 8}x{latent.shape[2] * 8} with block size {block_size}.\n"
                    f"Closest valid sizes: {closest_w * 8}x{closest_h * 8} (width x height in pixels)."
                )
            tokens_per_frame = (latent.shape[2] * latent.shape[3]) // 4
            if tokens_per_frame % block_size != 0:
                closest_latent_h = find_closest_valid_dim(latent.shape[3], latent.shape[2], block_size)
                closest_latent_w = find_closest_valid_dim(latent.shape[2], latent.shape[3], block_size)
                raise Exception(
                    f"Radial attention mode requires tokens per frame ((latent_h * latent_w) // 4) to be divisible by block size ({block_size}).\n"
                    f"Current size in latent space:{latent.shape[3]}x{latent.shape[2]}, pixel space: {latent.shape[3]*8}x{latent.shape[2]*8} tokens_per_frame={tokens_per_frame}.\n"
                    f"Try adjusting to one of these latent sizes (in pixels):\n"
                    f"  Height: {latent.shape[2]*8} -> {closest_latent_h * 8}\n"
                    f"  Width: {latent.shape[3]*8} -> {closest_latent_w * 8}\n"
                    f"Or choose another resolution so that (latent_h * latent_w) // 4 is divisible by {block_size}."
                )

            from .wanvideo.radial_attention.attn_mask import MaskMap
            for i, block in enumerate(transformer.blocks):
                block.self_attn.mask_map = block.dense_attention_mode = block.dense_timesteps = block.self_attn.decay_factor = None
                if isinstance(dense_blocks, list):
                    block.dense_block = i in dense_blocks
                else:
                    block.dense_block = i < dense_blocks
                block.self_attn.mask_map = MaskMap(video_token_num=seq_len, num_frame=latent_video_length if context_options is None else context_frames, block_size=block_size)
                block.dense_attention_mode = dense_attention_mode
                block.dense_timesteps = dense_timesteps
                block.self_attn.decay_factor = decay_factor
            if transformer.vace_layers is not None:
                for i, block in enumerate(transformer.vace_blocks):
                    block.self_attn.mask_map = block.dense_attention_mode = block.dense_timesteps = block.self_attn.decay_factor = None
                    if isinstance(dense_vace_blocks, list):
                        block.dense_block = i in dense_vace_blocks
                    else:
                        block.dense_block = i < dense_vace_blocks
                    block.self_attn.mask_map = MaskMap(video_token_num=seq_len, num_frame=latent_video_length if context_options is None else context_frames, block_size=block_size)
                    block.dense_attention_mode = dense_attention_mode
                    block.dense_timesteps = dense_timesteps
                    block.self_attn.decay_factor = decay_factor
                            
            log.info(f"Radial attention mode enabled.")
            log.info(f"dense_attention_mode: {dense_attention_mode}, dense_timesteps: {dense_timesteps}, decay_factor: {decay_factor}")
            log.info(f"dense_blocks: {[i for i, block in enumerate(transformer.blocks) if getattr(block, 'dense_block', False)]})")


        # FlowEdit setup
        if flowedit_args is not None:
            source_embeds = flowedit_args["source_embeds"]
            source_image_embeds = flowedit_args.get("source_image_embeds", image_embeds)
            source_image_cond = source_image_embeds.get("image_embeds", None)
            source_clip_fea = source_image_embeds.get("clip_fea", clip_fea)
            if source_image_cond is not None:
                source_image_cond = source_image_cond.to(dtype)
            skip_steps = flowedit_args["skip_steps"]
            drift_steps = flowedit_args["drift_steps"]
            source_cfg = flowedit_args["source_cfg"]
            if not isinstance(source_cfg, list):
                source_cfg = [source_cfg] * (steps +1)
            drift_cfg = flowedit_args["drift_cfg"]
            if not isinstance(drift_cfg, list):
                drift_cfg = [drift_cfg] * (steps +1)

            x_init = samples["samples"].clone().squeeze(0).to(device)
            x_tgt = samples["samples"].squeeze(0).to(device)

            sample_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=flowedit_args["drift_flow_shift"],
                use_dynamic_shifting=False)

            sampling_sigmas = get_sampling_sigmas(steps, flowedit_args["drift_flow_shift"])
           
            drift_timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=device,
                sigmas=sampling_sigmas)

            if drift_steps > 0:
                drift_timesteps = torch.cat([drift_timesteps, torch.tensor([0]).to(drift_timesteps.device)]).to(drift_timesteps.device)
                timesteps[-drift_steps:] = drift_timesteps[-drift_steps:]

        # Experimental args
        use_cfg_zero_star = use_fresca = False
        if experimental_args is not None:
            video_attention_split_steps = experimental_args.get("video_attention_split_steps", [])
            if video_attention_split_steps:
                transformer.video_attention_split_steps = [int(x.strip()) for x in video_attention_split_steps.split(",")]
            else:
                transformer.video_attention_split_steps = []

            use_zero_init = experimental_args.get("use_zero_init", True)
            use_cfg_zero_star = experimental_args.get("cfg_zero_star", False)
            zero_star_steps = experimental_args.get("zero_star_steps", 0)

            use_fresca = experimental_args.get("use_fresca", False)
            if use_fresca:
                fresca_scale_low = experimental_args.get("fresca_scale_low", 1.0)
                fresca_scale_high = experimental_args.get("fresca_scale_high", 1.25)
                fresca_freq_cutoff = experimental_args.get("fresca_freq_cutoff", 20)

        #region model pred
        def predict_with_cfg(z, cfg_scale, positive_embeds, negative_embeds, timestep, idx, image_cond=None, clip_fea=None, 
                             control_latents=None, vace_data=None, unianim_data=None, audio_proj=None, control_camera_latents=None, 
                             add_cond=None, cache_state=None, context_window=None, multitalk_audio_embeds=None):
            z = z.to(dtype)
            with torch.autocast(device_type=mm.get_autocast_device(device), dtype=dtype, enabled=("fp8" in model["quantization"])):

                if use_cfg_zero_star and (idx <= zero_star_steps) and use_zero_init:
                    return z*0, None

                nonlocal patcher
                current_step_percentage = idx / len(timesteps)
                control_lora_enabled = False
                image_cond_input = None
                if control_latents is not None:
                    if control_lora:
                        control_lora_enabled = True
                    else:
                        if (control_start_percent <= current_step_percentage <= control_end_percent) or \
                            (control_end_percent > 0 and idx == 0 and current_step_percentage >= control_start_percent):
                            image_cond_input = torch.cat([control_latents.to(z), image_cond.to(z)])
                        else:
                            image_cond_input = torch.cat([torch.zeros_like(image_cond, dtype=dtype), image_cond.to(z)])
                        if fun_ref_image is not None:
                            fun_ref_input = fun_ref_image.to(z)
                        else:
                            fun_ref_input = torch.zeros_like(z, dtype=z.dtype)[:, 0].unsqueeze(1)
                            #fun_ref_input = None

                    if control_lora:
                        if not control_start_percent <= current_step_percentage <= control_end_percent:
                            control_lora_enabled = False
                            if patcher.model.is_patched:
                                log.info("Unloading LoRA...")
                                patcher.unpatch_model(device)
                                patcher.model.is_patched = False
                        else:
                            image_cond_input = control_latents.to(z)
                            if not patcher.model.is_patched:
                                log.info("Loading LoRA...")
                                patcher = apply_lora(patcher, device, device, low_mem_load=False)
                                patcher.model.is_patched = True
                                
                elif ATI_tracks is not None and ((ati_start_percent <= current_step_percentage <= ati_end_percent) or 
                              (ati_end_percent > 0 and idx == 0 and current_step_percentage >= ati_start_percent)):
                    image_cond_input = image_cond_ati.to(z)
                else:
                    image_cond_input = image_cond.to(z) if image_cond is not None else None

                if control_camera_latents is not None:
                    if (control_camera_start_percent <= current_step_percentage <= control_camera_end_percent) or \
                            (control_end_percent > 0 and idx == 0 and current_step_percentage >= control_camera_start_percent):
                        control_camera_input = control_camera_latents.to(z)
                    else:
                        control_camera_input = None

                if recammaster is not None:
                    z = torch.cat([z, recam_latents.to(z)], dim=1)
                    
                use_phantom = False
                if phantom_latents is not None:
                    if (phantom_start_percent <= current_step_percentage <= phantom_end_percent) or \
                        (phantom_end_percent > 0 and idx == 0 and current_step_percentage >= phantom_start_percent):

                        z_pos = torch.cat([z[:,:-phantom_latents.shape[1]], phantom_latents.to(z)], dim=1)
                        z_phantom_img = torch.cat([z[:,:-phantom_latents.shape[1]], phantom_latents.to(z)], dim=1)
                        z_neg = torch.cat([z[:,:-phantom_latents.shape[1]], torch.zeros_like(phantom_latents).to(z)], dim=1)
                        use_phantom = True
                        if cache_state is not None and len(cache_state) != 3:
                            cache_state.append(None)
                if not use_phantom:
                    z_pos = z_neg = z

                if controlnet_latents is not None:
                    if (controlnet_start <= current_step_percentage < controlnet_end):
                        self.controlnet.to(device)
                        controlnet_states = self.controlnet(
                            hidden_states=z.unsqueeze(0).to(device, self.controlnet.dtype),
                            timestep=timestep,
                            encoder_hidden_states=positive_embeds[0].unsqueeze(0).to(device, self.controlnet.dtype),
                            attention_kwargs=None,
                            controlnet_states=controlnet_latents.to(device, self.controlnet.dtype),
                            return_dict=False,
                        )[0]
                        if isinstance(controlnet_states, (tuple, list)):
                            controlnet["controlnet_states"] = [x.to(z) for x in controlnet_states]
                        else:
                            controlnet["controlnet_states"] = controlnet_states.to(z)

                add_cond_input = None
                if add_cond is not None:
                    if (add_cond_start_percent <= current_step_percentage <= add_cond_end_percent) or \
                        (add_cond_end_percent > 0 and idx == 0 and current_step_percentage >= add_cond_start_percent):
                        add_cond_input = add_cond

                if minimax_latents is not None:
                    z_pos = z_neg = torch.cat([z, minimax_latents, minimax_mask_latents], dim=0)
                
                if not multitalk_sampling and multitalk_audio_embedding is not None:
                    audio_embedding = multitalk_audio_embedding
                    audio_embs = []
                    indices = (torch.arange(4 + 1) - 2) * 1
                    human_num = len(audio_embedding)
                    # split audio with window size
                    if context_window is None:
                        for human_idx in range(human_num):   
                            center_indices = torch.arange(
                                0,
                                latent_video_length * 4 + 1 if add_cond is not None else (latent_video_length-1) * 4 + 1,
                                1).unsqueeze(1) + indices.unsqueeze(0)
                            center_indices = torch.clamp(center_indices, min=0, max=audio_embedding[human_idx].shape[0] - 1)
                            audio_emb = audio_embedding[human_idx][center_indices].unsqueeze(0).to(device)
                            audio_embs.append(audio_emb)
                    else:
                        for human_idx in range(human_num):
                            audio_start = context_window[0] * 4
                            audio_end = context_window[-1] * 4 + 1
                            print("audio_start: ", audio_start, "audio_end: ", audio_end)
                            center_indices = torch.arange(audio_start, audio_end, 1).unsqueeze(1) + indices.unsqueeze(0)
                            center_indices = torch.clamp(center_indices, min=0, max=audio_embedding[human_idx].shape[0] - 1)
                            audio_emb = audio_embedding[human_idx][center_indices].unsqueeze(0).to(device)
                            audio_embs.append(audio_emb)
                    multitalk_audio_input = torch.concat(audio_embs, dim=0).to(dtype)
                    
                elif multitalk_sampling and multitalk_audio_embeds is not None:
                    multitalk_audio_input = multitalk_audio_embeds
                
                if context_window is not None and pcd_data is not None and pcd_data["render_latent"].shape[2] != context_frames:
                    pcd_data_input = {"render_latent": pcd_data["render_latent"][:, :, context_window]}
                    for k in pcd_data:
                        if k != "render_latent":
                            pcd_data_input[k] = pcd_data[k]
                else:
                    pcd_data_input = pcd_data

                 
                base_params = {
                    'seq_len': seq_len,
                    'device': device,
                    'freqs': freqs,
                    't': timestep,
                    'current_step': idx,
                    'control_lora_enabled': control_lora_enabled,
                    'enhance_enabled': enhance_enabled,
                    'camera_embed': camera_embed,
                    'unianim_data': unianim_data,
                    'fun_ref': fun_ref_input if fun_ref_image is not None else None,
                    'fun_camera': control_camera_input if control_camera_latents is not None else None,
                    'audio_proj': audio_proj if fantasytalking_embeds is not None else None,
                    'audio_context_lens': audio_context_lens if fantasytalking_embeds is not None else None,
                    'audio_scale': audio_scale,
                    "pcd_data": pcd_data_input,
                    "controlnet": controlnet,
                    "add_cond": add_cond_input,
                    "nag_params": text_embeds.get("nag_params", {}),
                    "nag_context": text_embeds.get("nag_prompt_embeds", None),
                    "multitalk_audio": multitalk_audio_input if multitalk_audio_embedding is not None else None,
                    "ref_target_masks": ref_target_masks if multitalk_audio_embedding is not None else None,
                }

                batch_size = 1

                if not math.isclose(cfg_scale, 1.0) and len(positive_embeds) > 1:
                    negative_embeds = negative_embeds * len(positive_embeds)

                if not batched_cfg:
                    #cond
                    noise_pred_cond, cache_state_cond = transformer(
                        [z_pos], context=positive_embeds, y=[image_cond_input] if image_cond_input is not None else None,
                        clip_fea=clip_fea, is_uncond=False, current_step_percentage=current_step_percentage,
                        pred_id=cache_state[0] if cache_state else None,
                        vace_data=vace_data, attn_cond=attn_cond,
                        **base_params
                    )
                    noise_pred_cond = noise_pred_cond[0].to(intermediate_device)
                    if math.isclose(cfg_scale, 1.0):
                        if use_fresca:
                            noise_pred_cond = fourier_filter(
                                noise_pred_cond,
                                scale_low=fresca_scale_low,
                                scale_high=fresca_scale_high,
                                freq_cutoff=fresca_freq_cutoff,
                            )
                        return noise_pred_cond, [cache_state_cond]
                    #uncond
                    if fantasytalking_embeds is not None:
                        if not math.isclose(audio_cfg_scale[idx], 1.0):
                            base_params['audio_proj'] = None
                    noise_pred_uncond, cache_state_uncond = transformer(
                        [z_neg], context=negative_embeds, clip_fea=clip_fea_neg if clip_fea_neg is not None else clip_fea,
                        y=[image_cond_input] if image_cond_input is not None else None, 
                        is_uncond=True, current_step_percentage=current_step_percentage,
                        pred_id=cache_state[1] if cache_state else None,
                        vace_data=vace_data, attn_cond=attn_cond_neg,
                        **base_params
                    )
                    noise_pred_uncond = noise_pred_uncond[0].to(intermediate_device)
                    #phantom
                    if use_phantom and not math.isclose(phantom_cfg_scale[idx], 1.0):
                        noise_pred_phantom, cache_state_phantom = transformer(
                        [z_phantom_img], context=negative_embeds, clip_fea=clip_fea_neg if clip_fea_neg is not None else clip_fea,
                        y=[image_cond_input] if image_cond_input is not None else None, 
                        is_uncond=True, current_step_percentage=current_step_percentage,
                        pred_id=cache_state[2] if cache_state else None,
                        vace_data=None,
                        **base_params
                    )
                        noise_pred_phantom = noise_pred_phantom[0].to(intermediate_device)
                        
                        noise_pred = noise_pred_uncond + phantom_cfg_scale[idx] * (noise_pred_phantom - noise_pred_uncond) + cfg_scale * (noise_pred_cond - noise_pred_phantom)
                        return noise_pred, [cache_state_cond, cache_state_uncond, cache_state_phantom]
                    #fantasytalking
                    if fantasytalking_embeds is not None:
                        if not math.isclose(audio_cfg_scale[idx], 1.0):
                            if cache_state is not None and len(cache_state) != 3:
                                cache_state.append(None)
                            base_params['audio_proj'] = None
                            noise_pred_no_audio, cache_state_audio = transformer(
                                [z_pos], context=positive_embeds, y=[image_cond_input] if image_cond_input is not None else None,
                                clip_fea=clip_fea, is_uncond=False, current_step_percentage=current_step_percentage,
                                pred_id=cache_state[2] if cache_state else None,
                                vace_data=vace_data,
                                **base_params
                            )
                            noise_pred_no_audio = noise_pred_no_audio[0].to(intermediate_device)
                            noise_pred = (
                                noise_pred_uncond
                                + cfg_scale * (noise_pred_no_audio - noise_pred_uncond)
                                + audio_cfg_scale[idx] * (noise_pred_cond - noise_pred_no_audio)
                                )
                            return noise_pred, [cache_state_cond, cache_state_uncond, cache_state_audio]
                    elif multitalk_audio_embedding is not None:
                        if not math.isclose(audio_cfg_scale[idx], 1.0):
                            if cache_state is not None and len(cache_state) != 3:
                                cache_state.append(None)
                            base_params['multitalk_audio'] = torch.zeros_like(multitalk_audio_input)[-1:]
                            noise_pred_no_audio, cache_state_audio = transformer(
                                [z_pos], context=negative_embeds, y=[image_cond_input] if image_cond_input is not None else None,
                                clip_fea=clip_fea, is_uncond=False, current_step_percentage=current_step_percentage,
                                pred_id=cache_state[2] if cache_state else None,
                                vace_data=vace_data,
                                **base_params
                            )
                            noise_pred_no_audio = noise_pred_no_audio[0].to(intermediate_device)
                            noise_pred = (
                                noise_pred_no_audio
                                + cfg_scale * (noise_pred_cond - noise_pred_uncond)
                                + audio_cfg_scale[idx] * (noise_pred_uncond - noise_pred_no_audio)
                                )
                            return noise_pred, [cache_state_cond, cache_state_uncond, cache_state_audio]

                #batched
                else:
                    cache_state_uncond = None
                    [noise_pred_cond, noise_pred_uncond], cache_state_cond = transformer(
                        [z] + [z], context=positive_embeds + negative_embeds, 
                        y=[image_cond_input] + [image_cond_input] if image_cond_input is not None else None,
                        clip_fea=clip_fea.repeat(2,1,1), is_uncond=False, current_step_percentage=current_step_percentage,
                        pred_id=cache_state[0] if cache_state else None,
                        **base_params
                    )
                #cfg

                #https://github.com/WeichenFan/CFG-Zero-star/
                if use_cfg_zero_star:
                    alpha = optimized_scale(
                        noise_pred_cond.view(batch_size, -1),
                        noise_pred_uncond.view(batch_size, -1)
                    ).view(batch_size, 1, 1, 1)
                else:
                    alpha = 1.0

                #https://github.com/WikiChao/FreSca
                if use_fresca:
                    filtered_cond = fourier_filter(
                        noise_pred_cond - noise_pred_uncond,
                        scale_low=fresca_scale_low,
                        scale_high=fresca_scale_high,
                        freq_cutoff=fresca_freq_cutoff,
                    )
                    noise_pred = noise_pred_uncond * alpha + cfg_scale * filtered_cond * alpha
                else:
                    noise_pred = noise_pred_uncond * alpha + cfg_scale * (noise_pred_cond - noise_pred_uncond * alpha)
                

                return noise_pred, [cache_state_cond, cache_state_uncond]
            
        log.info(f"Seq len: {seq_len}")
           
        pbar = ProgressBar(steps)

        if args.preview_method in [LatentPreviewMethod.Auto, LatentPreviewMethod.Latent2RGB]: #default for latent2rgb
            from latent_preview import prepare_callback
        else:
            from .latent_preview import prepare_callback #custom for tiny VAE previews
        callback = prepare_callback(patcher, steps)

        log.info(f"Sampling {(latent_video_length-1) * 4 + 1} frames at {latent.shape[3]*8}x{latent.shape[2]*8} with {steps} steps")

        intermediate_device = device

        # diff diff prep
        masks = None
        if samples is not None and mask is not None:
            mask = 1 - mask
            thresholds = torch.arange(len(timesteps), dtype=original_image.dtype) / len(timesteps)
            thresholds = thresholds.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1).to(device)
            masks = mask.repeat(len(timesteps), 1, 1, 1, 1).to(device) 
            masks = masks > thresholds

        latent_shift_loop = False
        if loop_args is not None:
            latent_shift_loop = True
            is_looped = True
            latent_skip = loop_args["shift_skip"]
            latent_shift_start_percent = loop_args["start_percent"]
            latent_shift_end_percent = loop_args["end_percent"]
            shift_idx = 0

        #clear memory before sampling
        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()
        try:
            torch.cuda.reset_peak_memory_stats(device)
            #torch.cuda.memory._record_memory_history(max_entries=100000)
        except:
            pass

        # Main sampling loop with FreeInit iterations
        iterations = freeinit_args.get("freeinit_num_iters", 3) if freeinit_args is not None else 1
        current_latent = latent
        
        for iter_idx in range(iterations): 
            # FreeInit noise reinitialization (after first iteration)
            if freeinit_args is not None and iter_idx > 0:
                # restart scheduler for each iteration
                sample_scheduler, timesteps = get_scheduler(scheduler, steps, shift, device, transformer.dim, flowedit_args, denoise_strength, sigmas=sigmas)

                # Diffuse current latent to t=999
                diffuse_timesteps = torch.full((noise.shape[0],), 999, device=device, dtype=torch.long)
                z_T = add_noise(
                    current_latent.to(device),
                    initial_noise_saved.to(device),
                    diffuse_timesteps
                )

                # Generate new random noise
                z_rand = torch.randn(z_T.shape, dtype=torch.float32, generator=seed_g, device=torch.device("cpu"))

                # Apply frequency mixing
                current_latent = freq_mix_3d(z_T.to(torch.float32), z_rand.to(device), LPF=freq_filter)
                current_latent = current_latent.to(dtype)

            # Store initial noise for first iteration
            if freeinit_args is not None and iter_idx == 0:
                initial_noise_saved = current_latent.detach().clone()
                if samples is not None:
                    current_latent = input_samples.to(device)
                    continue
            
            # Reset per-iteration states
            self.cache_state = [None, None]
            self.cache_state_source = [None, None]
            self.cache_states_context = []
            if context_options is not None:
                self.window_tracker = WindowTracker(verbose=context_options["verbose"])
            
            # Set latent for denoising
            latent = current_latent

            #region main loop start
            for idx, t in enumerate(tqdm(timesteps)):
                if flowedit_args is not None:
                    if idx < skip_steps:
                        continue

                # diff diff
                if masks is not None:
                    if idx < len(timesteps) - 1:
                        noise_timestep = timesteps[idx+1]
                        image_latent = sample_scheduler.scale_noise(
                            original_image, torch.tensor([noise_timestep]), noise.to(device)
                        )
                        mask = masks[idx]
                        mask = mask.to(latent)
                        latent = image_latent * mask + latent * (1-mask)
                        # end diff diff

                latent_model_input = latent.to(device)

                timestep = torch.tensor([t]).to(device)
                if scheduler == "flowmatch_pusa":
                    timestep = timestep.unsqueeze(1).repeat(1, latent_video_length)
                    if extra_latents is not None:
                        if empty_latent_indices is not None and len(empty_latent_indices) > 0:
                            # Set timestep to zero for all non-noise (non-empty) indices
                            non_noise_indices = [i for i in range(timestep.shape[1]) if i not in empty_latent_indices]
                            timestep[:, non_noise_indices] = 0
                        else:
                            timestep[:,0:encoded_image_latents.shape[1]] = 0 
                    #print(f"timestep: {timestep}")
                current_step_percentage = idx / len(timesteps)

                ### latent shift
                if latent_shift_loop:
                    if latent_shift_start_percent <= current_step_percentage <= latent_shift_end_percent:
                        latent_model_input = torch.cat([latent_model_input[:, shift_idx:]] + [latent_model_input[:, :shift_idx]], dim=1)

                #enhance-a-video
                enhance_enabled = False
                if feta_args is not None and feta_start_percent <= current_step_percentage <= feta_end_percent:
                    enhance_enabled = True                    

                #flow-edit
                if flowedit_args is not None:
                    sigma = t / 1000.0
                    sigma_prev = (timesteps[idx + 1] if idx < len(timesteps) - 1 else timesteps[-1]) / 1000.0
                    noise = torch.randn(x_init.shape, generator=seed_g, device=torch.device("cpu"))
                    if idx < len(timesteps) - drift_steps:
                        cfg = drift_cfg
                    
                    zt_src = (1-sigma) * x_init + sigma * noise.to(t)
                    zt_tgt = x_tgt + zt_src - x_init

                    #source
                    if idx < len(timesteps) - drift_steps:
                        if context_options is not None:
                            counter = torch.zeros_like(zt_src, device=intermediate_device)
                            vt_src = torch.zeros_like(zt_src, device=intermediate_device)
                            context_queue = list(context(idx, steps, latent_video_length, context_frames, context_stride, context_overlap))
                            for c in context_queue:
                                window_id = self.window_tracker.get_window_id(c)

                                if cache_args is not None:
                                    current_teacache = self.window_tracker.get_teacache(window_id, self.cache_state)
                                else:
                                    current_teacache = None

                                prompt_index = min(int(max(c) / section_size), num_prompts - 1)
                                if context_options["verbose"]:
                                    log.info(f"Prompt index: {prompt_index}")

                                if len(source_embeds["prompt_embeds"]) > 1:
                                    positive = source_embeds["prompt_embeds"][prompt_index]
                                else:
                                    positive = source_embeds["prompt_embeds"]

                                partial_img_emb = None
                                if source_image_cond is not None:
                                    partial_img_emb = source_image_cond[:, c, :, :]
                                    partial_img_emb[:, 0, :, :] = source_image_cond[:, 0, :, :].to(intermediate_device)

                                partial_zt_src = zt_src[:, c, :, :]
                                vt_src_context, new_teacache = predict_with_cfg(
                                    partial_zt_src, cfg[idx], 
                                    positive, source_embeds["negative_prompt_embeds"],
                                    timestep, idx, partial_img_emb, control_latents,
                                    source_clip_fea, current_teacache)
                                
                                if cache_args is not None:
                                    self.window_tracker.cache_states[window_id] = new_teacache

                                window_mask = create_window_mask(vt_src_context, c, latent_video_length, context_overlap)
                                vt_src[:, c, :, :] += vt_src_context * window_mask
                                counter[:, c, :, :] += window_mask
                            vt_src /= counter
                        else:
                            vt_src, self.cache_state_source = predict_with_cfg(
                                zt_src, cfg[idx], 
                                source_embeds["prompt_embeds"], 
                                source_embeds["negative_prompt_embeds"],
                                timestep, idx, source_image_cond, 
                                source_clip_fea, control_latents,
                                cache_state=self.cache_state_source)
                    else:
                        if idx == len(timesteps) - drift_steps:
                            x_tgt = zt_tgt
                        zt_tgt = x_tgt
                        vt_src = 0
                    #target
                    if context_options is not None:
                        counter = torch.zeros_like(zt_tgt, device=intermediate_device)
                        vt_tgt = torch.zeros_like(zt_tgt, device=intermediate_device)
                        context_queue = list(context(idx, steps, latent_video_length, context_frames, context_stride, context_overlap))
                        for c in context_queue:
                            window_id = self.window_tracker.get_window_id(c)

                            if cache_args is not None:
                                current_teacache = self.window_tracker.get_teacache(window_id, self.cache_state)
                            else:
                                current_teacache = None

                            prompt_index = min(int(max(c) / section_size), num_prompts - 1)
                            if context_options["verbose"]:
                                log.info(f"Prompt index: {prompt_index}")
                        
                            if len(text_embeds["prompt_embeds"]) > 1:
                                positive = text_embeds["prompt_embeds"][prompt_index]
                            else:
                                positive = text_embeds["prompt_embeds"]
                            
                            partial_img_emb = None
                            partial_control_latents = None
                            if image_cond is not None:
                                partial_img_emb = image_cond[:, c, :, :]
                                partial_img_emb[:, 0, :, :] = image_cond[:, 0, :, :].to(intermediate_device)
                            if control_latents is not None:
                                partial_control_latents = control_latents[:, c, :, :]

                            partial_zt_tgt = zt_tgt[:, c, :, :]
                            vt_tgt_context, new_teacache = predict_with_cfg(
                                partial_zt_tgt, cfg[idx], 
                                positive, text_embeds["negative_prompt_embeds"],
                                timestep, idx, partial_img_emb, partial_control_latents,
                                clip_fea, current_teacache)
                            
                            if cache_args is not None:
                                self.window_tracker.cache_states[window_id] = new_teacache
                            
                            window_mask = create_window_mask(vt_tgt_context, c, latent_video_length, context_overlap)
                            vt_tgt[:, c, :, :] += vt_tgt_context * window_mask
                            counter[:, c, :, :] += window_mask
                        vt_tgt /= counter
                    else:
                        vt_tgt, self.cache_state = predict_with_cfg(
                            zt_tgt, cfg[idx], 
                            text_embeds["prompt_embeds"], 
                            text_embeds["negative_prompt_embeds"], 
                            timestep, idx, image_cond, clip_fea, control_latents,
                            cache_state=self.cache_state)
                    v_delta = vt_tgt - vt_src
                    x_tgt = x_tgt.to(torch.float32)
                    v_delta = v_delta.to(torch.float32)
                    x_tgt = x_tgt + (sigma_prev - sigma) * v_delta
                    x0 = x_tgt
                #region context windowing
                elif context_options is not None:
                    counter = torch.zeros_like(latent_model_input, device=intermediate_device)
                    noise_pred = torch.zeros_like(latent_model_input, device=intermediate_device)
                    context_queue = list(context(idx, steps, latent_video_length, context_frames, context_stride, context_overlap))
                    fraction_per_context = 1.0 / len(context_queue)
                    context_pbar = ProgressBar(steps)
                    step_start_progress = idx

                    for i, c in enumerate(context_queue):
                        window_id = self.window_tracker.get_window_id(c)
                        
                        if cache_args is not None:
                            current_teacache = self.window_tracker.get_teacache(window_id, self.cache_state)
                        else:
                            current_teacache = None

                        prompt_index = min(int(max(c) / section_size), num_prompts - 1)
                        if context_options["verbose"]:
                            log.info(f"Prompt index: {prompt_index}")
                        
                        # Use the appropriate prompt for this section
                        if len(text_embeds["prompt_embeds"]) > 1:
                            positive = text_embeds["prompt_embeds"][prompt_index]
                        else:
                            positive = text_embeds["prompt_embeds"]

                        partial_img_emb = None
                        partial_control_latents = None
                        if image_cond is not None:
                            partial_img_emb = image_cond[:, c]
                            partial_img_emb[:, 0] = image_cond[:, 0].to(intermediate_device)

                            if control_latents is not None:
                                partial_control_latents = control_latents[:, c]
                        
                        partial_control_camera_latents = None
                        if control_camera_latents is not None:
                            partial_control_camera_latents = control_camera_latents[:, :, c]
                        
                        partial_vace_context = None
                        if vace_data is not None:
                            window_vace_data = []
                            for vace_entry in vace_data:
                                partial_context = vace_entry["context"][0][:, c]
                                if has_ref:
                                    partial_context[:, 0] = vace_entry["context"][0][:, 0]
                                
                                window_vace_data.append({
                                    "context": [partial_context], 
                                    "scale": vace_entry["scale"],
                                    "start": vace_entry["start"], 
                                    "end": vace_entry["end"],
                                    "seq_len": vace_entry["seq_len"]
                                })
                            
                            partial_vace_context = window_vace_data

                        partial_audio_proj = None
                        if fantasytalking_embeds is not None:
                            partial_audio_proj = audio_proj[:, c]

                        partial_latent_model_input = latent_model_input[:, c]

                        partial_unianim_data = None
                        if unianim_data is not None:
                            partial_dwpose = dwpose_data[:, :, c]
                            partial_dwpose_flat=rearrange(partial_dwpose, 'b c f h w -> b (f h w) c')
                            partial_unianim_data = {
                                "dwpose": partial_dwpose_flat,
                                "random_ref": unianim_data["random_ref"],
                                "strength": unianimate_poses["strength"],
                                "start_percent": unianimate_poses["start_percent"],
                                "end_percent": unianimate_poses["end_percent"]
                            }
                            
                        partial_add_cond = None
                        if add_cond is not None:
                            partial_add_cond = add_cond[:, :, c].to(device, dtype)

                        noise_pred_context, new_teacache = predict_with_cfg(
                            partial_latent_model_input, 
                            cfg[idx], positive, 
                            text_embeds["negative_prompt_embeds"], 
                            timestep, idx, partial_img_emb, clip_fea, partial_control_latents, partial_vace_context, partial_unianim_data,partial_audio_proj,
                            partial_control_camera_latents, partial_add_cond, current_teacache, context_window=c)

                        if cache_args is not None:
                            self.window_tracker.cache_states[window_id] = new_teacache

                        window_mask = create_window_mask(noise_pred_context, c, latent_video_length, context_overlap, looped=is_looped, window_type=context_options["fuse_method"])                    
                        noise_pred[:, c] += noise_pred_context * window_mask
                        counter[:, c] += window_mask
                        context_pbar.update_absolute(step_start_progress + (i + 1) * fraction_per_context, steps)
                    noise_pred /= counter
                #region multitalk
                elif multitalk_embeds is not None:
                    original_image = cond_image = image_embeds.get("multitalk_start_image", None)
                    offload = image_embeds.get("force_offload", False)
                    tiled_vae = image_embeds.get("tiled_vae", False)
                    frame_num = clip_length = image_embeds.get("num_frames", 81)
                    vae = image_embeds.get("vae", None)
                    clip_embeds = image_embeds.get("clip_context", None)
                    colormatch = image_embeds.get("colormatch", "disabled")
                    motion_frame = image_embeds.get("motion_frame", 25)
                    target_w = image_embeds.get("target_w", None)
                    target_h = image_embeds.get("target_h", None)

                    gen_video_list = []
                    is_first_clip = True
                    arrive_last_frame = False
                    cur_motion_frames_num = 1
                    audio_start_idx = iteration_count = 0
                    audio_end_idx = audio_start_idx + clip_length
                    indices = (torch.arange(4 + 1) - 2) * 1
                    
                    if multitalk_embeds is not None:
                        total_frames = len(multitalk_audio_embedding[0])

                    estimated_iterations = total_frames // (frame_num - motion_frame) + 1
                    loop_pbar = tqdm(total=estimated_iterations, desc="Generating video clips")
                    callback = prepare_callback(patcher, estimated_iterations)

                    audio_embedding = multitalk_audio_embedding
                    human_num = len(audio_embedding)
                    audio_embs = None
                    while True: # start video generation iteratively
                        if multitalk_embeds is not None:
                            audio_embs = []
                            # split audio with window size
                            for human_idx in range(human_num):   
                                center_indices = torch.arange(audio_start_idx, audio_end_idx, 1).unsqueeze(1) + indices.unsqueeze(0)
                                center_indices = torch.clamp(center_indices, min=0, max=audio_embedding[human_idx].shape[0]-1)
                                audio_emb = audio_embedding[human_idx][center_indices].unsqueeze(0).to(device)
                                audio_embs.append(audio_emb)
                            audio_embs = torch.concat(audio_embs, dim=0).to(dtype)

                        h, w = cond_image.shape[-2], cond_image.shape[-1]
                        lat_h, lat_w = h // VAE_STRIDE[1], w // VAE_STRIDE[2]
                        seq_len = ((frame_num - 1) // VAE_STRIDE[0] + 1) * lat_h * lat_w // (PATCH_SIZE[1] * PATCH_SIZE[2])

                        noise = torch.randn(
                            16, (frame_num - 1) // 4 + 1,
                            lat_h, lat_w, dtype=torch.float32, device=torch.device("cpu"), generator=seed_g).to(device)

                        # get mask
                        msk = torch.ones(1, frame_num, lat_h, lat_w, device=device)
                        msk[:, cur_motion_frames_num:] = 0
                        msk = torch.concat([
                            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
                        ], dim=1)
                        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
                        msk = msk.transpose(1, 2).to(dtype) # B 4 T H W

                        mm.soft_empty_cache()

                        # zero padding and vae encode
                        video_frames = torch.zeros(1, cond_image.shape[1], frame_num-cond_image.shape[2], target_h, target_w, device=device, dtype=vae.dtype)
                        padding_frames_pixels_values = torch.concat([cond_image.to(device, vae.dtype), video_frames], dim=2)

                        vae.to(device)
                        y = vae.encode(padding_frames_pixels_values, device=device, tiled=tiled_vae).to(dtype)
                        vae.to(offload_device)

                        cur_motion_frames_latent_num = int(1 + (cur_motion_frames_num-1) // 4)
                        latent_motion_frames = y[:, :, :cur_motion_frames_latent_num][0] # C T H W
                        y = torch.concat([msk, y], dim=1) # B 4+C T H W
                        mm.soft_empty_cache()

                        if scheduler == "multitalk":
                            timesteps = list(np.linspace(1000, 1, steps, dtype=np.float32))
                            timesteps.append(0.)
                            timesteps = [torch.tensor([t], device=device) for t in timesteps]
                            timesteps = [timestep_transform(t, shift=shift, num_timesteps=1000) for t in timesteps]
                        else:
                            sample_scheduler, timesteps = get_scheduler(scheduler, steps, shift, device, transformer.dim, flowedit_args, denoise_strength, sigmas=sigmas)

                            transformed_timesteps = []
                            for t in timesteps:
                                t_tensor = torch.tensor([t.item()], device=device)
                                transformed_timesteps.append(t_tensor)

                            transformed_timesteps.append(torch.tensor([0.], device=device))
                            timesteps = transformed_timesteps
                        
                        # sample videos
                        latent = noise

                        # injecting motion frames
                        if not is_first_clip:
                            latent_motion_frames = latent_motion_frames.to(latent.dtype).to(device)
                            motion_add_noise = torch.randn(latent_motion_frames.shape, device=torch.device("cpu"), generator=seed_g).to(device).contiguous()
                            add_latent = add_noise(latent_motion_frames, motion_add_noise, timesteps[0])
                            _, T_m, _, _ = add_latent.shape
                            latent[:, :T_m] = add_latent

                        if offload:
                            #blockswap init
                            if transformer_options is not None:
                                block_swap_args = transformer_options.get("block_swap_args", None)

                            if block_swap_args is not None:
                                transformer.use_non_blocking = block_swap_args.get("use_non_blocking", True)
                                for name, param in transformer.named_parameters():
                                    if "block" not in name:
                                        param.data = param.data.to(device)
                                    if "control_adapter" in name:
                                        param.data = param.data.to(device)
                                    elif block_swap_args["offload_txt_emb"] and "txt_emb" in name:
                                        param.data = param.data.to(offload_device, non_blocking=transformer.use_non_blocking)
                                    elif block_swap_args["offload_img_emb"] and "img_emb" in name:
                                        param.data = param.data.to(offload_device, non_blocking=transformer.use_non_blocking)

                                transformer.block_swap(
                                    block_swap_args["blocks_to_swap"] - 1 ,
                                    block_swap_args["offload_txt_emb"],
                                    block_swap_args["offload_img_emb"],
                                    vace_blocks_to_swap = block_swap_args.get("vace_blocks_to_swap", None),
                                )

                            elif model["auto_cpu_offload"]:
                                for module in transformer.modules():
                                    if hasattr(module, "offload"):
                                        module.offload()
                                    if hasattr(module, "onload"):
                                        module.onload()
                            elif model["manual_offloading"]:
                                transformer.to(device)

                        comfy_pbar = ProgressBar(len(timesteps)-1)
                        for i in tqdm(range(len(timesteps)-1)):
                            timestep = timesteps[i]
                            latent_model_input = latent.to(device)

                            noise_pred, self.cache_state = predict_with_cfg(
                                latent_model_input, 
                                cfg[idx], 
                                text_embeds["prompt_embeds"], 
                                text_embeds["negative_prompt_embeds"], 
                                timestep, idx, y.squeeze(0), clip_embeds.to(dtype), control_latents, vace_data, unianim_data, audio_proj, control_camera_latents, add_cond,
                                cache_state=self.cache_state, multitalk_audio_embeds=audio_embs)

                            if callback is not None:
                                callback_latent = (latent_model_input.to(device) - noise_pred.to(device) * t.to(device) / 1000).detach().permute(1,0,2,3)
                                callback(iteration_count, callback_latent, None, estimated_iterations)

                            # update latent
                            if scheduler == "multitalk":
                                noise_pred = -noise_pred
                                dt = timesteps[i] - timesteps[i + 1]
                                dt = dt / 1000
                                latent = latent + noise_pred * dt[:, None, None, None]
                            else:
                                latent = latent.to(intermediate_device)

                                temp_x0 = sample_scheduler.step(
                                    noise_pred.unsqueeze(0),
                                    timestep,
                                    latent.unsqueeze(0),
                                    **scheduler_step_args)[0]
                                latent = temp_x0.squeeze(0)

                            # injecting motion frames
                            if not is_first_clip:
                                latent_motion_frames = latent_motion_frames.to(latent.dtype).to(device)
                                motion_add_noise = torch.randn(latent_motion_frames.shape, device=torch.device("cpu"), generator=seed_g).to(device).contiguous()
                                add_latent = add_noise(latent_motion_frames, motion_add_noise, timesteps[i+1])
                                _, T_m, _, _ = add_latent.shape
                                latent[:, :T_m] = add_latent

                            x0 = latent.to(device)
                            del latent_model_input, timestep
                            comfy_pbar.update(1)

                        if offload:
                            transformer.to(offload_device)
                        vae.to(device)
                        videos = vae.decode(x0.unsqueeze(0).to(vae.dtype), device=device, tiled=tiled_vae)
                        vae.to(offload_device)
                        
                        # cache generated samples
                        videos = torch.stack(videos).cpu() # B C T H W
                        if colormatch != "disabled":
                            videos = videos[0].permute(1, 2, 3, 0).cpu().numpy()
                            from color_matcher import ColorMatcher
                            cm = ColorMatcher()
                            cm_result_list = []
                            for img in videos:
                                cm_result = cm.transfer(src=img, ref=original_image[0].permute(1, 2, 3, 0).squeeze(0).cpu().numpy(), method=colormatch)
                                cm_result_list.append(torch.from_numpy(cm_result))
                    
                            videos = torch.stack(cm_result_list, dim=0).to(torch.float32).permute(3, 0, 1, 2).unsqueeze(0)

                        if is_first_clip:
                            gen_video_list.append(videos)
                        else:
                            gen_video_list.append(videos[:, :, cur_motion_frames_num:])

                        # decide whether is done
                        if arrive_last_frame: 
                            loop_pbar.update(estimated_iterations - iteration_count)
                            loop_pbar.close()
                            break

                        # update next condition frames
                        is_first_clip = False
                        cur_motion_frames_num = motion_frame

                        cond_image = videos[:, :, -cur_motion_frames_num:].to(torch.float32).to(device)

                        # Update progress bar
                        iteration_count += 1
                        loop_pbar.update(1)

                        # Repeat audio emb
                        if multitalk_embeds is not None:
                            audio_start_idx += (frame_num - cur_motion_frames_num)
                            audio_end_idx = audio_start_idx + clip_length
                            if audio_end_idx >= len(audio_embedding[0]):
                                arrive_last_frame = True
                                miss_lengths = []
                                source_frames = []
                                for human_inx in range(human_num):
                                    source_frame = len(audio_embedding[human_inx])
                                    source_frames.append(source_frame)
                                    if audio_end_idx >= len(audio_embedding[human_inx]):
                                        miss_length   = audio_end_idx - len(audio_embedding[human_inx]) + 3 
                                        add_audio_emb = torch.flip(audio_embedding[human_inx][-1*miss_length:], dims=[0])
                                        audio_embedding[human_inx] = torch.cat([audio_embedding[human_inx], add_audio_emb], dim=0)
                                        miss_lengths.append(miss_length)
                                    else:
                                        miss_lengths.append(0)
                    
                    gen_video_samples = torch.cat(gen_video_list, dim=2).to(torch.float32)
                    gen_video_samples = gen_video_samples[:, :, :total_frames]
                    
                    del noise, latent
                    if force_offload:
                        if model["manual_offloading"]:
                            transformer.to(offload_device)
                            mm.soft_empty_cache()
                            gc.collect()
                    try:
                        print_memory(device)
                        torch.cuda.reset_peak_memory_stats(device)
                    except:
                        pass
                    return {"video": gen_video_samples[0].permute(1, 2, 3, 0).cpu()},
                elif infinitetalk_embeds is not None:
                    # original_image = cond_image = image_embeds.get("multitalk_start_image", None)
                    original_video = image_embeds.get("multitalk_start_image", None)
                    offload = image_embeds.get("force_offload", False)
                    tiled_vae = image_embeds.get("tiled_vae", False)
                    frame_num = clip_length = image_embeds.get("num_frames", 81)
                    vae = image_embeds.get("vae", None)
                    clip_embeds = image_embeds.get("clip_context", None)
                    colormatch = image_embeds.get("colormatch", "disabled")
                    motion_frame = image_embeds.get("motion_frame", 25)
                    target_w = image_embeds.get("target_w", None)
                    target_h = image_embeds.get("target_h", None)

                    if len(infinitetalk_embeds['audio_features'])==2 and (infinitetalk_embeds['ref_target_masks'] is None):
                        face_scale = 0.1
                        x_min, x_max = int(target_h * face_scale), int(target_h * (1 - face_scale))
                        background_mask = torch.zeros([target_h, target_w])
                        background_mask = torch.zeros([target_h, target_w])
                        human_mask1 = torch.zeros([target_h, target_w])
                        human_mask2 = torch.zeros([target_h, target_w])
                        lefty_min, lefty_max = int((target_w//2) * face_scale), int((target_w//2) * (1 - face_scale))
                        righty_min, righty_max = int((target_w//2) * face_scale + (target_w//2)), int((target_w//2) * (1 - face_scale) + (target_w//2))
                        human_mask1[x_min:x_max, lefty_min:lefty_max] = 1
                        human_mask2[x_min:x_max, righty_min:righty_max] = 1
                        background_mask += human_mask1
                        background_mask += human_mask2
                        human_masks = [human_mask1, human_mask2]
                        background_mask = torch.where(background_mask > 0, torch.tensor(0), torch.tensor(1))
                        human_masks.append(background_mask)
                        ref_target_masks = torch.stack(human_masks, dim=0)
                        infinitetalk_embeds['ref_target_masks'] = ref_target_masks

                    gen_video_list = []
                    is_first_clip = True
                    arrive_last_frame = False
                    cur_motion_frames_num = 1
                    audio_start_idx = iteration_count = 0
                    audio_end_idx = audio_start_idx + clip_length
                    indices = (torch.arange(4 + 1) - 2) * 1
                    current_condframe_index = 0
                    
                    if infinitetalk_embeds is not None:
                        total_frames = len(multitalk_audio_embedding[0])
                    
                    pcd_data = pcd_data_input = None
                    if uni3c_embeds is not None:
                        transformer.controlnet = uni3c_embeds["controlnet"]
                        pcd_data = {
                            "render_latent": uni3c_embeds["render_latent"].to(dtype),
                            "render_mask": uni3c_embeds["render_mask"],
                            "camera_embedding": uni3c_embeds["camera_embedding"],
                            "controlnet_weight": uni3c_embeds["controlnet_weight"],
                            "start": uni3c_embeds["start"],
                            "end": uni3c_embeds["end"],
                        }

                    estimated_iterations = total_frames // (frame_num - motion_frame) + 1
                    loop_pbar = tqdm(total=estimated_iterations, desc="Generating video clips")
                    callback = prepare_callback(patcher, estimated_iterations)

                    audio_embedding = multitalk_audio_embedding
                    human_num = len(audio_embedding)
                    audio_embs = None
                    while True: # start video generation iteratively
                        # cond_image = original_video[:, :, audio_start_idx:audio_start_idx+1]
                        cond_image = original_video[:, :, current_condframe_index:current_condframe_index+1]
                        print("current_condframe_index:" + str(current_condframe_index))
                        print("audio_start_idx:" + str(audio_start_idx))
                        if infinitetalk_embeds is not None:
                            audio_embs = []
                            # split audio with window size
                            for human_idx in range(human_num):   
                                center_indices = torch.arange(audio_start_idx, audio_end_idx, 1).unsqueeze(1) + indices.unsqueeze(0)
                                center_indices = torch.clamp(center_indices, min=0, max=audio_embedding[human_idx].shape[0]-1)
                                audio_emb = audio_embedding[human_idx][center_indices].unsqueeze(0).to(device)
                                audio_embs.append(audio_emb)
                            audio_embs = torch.concat(audio_embs, dim=0).to(dtype)
                        
                        if uni3c_embeds is not None:
                            vae.to(device)
                            render_latent = vae.encode(original_video[:, :, audio_start_idx:audio_end_idx].to(device, vae.dtype), device=device, tiled=tiled_vae).to(dtype)
                            pcd_data['render_latent'] = render_latent

                        

                        h, w = cond_image.shape[-2], cond_image.shape[-1]
                        lat_h, lat_w = h // VAE_STRIDE[1], w // VAE_STRIDE[2]
                        seq_len = ((frame_num - 1) // VAE_STRIDE[0] + 1) * lat_h * lat_w // (PATCH_SIZE[1] * PATCH_SIZE[2])

                        noise = torch.randn(
                            16, (frame_num - 1) // 4 + 1,
                            lat_h, lat_w, dtype=torch.float32, device=torch.device("cpu"), generator=seed_g).to(device)

                        # get mask
                        msk = torch.ones(1, frame_num, lat_h, lat_w, device=device)
                        msk[:, 1:] = 0
                        msk = torch.concat([
                            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
                        ], dim=1)
                        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
                        msk = msk.transpose(1, 2).to(dtype) # B 4 T H W

                        mm.soft_empty_cache()

                        # zero padding and vae encode
                        video_frames = torch.zeros(1, cond_image.shape[1], frame_num-cond_image.shape[2], target_h, target_w, device=device, dtype=vae.dtype)
                        padding_frames_pixels_values = torch.concat([cond_image.to(device, vae.dtype), video_frames], dim=2)

                        vae.to(device)
                        y = vae.encode(padding_frames_pixels_values, device=device, tiled=tiled_vae).to(dtype)

                        cur_motion_frames_latent_num = int(1 + (cur_motion_frames_num-1) // 4)
                        if is_first_clip:
                            latent_motion_frames = vae.encode(cond_image.to(device, vae.dtype), device=device, tiled=tiled_vae).to(dtype)
                        else:
                            latent_motion_frames = vae.encode(cond_frame.to(device, vae.dtype), device=device, tiled=tiled_vae).to(dtype)
                        latent_motion_frames = latent_motion_frames[0]
                            # latent_motion_frames = y[:, :, :cur_motion_frames_latent_num][0] # C T H W
                        vae.to(offload_device)
                        y = torch.concat([msk, y], dim=1) # B 4+C T H W
                        mm.soft_empty_cache()

                        if scheduler == "multitalk":
                            timesteps = list(np.linspace(1000, 1, steps, dtype=np.float32))
                            timesteps.append(0.)
                            timesteps = [torch.tensor([t], device=device) for t in timesteps]
                            timesteps = [timestep_transform(t, shift=shift, num_timesteps=1000) for t in timesteps]
                        else:
                            sample_scheduler, timesteps = get_scheduler(scheduler, steps, shift, device, transformer.dim, flowedit_args, denoise_strength, sigmas=sigmas)

                            transformed_timesteps = []
                            for t in timesteps:
                                t_tensor = torch.tensor([t.item()], device=device)
                                transformed_timesteps.append(t_tensor)

                            transformed_timesteps.append(torch.tensor([0.], device=device))
                            timesteps = transformed_timesteps
                        
                        # sample videos
                        latent = noise

                        # injecting motion frames
                        # if not is_first_clip:
                        #     latent_motion_frames = latent_motion_frames.to(latent.dtype).to(device)
                        #     motion_add_noise = torch.randn(latent_motion_frames.shape, device=torch.device("cpu"), generator=seed_g).to(device).contiguous()
                        #     add_latent = add_noise(latent_motion_frames, motion_add_noise, timesteps[0])
                        #     _, T_m, _, _ = add_latent.shape
                        #     latent[:, :T_m] = add_latent

                        if offload:
                            #blockswap init
                            if transformer_options is not None:
                                block_swap_args = transformer_options.get("block_swap_args", None)

                            if block_swap_args is not None:
                                transformer.use_non_blocking = block_swap_args.get("use_non_blocking", True)
                                for name, param in transformer.named_parameters():
                                    if "block" not in name:
                                        param.data = param.data.to(device)
                                    if "control_adapter" in name:
                                        param.data = param.data.to(device)
                                    elif block_swap_args["offload_txt_emb"] and "txt_emb" in name:
                                        param.data = param.data.to(offload_device, non_blocking=transformer.use_non_blocking)
                                    elif block_swap_args["offload_img_emb"] and "img_emb" in name:
                                        param.data = param.data.to(offload_device, non_blocking=transformer.use_non_blocking)

                                transformer.block_swap(
                                    block_swap_args["blocks_to_swap"] - 1 ,
                                    block_swap_args["offload_txt_emb"],
                                    block_swap_args["offload_img_emb"],
                                    vace_blocks_to_swap = block_swap_args.get("vace_blocks_to_swap", None),
                                )

                            elif model["auto_cpu_offload"]:
                                for module in transformer.modules():
                                    if hasattr(module, "offload"):
                                        module.offload()
                                    if hasattr(module, "onload"):
                                        module.onload()
                            elif model["manual_offloading"]:
                                transformer.to(device)

                        comfy_pbar = ProgressBar(len(timesteps)-1)
                        for i in tqdm(range(len(timesteps)-1)):
                            timestep = timesteps[i]
                            latent_model_input = latent.to(device)
                            latent_model_input[:, :cur_motion_frames_latent_num] = latent_motion_frames
                            
                            noise_pred, self.cache_state = predict_with_cfg(
                                latent_model_input, 
                                cfg[idx], 
                                text_embeds["prompt_embeds"], 
                                text_embeds["negative_prompt_embeds"], 
                                timestep, idx, y.squeeze(0), clip_embeds.to(dtype), control_latents, vace_data, unianim_data, audio_proj, control_camera_latents, add_cond,
                                cache_state=self.cache_state, multitalk_audio_embeds=audio_embs)

                            if callback is not None:
                                callback_latent = (latent_model_input.to(device) - noise_pred.to(device) * t.to(device) / 1000).detach().permute(1,0,2,3)
                                callback(iteration_count, callback_latent, None, estimated_iterations)

                            # update latent
                            if scheduler == "multitalk":
                                noise_pred = -noise_pred
                                dt = timesteps[i] - timesteps[i + 1]
                                dt = dt / 1000
                                latent = latent + noise_pred * dt[:, None, None, None]
                            else:
                                latent = latent.to(intermediate_device)

                                temp_x0 = sample_scheduler.step(
                                    noise_pred.unsqueeze(0),
                                    timestep,
                                    latent.unsqueeze(0),
                                    **scheduler_step_args)[0]
                                latent = temp_x0.squeeze(0)

                            # injecting motion frames
                            # if not is_first_clip:
                            #     latent_motion_frames = latent_motion_frames.to(latent.dtype).to(device)
                            #     motion_add_noise = torch.randn(latent_motion_frames.shape, device=torch.device("cpu"), generator=seed_g).to(device).contiguous()
                            #     add_latent = add_noise(latent_motion_frames, motion_add_noise, timesteps[i+1])
                            #     _, T_m, _, _ = add_latent.shape
                            #     latent[:, :T_m] = add_latent
                            
                            latent[:, :cur_motion_frames_latent_num] = latent_motion_frames

                            x0 = latent.to(device)
                            del latent_model_input, timestep
                            comfy_pbar.update(1)

                        if offload:
                            transformer.to(offload_device)
                        vae.to(device)
                        videos = vae.decode(x0.unsqueeze(0).to(vae.dtype), device=device, tiled=tiled_vae)
                        vae.to(offload_device)
                        
                        # cache generated samples
                        videos = torch.stack(videos).cpu() # B C T H W
                        if colormatch != "disabled":
                            videos = videos[0].permute(1, 2, 3, 0).cpu().numpy()
                            from color_matcher import ColorMatcher
                            cm = ColorMatcher()
                            cm_result_list = []
                            for img in videos:
                                cm_result = cm.transfer(src=img, ref=cond_image[0].permute(1, 2, 3, 0).squeeze(0).cpu().numpy(), method=colormatch)
                                cm_result_list.append(torch.from_numpy(cm_result))
                    
                            videos = torch.stack(cm_result_list, dim=0).to(torch.float32).permute(3, 0, 1, 2).unsqueeze(0)

                        if is_first_clip:
                            gen_video_list.append(videos)
                        else:
                            gen_video_list.append(videos[:, :, cur_motion_frames_num:])
                        current_condframe_index += 1

                        # decide whether is done
                        if arrive_last_frame: 
                            loop_pbar.update(estimated_iterations - iteration_count)
                            loop_pbar.close()
                            break

                        # update next condition frames
                        is_first_clip = False
                        cur_motion_frames_num = motion_frame
                        
                        # cur_motion_frames_latent_num = int(1 + (cur_motion_frames_num-1) // 4)
                        # latent_motion_frames = x0[:, -cur_motion_frames_latent_num:]

                        cond_frame = videos[:, :, -cur_motion_frames_num:].to(torch.float32).to(device)

                        # Update progress bar
                        iteration_count += 1
                        loop_pbar.update(1)

                        # Repeat audio emb
                        if infinitetalk_embeds is not None:
                            audio_start_idx += (frame_num - cur_motion_frames_num)
                            audio_end_idx = audio_start_idx + clip_length
                            if audio_end_idx >= len(audio_embedding[0]):
                                arrive_last_frame = True
                                miss_lengths = []
                                source_frames = []
                                for human_inx in range(human_num):
                                    source_frame = len(audio_embedding[human_inx])
                                    source_frames.append(source_frame)
                                    if audio_end_idx >= len(audio_embedding[human_inx]):
                                        miss_length   = audio_end_idx - len(audio_embedding[human_inx]) + 3 
                                        add_audio_emb = torch.flip(audio_embedding[human_inx][-1*miss_length:], dims=[0])
                                        audio_embedding[human_inx] = torch.cat([audio_embedding[human_inx], add_audio_emb], dim=0)
                                        miss_lengths.append(miss_length)
                                    else:
                                        miss_lengths.append(0)
                            if current_condframe_index >= original_video.shape[2]:
                                last_frame = original_video[:, :, -1:, :, :]
                                miss_length   = 1
                                original_video = torch.cat([original_video, last_frame.repeat(1, 1, miss_length, 1, 1)], dim=2)
                    
                    gen_video_samples = torch.cat(gen_video_list, dim=2).to(torch.float32)
                    gen_video_samples = gen_video_samples[:, :, :total_frames]
                    
                    del noise, latent
                    if force_offload:
                        if model["manual_offloading"]:
                            transformer.to(offload_device)
                            mm.soft_empty_cache()
                            gc.collect()
                    try:
                        print_memory(device)
                        torch.cuda.reset_peak_memory_stats(device)
                    except:
                        pass
                    return {"video": gen_video_samples[0].permute(1, 2, 3, 0).cpu()},
                
                #region normal inference
                else:
                    noise_pred, self.cache_state = predict_with_cfg(
                        latent_model_input, 
                        cfg[idx], 
                        text_embeds["prompt_embeds"], 
                        text_embeds["negative_prompt_embeds"], 
                        timestep, idx, image_cond, clip_fea, control_latents, vace_data, unianim_data, audio_proj, control_camera_latents, add_cond,
                        cache_state=self.cache_state)

                if latent_shift_loop:
                    #reverse latent shift
                    if latent_shift_start_percent <= current_step_percentage <= latent_shift_end_percent:
                        noise_pred = torch.cat([noise_pred[:, latent_video_length - shift_idx:]] + [noise_pred[:, :latent_video_length - shift_idx]], dim=1)
                        shift_idx = (shift_idx + latent_skip) % latent_video_length
                    
                
                if flowedit_args is None:
                    latent = latent.to(intermediate_device)
                    temp_x0 = sample_scheduler.step(
                        noise_pred[:, :orig_noise_len].unsqueeze(0) if recammaster is not None else noise_pred.unsqueeze(0),
                        timestep,
                        latent[:, :orig_noise_len].unsqueeze(0) if recammaster is not None else latent.unsqueeze(0),
                        **scheduler_step_args)[0]
                    latent = temp_x0.squeeze(0)

                    x0 = latent.to(device)
                    
                    if freeinit_args is not None:
                        current_latent = x0.clone()

                    if callback is not None:
                        if recammaster is not None:
                            callback_latent = (latent_model_input[:, :orig_noise_len].to(device) - noise_pred[:, :orig_noise_len].to(device) * t.to(device) / 1000).detach().permute(1,0,2,3)
                        elif phantom_latents is not None:
                            callback_latent = (latent_model_input[:,:-phantom_latents.shape[1]].to(device) - noise_pred[:,:-phantom_latents.shape[1]].to(device) * t.to(device) / 1000).detach().permute(1,0,2,3)
                        else:
                            callback_latent = (latent_model_input.to(device) - noise_pred.to(device) * t.to(device) / 1000).detach().permute(1,0,2,3)
                        callback(idx, callback_latent, None, steps)
                    else:
                        pbar.update(1)
                    del latent_model_input, timestep
                else:
                    if callback is not None:
                        callback_latent = (zt_tgt.to(device) - vt_tgt.to(device) * t.to(device) / 1000).detach().permute(1,0,2,3)
                        callback(idx, callback_latent, None, steps)
                    else:
                        pbar.update(1)

        if phantom_latents is not None:
            x0 = x0[:,:-phantom_latents.shape[1]]
                
        if cache_args is not None:
            cache_report(transformer, cache_args)

        if force_offload:
            if model["manual_offloading"]:
                transformer.to(offload_device)
                mm.soft_empty_cache()
                gc.collect()

        try:
            print_memory(device)
            #torch.cuda.memory._dump_snapshot("wanvideowrapper_memory_dump.pt")
            #torch.cuda.memory._record_memory_history(enabled=None)
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass

        return ({
            "samples": x0.unsqueeze(0).cpu(), 
            "looped": is_looped, 
            "end_image": end_image if not fun_or_fl2v_model else None, 
            "has_ref": has_ref, 
            "drop_last": drop_last,
            "generator_state": seed_g.get_state(),
        }, )

#region VideoDecode
class WanVideoDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("WANVAE",),
                    "samples": ("LATENT",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": False, "tooltip": (
                        "Drastically reduces memory use but will introduce seams at tile stride boundaries. "
                        "The location and number of seams is dictated by the tile stride size. "
                        "The visibility of seams can be controlled by increasing the tile size. "
                        "Seams become less obvious at 1.5x stride and are barely noticeable at 2x stride size. "
                        "Which is to say if you use a stride width of 160, the seams are barely noticeable with a tile width of 320."
                    )}),
                    "tile_x": ("INT", {"default": 272, "min": 40, "max": 2048, "step": 8, "tooltip": "Tile width in pixels. Smaller values use less VRAM but will make seams more obvious."}),
                    "tile_y": ("INT", {"default": 272, "min": 40, "max": 2048, "step": 8, "tooltip": "Tile height in pixels. Smaller values use less VRAM but will make seams more obvious."}),
                    "tile_stride_x": ("INT", {"default": 144, "min": 32, "max": 2040, "step": 8, "tooltip": "Tile stride width in pixels. Smaller values use less VRAM but will introduce more seams."}),
                    "tile_stride_y": ("INT", {"default": 128, "min": 32, "max": 2040, "step": 8, "tooltip": "Tile stride height in pixels. Smaller values use less VRAM but will introduce more seams."}),
                    },
                    "optional": {
                        "normalization": (["default", "minmax"], {"advanced": True}),
                    }
                }

    @classmethod
    def VALIDATE_INPUTS(s, tile_x, tile_y, tile_stride_x, tile_stride_y):
        if tile_x <= tile_stride_x:
            return "Tile width must be larger than the tile stride width."
        if tile_y <= tile_stride_y:
            return "Tile height must be larger than the tile stride height."
        return True

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "WanVideoWrapper"

    def decode(self, vae, samples, enable_vae_tiling, tile_x, tile_y, tile_stride_x, tile_stride_y, normalization="default"):
        mm.soft_empty_cache()
        video = samples.get("video", None)
        if video is not None:
            video = torch.clamp(video, -1.0, 1.0) 
            video = (video + 1.0) / 2.0
            return video.cpu(),
        latents = samples["samples"]
        end_image = samples.get("end_image", None)
        has_ref = samples.get("has_ref", False)
        drop_last = samples.get("drop_last", False)
        is_looped = samples.get("looped", False)

        vae.to(device)

        latents = latents.to(device = device, dtype = vae.dtype)

        mm.soft_empty_cache()

        if has_ref:
            latents = latents[:, :, 1:]
        if drop_last:
            latents = latents[:, :, :-1]

        #if is_looped:
        #   latents = torch.cat([latents[:, :, :warmup_latent_count],latents], dim=2)
        if type(vae).__name__ == "TAEHV":      
            images = vae.decode_video(latents.permute(0, 2, 1, 3, 4))[0].permute(1, 0, 2, 3)
            images = torch.clamp(images, 0.0, 1.0)
            images = images.permute(1, 2, 3, 0).cpu().float()
            return (images,)
        else:
            if end_image is not None:
                enable_vae_tiling = False
            images = vae.decode(latents, device=device, end_=(end_image is not None), tiled=enable_vae_tiling, tile_size=(tile_x//8, tile_y//8), tile_stride=(tile_stride_x//8, tile_stride_y//8))[0]
            vae.model.clear_cache()
        
        images = images.cpu()

        if normalization == "minmax":
            images = (images - images.min()) / (images.max() - images.min())
        else:  
            images = torch.clamp(images, -1.0, 1.0) 
            images = (images + 1.0) / 2.0
        
        if is_looped:
            #images = images[:, warmup_latent_count * 4:]
            temp_latents = torch.cat([latents[:, :, -3:]] + [latents[:, :, :2]], dim=2)
            temp_images = vae.decode(temp_latents, device=device, end_=(end_image is not None), tiled=enable_vae_tiling, tile_size=(tile_x//8, tile_y//8), tile_stride=(tile_stride_x//8, tile_stride_y//8))[0]
            temp_images = (temp_images - temp_images.min()) / (temp_images.max() - temp_images.min())
            images = torch.cat([temp_images[:, 9:].to(images), images[:, 5:]], dim=1)

        if end_image is not None: 
            #end_image = (end_image - end_image.min()) / (end_image.max() - end_image.min())
            #image[:, -1] = end_image[:, 0].to(image) #not sure about this
            images = images[:, 0:-1]

        vae.model.clear_cache()
        vae.to(offload_device)
        mm.soft_empty_cache()

        images = torch.clamp(images, 0.0, 1.0)
        images = images.permute(1, 2, 3, 0).float()

        return (images,)

#region VideoEncode
class WanVideoEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("WANVAE",),
                    "image": ("IMAGE",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": False, "tooltip": "Drastically reduces memory use but may introduce seams"}),
                    "tile_x": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_y": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_x": ("INT", {"default": 144, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_y": ("INT", {"default": 128, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride in pixels, smaller values use less VRAM, may introduce more seams"}),
                    },
                    "optional": {
                        "noise_aug_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Strength of noise augmentation, helpful for leapfusion I2V where some noise can add motion and give sharper results"}),
                        "latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional latent multiplier, helpful for leapfusion I2V where lower values allow for more motion"}),
                        "mask": ("MASK", ),
                    }
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    CATEGORY = "WanVideoWrapper"

    def encode(self, vae, image, enable_vae_tiling, tile_x, tile_y, tile_stride_x, tile_stride_y, noise_aug_strength=0.0, latent_strength=1.0, mask=None):
        vae.to(device)

        image = image.clone()

        B, H, W, C = image.shape
        if W % 16 != 0 or H % 16 != 0:
            new_height = (H // 16) * 16
            new_width = (W // 16) * 16
            log.warning(f"Image size {W}x{H} is not divisible by 16, resizing to {new_width}x{new_height}")
            image = common_upscale(image.movedim(-1, 1), new_width, new_height, "lanczos", "disabled").movedim(1, -1)

        image = image.to(vae.dtype).to(device).unsqueeze(0).permute(0, 4, 1, 2, 3) # B, C, T, H, W


        empty_frame_indices = []
        for i in range(image.shape[2]):
            if is_image_black(image[:, :, i]):
                empty_frame_indices.append(i)
        empty_frame_indices = []
        for i in range(image.shape[2]):
            if is_image_black(image[:, :, i]):
                empty_frame_indices.append(i)
        empty_latent_indices = []
        if empty_frame_indices:
            frames_per_latent = 4
            num_frames = image.shape[2]
            # Special mapping: latent 0 = [0], latent 1 = [1,2,3,4], latent 2 = [5,6,7,8], ...
            latent_frame_ranges = []
            latent_frame_ranges.append([0])
            for i in range(1, math.ceil((num_frames - 1) / frames_per_latent) + 1):
                start = 1 + (i - 1) * frames_per_latent
                end = min(start + frames_per_latent, num_frames)
                latent_frame_ranges.append(list(range(start, end)))
            for latent_idx, latent_frames in enumerate(latent_frame_ranges):
                print(f"latent {latent_idx}: frames {latent_frames}")
                if latent_frames and set(latent_frames).issubset(empty_frame_indices):
                    empty_latent_indices.append(latent_idx)
            if empty_latent_indices:
                log.info(f"Empty frames {empty_frame_indices} map to latents {empty_latent_indices}")
        

        if noise_aug_strength > 0.0:
            image = add_noise_to_reference_video(image, ratio=noise_aug_strength)

        if isinstance(vae, TAEHV):
            latents = vae.encode_video(image.permute(0, 2, 1, 3, 4), parallel=False)# B, T, C, H, W
            latents = latents.permute(0, 2, 1, 3, 4)
        else:
            latents = vae.encode(image * 2.0 - 1.0, device=device, tiled=enable_vae_tiling, tile_size=(tile_x//8, tile_y//8), tile_stride=(tile_stride_x//8, tile_stride_y//8))
            vae.model.clear_cache()
        if latent_strength != 1.0:
            latents *= latent_strength

        log.info(f"encoded latents shape {latents.shape}")
        latent_mask = None
        if mask is None:
            vae.to(offload_device)
        else:
            #latent_mask = mask.clone().to(vae.dtype).to(device) * 2.0 - 1.0
            #latent_mask = latent_mask.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1, 1)
            #latent_mask = vae.encode(latent_mask, device=device, tiled=enable_vae_tiling, tile_size=(tile_x, tile_y), tile_stride=(tile_stride_x, tile_stride_y))
            target_h, target_w = latents.shape[3:]

            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims [1,1,T,H,W]
                size=(latents.shape[2], target_h, target_w),
                mode='trilinear',
                align_corners=False
            ).squeeze(0)  # Remove batch dim, keep channel dim
            
            # Add batch & channel dims for final output
            latent_mask = mask.unsqueeze(0).repeat(1, latents.shape[1], 1, 1, 1)
            log.info(f"latent mask shape {latent_mask.shape}")
            vae.to(offload_device)
        mm.soft_empty_cache()
 
        return ({"samples": latents, "mask": latent_mask, "empty_latent_indices": empty_latent_indices},)

NODE_CLASS_MAPPINGS = {
    "WanVideoSampler": WanVideoSampler,
    "WanVideoDecode": WanVideoDecode,
    "WanVideoTextEncode": WanVideoTextEncode,
    "WanVideoTextEncodeSingle": WanVideoTextEncodeSingle,
    "WanVideoClipVisionEncode": WanVideoClipVisionEncode,
    "WanVideoImageToVideoEncode": WanVideoImageToVideoEncode,
    "WanVideoEncode": WanVideoEncode,
    "WanVideoEmptyEmbeds": WanVideoEmptyEmbeds,
    "WanVideoEnhanceAVideo": WanVideoEnhanceAVideo,
    "WanVideoContextOptions": WanVideoContextOptions,
    "WanVideoTextEmbedBridge": WanVideoTextEmbedBridge,
    "WanVideoFlowEdit": WanVideoFlowEdit,
    "WanVideoControlEmbeds": WanVideoControlEmbeds,
    "WanVideoSLG": WanVideoSLG,
    "WanVideoLoopArgs": WanVideoLoopArgs,
    "WanVideoSetBlockSwap": WanVideoSetBlockSwap,
    "WanVideoExperimentalArgs": WanVideoExperimentalArgs,
    "WanVideoVACEEncode": WanVideoVACEEncode,
    "WanVideoPhantomEmbeds": WanVideoPhantomEmbeds,
    "WanVideoRealisDanceLatents": WanVideoRealisDanceLatents,
    "WanVideoApplyNAG": WanVideoApplyNAG,
    "WanVideoMiniMaxRemoverEmbeds": WanVideoMiniMaxRemoverEmbeds,
    "WanVideoFreeInitArgs": WanVideoFreeInitArgs,
    "WanVideoSetRadialAttention": WanVideoSetRadialAttention,
    "WanVideoBlockList": WanVideoBlockList,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoSampler": "WanVideo Sampler",
    "WanVideoDecode": "WanVideo Decode",
    "WanVideoTextEncode": "WanVideo TextEncode",
    "WanVideoTextEncodeSingle": "WanVideo TextEncodeSingle",
    "WanVideoTextImageEncode": "WanVideo TextImageEncode (IP2V)",
    "WanVideoClipVisionEncode": "WanVideo ClipVision Encode",
    "WanVideoImageToVideoEncode": "WanVideo ImageToVideo Encode",
    "WanVideoEncode": "WanVideo Encode",
    "WanVideoEmptyEmbeds": "WanVideo Empty Embeds",
    "WanVideoEnhanceAVideo": "WanVideo Enhance-A-Video",
    "WanVideoContextOptions": "WanVideo Context Options",
    "WanVideoTextEmbedBridge": "WanVideo TextEmbed Bridge",
    "WanVideoFlowEdit": "WanVideo FlowEdit",
    "WanVideoControlEmbeds": "WanVideo Control Embeds",
    "WanVideoSLG": "WanVideo SLG",
    "WanVideoLoopArgs": "WanVideo Loop Args",
    "WanVideoSetBlockSwap": "WanVideo Set BlockSwap",
    "WanVideoExperimentalArgs": "WanVideo Experimental Args",
    "WanVideoVACEEncode": "WanVideo VACE Encode",
    "WanVideoPhantomEmbeds": "WanVideo Phantom Embeds",
    "WanVideoRealisDanceLatents": "WanVideo RealisDance Latents",
    "WanVideoApplyNAG": "WanVideo Apply NAG",
    "WanVideoMiniMaxRemoverEmbeds": "WanVideo MiniMax Remover Embeds",
    "WanVideoFreeInitArgs": "WanVideo Free Init Args",
    "WanVideoSetRadialAttention": "WanVideo Set Radial Attention",
    "WanVideoBlockList": "WanVideo Block List",
    }
