
import torch
from ..utils import log
import comfy.model_management as mm
from comfy.utils import ProgressBar, load_torch_file
from tqdm import tqdm
import gc

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
import folder_paths

import json
import numpy as np

class WanVideoUni3C_ControlnetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("controlnet"), {"tooltip": "These models are loaded from the 'ComfyUI/models/controlnet' -folder",}),

            "base_precision": (["fp32", "bf16", "fp16"], {"default": "fp16"}),
            "quantization": (['disabled', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_e5m2', 'fp8_e4m3fn_fast_no_ffn'], {"default": 'disabled', "tooltip": "optional quantization method"}),
            "load_device": (["main_device", "offload_device"], {"default": "main_device", "tooltip": "Initial device to load the model to, NOT recommended with the larger models unless you have 48GB+ VRAM"}),
            "attention_mode": ([
                    "sdpa",
                    "sageattn",
                    ], {"default": "sdpa"}),
            },
            "optional": {
                "compile_args": ("WANCOMPILEARGS", ),
                #"block_swap_args": ("BLOCKSWAPARGS", ),
            }
        }

    RETURN_TYPES = ("WANVIDEOCONTROLNET",)
    RETURN_NAMES = ("controlnet", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"

    def loadmodel(self, model, base_precision, load_device, quantization, attention_mode, compile_args=None):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        transformer_load_device = device if load_device == "main_device" else offload_device
        
        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[base_precision]
        

        model_path = folder_paths.get_full_path_or_raise("controlnet", model)
      
        sd = load_torch_file(model_path, device=transformer_load_device, safe_load=True)

        if not "controlnet_patch_embedding.weight" in sd:
            raise ValueError("Invalid ControlNet model")
        
        in_channels = sd["controlnet_patch_embedding.weight"].shape[1]
        ffn_dim = sd["controlnet_blocks.0.ffn.0.bias"].shape[0]

        controlnet_cfg = {
            "in_channels": in_channels,
            "conv_out_dim": 5120,
            "time_embed_dim": 5120,
            "dim": 1024,
            "ffn_dim": ffn_dim,
            "num_heads": 16,
            "num_layers": 20,
            "add_channels": 7,
            "mid_channels": 256,
            "attention_mode": attention_mode,
            "quantized": True if quantization != "disabled" else False,
            "base_dtype": base_dtype
        }

        from .controlnet import WanControlNet

        with init_empty_weights():
            controlnet = WanControlNet(controlnet_cfg)
        controlnet.eval()
        
        if quantization == "disabled":
            for k, v in sd.items():
                if isinstance(v, torch.Tensor):
                    if v.dtype == torch.float8_e4m3fn:
                        quantization = "fp8_e4m3fn"
                        break
                    elif v.dtype == torch.float8_e5m2:
                        quantization = "fp8_e5m2"
                        break

        if "fp8_e4m3fn" in quantization:
            dtype = torch.float8_e4m3fn
        elif quantization == "fp8_e5m2":
            dtype = torch.float8_e5m2
        else:
            dtype = base_dtype
        params_to_keep = {"norm", "head", "time_in", "vector_in", "controlnet_patch_embedding", "time_", "img_emb", "modulation", "text_embedding", "adapter", "proj_in"}
    
        log.info("Using accelerate to load and assign controlnet model weights to device...")
        param_count = sum(1 for _ in controlnet.named_parameters())
        for name, param in tqdm(controlnet.named_parameters(), 
                desc=f"Loading transformer parameters to {transformer_load_device}", 
                total=param_count,
                leave=True):
            dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype
            if "controlnet_patch_embedding" in name:
                dtype_to_use = torch.float32
            set_module_tensor_to_device(controlnet, name, device=transformer_load_device, dtype=dtype_to_use, value=sd[name])
        
        del sd

        if compile_args is not None:
            torch._dynamo.config.cache_size_limit = compile_args["dynamo_cache_size_limit"]
            try:
                if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'config'):
                    torch._dynamo.config.recompile_limit = compile_args["dynamo_recompile_limit"]
            except Exception as e:
                log.warning(f"Could not set recompile_limit: {e}")
            if compile_args["compile_transformer_blocks_only"]:
                for i, block in enumerate(controlnet.controlnet_blocks):
                    controlnet.controlnet_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
            else:
                controlnet = torch.compile(controlnet, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])        
        

        if load_device == "offload_device" and controlnet.device != offload_device:
            log.info(f"Moving controlnet model from {controlnet.device} to {offload_device}")
            controlnet.to(offload_device)
            gc.collect()
            mm.soft_empty_cache()

        return (controlnet,)

class WanVideoUni3C_embeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "controlnet": ("WANVIDEOCONTROLNET",),
            "render_latent": ("LATENT",),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the steps to apply the controlnet"}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the steps to apply the controlnet"}),
            },
            "optional": {
                "render_mask": ("MASK", {"tooltip": "NOT IMPLEMENTED!"}),
            },
        }

    RETURN_TYPES = ("UNI3C_EMBEDS", )
    RETURN_NAMES = ("uni3c_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, controlnet, render_latent, strength, start_percent, end_percent, render_mask=None):

        device = mm.get_torch_device()

        latent_mask = None

        latents = render_latent["samples"]
        nframe = latents.shape[2] * 4
        height = latents.shape[3] * 8
        width = latents.shape[4] * 8
        
        if render_mask is not None:
            raise NotImplementedError("render_mask is not implemented at this time")
            mask = torch.nn.functional.interpolate(
                    render_mask.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims [1,1,T,H,W]
                    size=(nframe, height, width),
                    mode='trilinear',
                    align_corners=False
                ).squeeze(0)
            latent_mask = mask.unsqueeze(0).to(device)
            log.info(f"latent mask shape {latent_mask.shape}")

        # # load camera
        # cam_info = json.load(open(f"{render_path}/cam_info.json"))
        # w2cs = torch.tensor(np.array(cam_info["extrinsic"]), dtype=torch.float32, device=device)
        # intrinsic = torch.tensor(np.array(cam_info["intrinsic"]), dtype=torch.float32, device=device)
        # intrinsic[0, :] = intrinsic[0, :] / cam_info["width"] * width
        # intrinsic[1, :] = intrinsic[1, :] / cam_info["height"] * height
        # intrinsic = intrinsic[None].repeat(nframe, 1, 1)

        # from .utils import build_cameras, set_initial_camera, traj_map

        # focal_length = 1.0
        # start_elevation = 5.0
        # depth_avg = 0.5
        # traj_type = "orbit"
        # cam_traj, x_offset, y_offset, z_offset, d_theta, d_phi, d_r = traj_map(traj_type)
        # focallength_px = focal_length * width

        # K = torch.tensor([[focallength_px, 0, width / 2],
        #                   [0, focallength_px, height / 2],
        #                   [0, 0, 1]], dtype=torch.float32)
        # K_inv = K.inverse()
        # intrinsic = K[None].repeat(nframe, 1, 1)

        
        # w2c_0, c2w_0 = set_initial_camera(start_elevation, depth_avg)
        # w2cs, c2ws, intrinsic = build_cameras(cam_traj=cam_traj,
        #                                     w2c_0=w2c_0,
        #                                     c2w_0=c2w_0,
        #                                     intrinsic=intrinsic,
        #                                     nframe=nframe,
        #                                     focal_length=focal_length,
        #                                     d_theta=d_theta,
        #                                     d_phi=d_phi,
        #                                     d_r=d_r,
        #                                     radius=depth_avg,
        #                                     x_offset=x_offset,
        #                                     y_offset=y_offset,
        #                                     z_offset=z_offset)

    
        # from .camera import get_camera_embedding
        # camera_embedding = get_camera_embedding(intrinsic, w2cs, nframe, height, width, normalize=True)
        #print("camera embedding shape", camera_embedding.shape)

        uni3c_embeds = {
            "controlnet": controlnet,
            "controlnet_weight": strength,
            "start": start_percent,
            "end": end_percent,
            "render_latent": latents.to(device),
            "render_mask": latent_mask,
            "camera_embedding": None
        }
    
        return (uni3c_embeds,)
    
NODE_CLASS_MAPPINGS = {
    "WanVideoUni3C_ControlnetLoader": WanVideoUni3C_ControlnetLoader,
    "WanVideoUni3C_embeds": WanVideoUni3C_embeds,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoUni3C_ControlnetLoader": "WanVideo Uni3C Controlnet Loader",
    "WanVideoUni3C_embeds": "WanVideo Uni3C Embeds",
    }

    