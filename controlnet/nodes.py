
import torch
from ..utils import log
import comfy.model_management as mm
from comfy.utils import load_torch_file
from tqdm import tqdm
import gc

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
import folder_paths

class WanVideoControlnetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("controlnet"), {"tooltip": "These models are loaded from the 'ComfyUI/models/controlnet' -folder",}),

            "base_precision": (["fp32", "bf16", "fp16"], {"default": "bf16"}),
            "quantization": (['disabled', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_e5m2', 'fp8_e4m3fn_fast_no_ffn'], {"default": 'disabled', "tooltip": "optional quantization method"}),
            "load_device": (["main_device", "offload_device"], {"default": "main_device", "tooltip": "Initial device to load the model to, NOT recommended with the larger models unless you have 48GB+ VRAM"}),
            },
        }

    RETURN_TYPES = ("WANVIDEOCONTROLNET",)
    RETURN_NAMES = ("controlnet", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Loads ControlNet model from 'https://huggingface.co/collections/TheDenk/wan21-controlnets-68302b430411dafc0d74d2fc'"

    def loadmodel(self, model, base_precision, load_device, quantization):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        transformer_load_device = device if load_device == "main_device" else offload_device
        
        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[base_precision]
        
        model_path = folder_paths.get_full_path_or_raise("controlnet", model)
      
        sd = load_torch_file(model_path, device=transformer_load_device, safe_load=True)

        num_layers = 8 if "blocks.7.scale_shift_table" in sd else 6
        out_proj_dim = 5120 if num_layers == 6 else 1536

        if not "control_encoder.0.0.weight" in sd:
            raise ValueError("Invalid ControlNet model")

        controlnet_cfg = {
            "added_kv_proj_dim": None,
            "attention_head_dim": 128,
            "cross_attn_norm": None,
            "downscale_coef": 8,
            "eps": 1e-06,
            "ffn_dim": 8960,
            "freq_dim": 256,
            "image_dim": None,
            "in_channels": 3,
            "num_attention_heads": 12,
            "num_layers": num_layers,
            "out_proj_dim": out_proj_dim,
            "patch_size": [
                1,
                2,
                2
            ],
            "qk_norm": "rms_norm_across_heads",
            "rope_max_seq_len": 1024,
            "text_dim": 4096,
            "vae_channels": 16
            }
        
        from .wan_controlnet import WanControlnet

        with init_empty_weights():
            controlnet = WanControlnet(**controlnet_cfg)
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
        params_to_keep = {"norm", "head", "time_in", "vector_in", "controlnet_patch_embedding", "time_", "img_emb", "modulation", "text_embedding", "adapter"}
    
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

        if load_device == "offload_device" and controlnet.device != offload_device:
            log.info(f"Moving controlnet model from {controlnet.device} to {offload_device}")
            controlnet.to(offload_device)
            gc.collect()
            mm.soft_empty_cache()

        return (controlnet,)
    
class WanVideoControlnetApply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL", ),
                "controlnet": ("WANVIDEOCONTROLNET", ),
                "control_images": ("IMAGE", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001, "tooltip": "controlnet strength"}),
                "control_stride": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1, "tooltip": "controlnet stride"}),
                "control_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent of the steps to apply controlnet"}),
                "control_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent of the steps to apply controlnet"}),
               }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"

    def loadmodel(self, model, controlnet, control_images, strength, control_stride, control_start_percent, control_end_percent):

        patcher = model.clone()
        if 'transformer_options' not in patcher.model_options:
            patcher.model_options['transformer_options'] = {}

        control_input = control_images.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
        control_input = control_input * 2.0 - 1.0
        
        controlnet = {
            "controlnet": controlnet,
            "control_latents": control_input,
            "controlnet_strength": strength,
            "control_stride": control_stride,
            "controlnet_start": control_start_percent,
            "controlnet_end": control_end_percent
        }
        patcher.model_options["transformer_options"]["controlnet"] = controlnet

        return (patcher,)
    
NODE_CLASS_MAPPINGS = {
    "WanVideoControlnetLoader": WanVideoControlnetLoader,
    "WanVideoControlnet": WanVideoControlnetApply,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoControlnetLoader": "WanVideo Controlnet Loader",
    "WanVideoControlnet": "WanVideo Controlnet Apply",
    }

    