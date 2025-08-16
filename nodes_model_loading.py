import torch
import os, gc
from .utils import log, apply_lora
import numpy as np
from tqdm import tqdm

from .wanvideo.modules.model import WanModel
from .wanvideo.modules.t5 import T5EncoderModel
from .wanvideo.modules.clip import CLIPModel

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar
import comfy.model_base
from comfy.sd import load_lora_for_models

script_directory = os.path.dirname(os.path.abspath(__file__))

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

try:
    from server import PromptServer
except:
    PromptServer = None

#from city96's gguf nodes
def update_folder_names_and_paths(key, targets=[]):
    # check for existing key
    base = folder_paths.folder_names_and_paths.get(key, ([], {}))
    base = base[0] if isinstance(base[0], (list, set, tuple)) else []
    # find base key & add w/ fallback, sanity check + warning
    target = next((x for x in targets if x in folder_paths.folder_names_and_paths), targets[0])
    orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
    folder_paths.folder_names_and_paths[key] = (orig or base, {".gguf"})
    if base and base != orig:
        log.warning(f"Unknown file list already present on key {key}: {base}")
update_folder_names_and_paths("unet_gguf", ["diffusion_models", "unet"])

class WanVideoModel(comfy.model_base.BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = {}

    def __getitem__(self, k):
        return self.pipeline[k]

    def __setitem__(self, k, v):
        self.pipeline[k] = v

try:
    from comfy.latent_formats import Wan21
    latent_format = Wan21
except: #for backwards compatibility
    log.warning("Wan21 latent format not found, update ComfyUI for better livepreview")
    from comfy.latent_formats import HunyuanVideo
    latent_format = HunyuanVideo

class WanVideoModelConfig:
    def __init__(self, dtype):
        self.unet_config = {}
        self.unet_extra_config = {}
        self.latent_format = latent_format
        self.latent_format.latent_channels = 16
        self.manual_cast_dtype = dtype
        self.sampling_settings = {"multiplier": 1.0}
        self.memory_usage_factor = 2.0
        self.unet_config["disable_unet_model_creation"] = True

def filter_state_dict_by_blocks(state_dict, blocks_mapping, layer_filter=[]):
    filtered_dict = {}

    if isinstance(layer_filter, str):
        layer_filters = [layer_filter] if layer_filter else []
    else:
        # Filter out empty strings
        layer_filters = [f for f in layer_filter if f] if layer_filter else []

    #print("layer_filter: ", layer_filters)

    for key in state_dict:
        if not any(filter_str in key for filter_str in layer_filters):
            if 'blocks.' in key:
                
                block_pattern = key.split('diffusion_model.')[1].split('.', 2)[0:2]
                block_key = f'{block_pattern[0]}.{block_pattern[1]}.'

                if block_key in blocks_mapping:
                    filtered_dict[key] = state_dict[key]
            else:
                filtered_dict[key] = state_dict[key]
    
    for key in filtered_dict:
        print(key)

    #from safetensors.torch import save_file
    #save_file(filtered_dict, "filtered_state_dict_2.safetensors")

    return filtered_dict

def standardize_lora_key_format(lora_sd):
    new_sd = {}
    for k, v in lora_sd.items():
        # Diffusers format
        if k.startswith('transformer.'):
            k = k.replace('transformer.', 'diffusion_model.')
        if k.startswith('pipe.dit.'): #unianimate-dit/diffsynth
            k = k.replace('pipe.dit.', 'diffusion_model.')

        # Fun LoRA format
        if k.startswith('lora_unet__'):
            # Split into main path and weight type parts
            parts = k.split('.')
            main_part = parts[0]  # e.g. lora_unet__blocks_0_cross_attn_k
            weight_type = '.'.join(parts[1:]) if len(parts) > 1 else None  # e.g. lora_down.weight
            
            # Process the main part - convert from underscore to dot format
            if 'blocks_' in main_part:
                # Extract components
                components = main_part[len('lora_unet__'):].split('_')
                
                # Start with diffusion_model
                new_key = "diffusion_model"
                
                # Add blocks.N
                if components[0] == 'blocks':
                    new_key += f".blocks.{components[1]}"
                    
                    # Handle different module types
                    idx = 2
                    if idx < len(components):
                        if components[idx] == 'self' and idx+1 < len(components) and components[idx+1] == 'attn':
                            new_key += ".self_attn"
                            idx += 2
                        elif components[idx] == 'cross' and idx+1 < len(components) and components[idx+1] == 'attn':
                            new_key += ".cross_attn"
                            idx += 2
                        elif components[idx] == 'ffn':
                            new_key += ".ffn"
                            idx += 1
                    
                    # Add the component (k, q, v, o) and handle img suffix
                    if idx < len(components):
                        component = components[idx]
                        idx += 1
                        
                        # Check for img suffix
                        if idx < len(components) and components[idx] == 'img':
                            component += '_img'
                            idx += 1
                            
                        new_key += f".{component}"
                
                # Handle weight type - this is the critical fix
                if weight_type:
                    if weight_type == 'alpha':
                        new_key += '.alpha'
                    elif weight_type == 'lora_down.weight' or weight_type == 'lora_down':
                        new_key += '.lora_A.weight'
                    elif weight_type == 'lora_up.weight' or weight_type == 'lora_up':
                        new_key += '.lora_B.weight'
                    else:
                        # Keep original weight type if not matching our patterns
                        new_key += f'.{weight_type}'
                        # Add .weight suffix if missing
                        if not new_key.endswith('.weight'):
                            new_key += '.weight'
                
                k = new_key
            else:
                # For other lora_unet__ formats (head, embeddings, etc.)
                new_key = main_part.replace('lora_unet__', 'diffusion_model.')
                
                # Fix specific component naming patterns
                new_key = new_key.replace('_self_attn', '.self_attn')
                new_key = new_key.replace('_cross_attn', '.cross_attn')
                new_key = new_key.replace('_ffn', '.ffn')
                new_key = new_key.replace('blocks_', 'blocks.')
                new_key = new_key.replace('head_head', 'head.head')
                new_key = new_key.replace('img_emb', 'img_emb')
                new_key = new_key.replace('text_embedding', 'text.embedding')
                new_key = new_key.replace('time_embedding', 'time.embedding')
                new_key = new_key.replace('time_projection', 'time.projection')
                
                # Replace remaining underscores with dots, carefully
                parts = new_key.split('.')
                final_parts = []
                for part in parts:
                    if part in ['img_emb', 'self_attn', 'cross_attn']:
                        final_parts.append(part)  # Keep these intact
                    else:
                        final_parts.append(part.replace('_', '.'))
                new_key = '.'.join(final_parts)
                
                # Handle weight type
                if weight_type:
                    if weight_type == 'alpha':
                        new_key += '.alpha'
                    elif weight_type == 'lora_down.weight' or weight_type == 'lora_down':
                        new_key += '.lora_A.weight'
                    elif weight_type == 'lora_up.weight' or weight_type == 'lora_up':
                        new_key += '.lora_B.weight'
                    else:
                        new_key += f'.{weight_type}'
                        if not new_key.endswith('.weight'):
                            new_key += '.weight'
                
                k = new_key
                
            # Handle special embedded components
            special_components = {
                'time.projection': 'time_projection',
                'img.emb': 'img_emb',
                'text.emb': 'text_emb',
                'time.emb': 'time_emb',
            }
            for old, new in special_components.items():
                if old in k:
                    k = k.replace(old, new)

        # Fix diffusion.model -> diffusion_model
        if k.startswith('diffusion.model.'):
            k = k.replace('diffusion.model.', 'diffusion_model.')
            
        # Finetrainer format
        if '.attn1.' in k:
            k = k.replace('.attn1.', '.cross_attn.')
            k = k.replace('.to_k.', '.k.')
            k = k.replace('.to_q.', '.q.')
            k = k.replace('.to_v.', '.v.')
            k = k.replace('.to_out.0.', '.o.')
        elif '.attn2.' in k:
            k = k.replace('.attn2.', '.cross_attn.')
            k = k.replace('.to_k.', '.k.')
            k = k.replace('.to_q.', '.q.')
            k = k.replace('.to_v.', '.v.')
            k = k.replace('.to_out.0.', '.o.')
            
        if "img_attn.proj" in k:
            k = k.replace("img_attn.proj", "img_attn_proj")
        if "img_attn.qkv" in k:
            k = k.replace("img_attn.qkv", "img_attn_qkv")
        if "txt_attn.proj" in k:
            k = k.replace("txt_attn.proj", "txt_attn_proj")
        if "txt_attn.qkv" in k:
            k = k.replace("txt_attn.qkv", "txt_attn_qkv")
        new_sd[k] = v
    return new_sd

class WanVideoBlockSwap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "blocks_to_swap": ("INT", {"default": 20, "min": 0, "max": 40, "step": 1, "tooltip": "Number of transformer blocks to swap, the 14B model has 40, while the 1.3B model has 30 blocks"}),
                "offload_img_emb": ("BOOLEAN", {"default": False, "tooltip": "Offload img_emb to offload_device"}),
                "offload_txt_emb": ("BOOLEAN", {"default": False, "tooltip": "Offload time_emb to offload_device"}),
            },
            "optional": {
                "use_non_blocking": ("BOOLEAN", {"default": True, "tooltip": "Use non-blocking memory transfer for offloading, reserves more RAM but is faster"}),
                "vace_blocks_to_swap": ("INT", {"default": 0, "min": 0, "max": 15, "step": 1, "tooltip": "Number of VACE blocks to swap, the VACE model has 15 blocks"}),
            },
        }
    RETURN_TYPES = ("BLOCKSWAPARGS",)
    RETURN_NAMES = ("block_swap_args",)
    FUNCTION = "setargs"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Settings for block swapping, reduces VRAM use by swapping blocks to CPU memory"

    def setargs(self, **kwargs):
        return (kwargs, )

class WanVideoVRAMManagement:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "offload_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Percentage of parameters to offload"}),
            },
        }
    RETURN_TYPES = ("VRAM_MANAGEMENTARGS",)
    RETURN_NAMES = ("vram_management_args",)
    FUNCTION = "setargs"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Alternative offloading method from DiffSynth-Studio, more aggressive in reducing memory use than block swapping, but can be slower"

    def setargs(self, **kwargs):
        return (kwargs, )

class WanVideoTorchCompileSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "backend": (["inductor","cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                "dynamo_cache_size_limit": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
                "compile_transformer_blocks_only": ("BOOLEAN", {"default": True, "tooltip": "Compile only the transformer blocks, usually enough and can make compilation faster and less error prone"}),
            },
            "optional": {
                "dynamo_recompile_limit": ("INT", {"default": 128, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.recompile_limit"}),
            },
        }
    RETURN_TYPES = ("WANCOMPILEARGS",)
    RETURN_NAMES = ("torch_compile_args",)
    FUNCTION = "set_args"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "torch.compile settings, when connected to the model loader, torch.compile of the selected layers is attempted. Requires Triton and torch 2.5.0 is recommended"

    def set_args(self, backend, fullgraph, mode, dynamic, dynamo_cache_size_limit, compile_transformer_blocks_only, dynamo_recompile_limit=128):

        compile_args = {
            "backend": backend,
            "fullgraph": fullgraph,
            "mode": mode,
            "dynamic": dynamic,
            "dynamo_cache_size_limit": dynamo_cache_size_limit,
            "dynamo_recompile_limit": dynamo_recompile_limit,
            "compile_transformer_blocks_only": compile_transformer_blocks_only,
        }

        return (compile_args, )

    
class WanVideoLoraSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
               "lora": (folder_paths.get_filename_list("loras"),
                {"tooltip": "LORA models are expected to be in ComfyUI/models/loras with .safetensors extension"}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
            },
            "optional": {
                "prev_lora":("WANVIDLORA", {"default": None, "tooltip": "For loading multiple LoRAs"}),
                "blocks":("SELECTEDBLOCKS", ),
                "low_mem_load": ("BOOLEAN", {"default": False, "tooltip": "Load the LORA model with less VRAM usage, slower loading. This affects ALL LoRAs, not just the current one"}),
                "merge_loras": ("BOOLEAN", {"default": True, "tooltip": "Merge LoRAs into the model, otherwise they are loaded on the fly. Always enabled for GGUF and scaled fp8 models. This affects ALL LoRAs, not just the current one"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("WANVIDLORA",)
    RETURN_NAMES = ("lora", )
    FUNCTION = "getlorapath"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Select a LoRA model from ComfyUI/models/loras"

    def getlorapath(self, lora, strength, unique_id, blocks={}, prev_lora=None, low_mem_load=False, merge_loras=True):
        loras_list = []

        strength = round(strength, 4)

        try:
            lora_path = folder_paths.get_full_path("loras", lora)
        except:
            lora_path = lora

        # Load metadata from the safetensors file
        metadata = {}
        try:
            from safetensors.torch import safe_open
            with safe_open(lora_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
        except Exception as e:
            print(f"Could not load metadata from {lora}: {e}")

        if unique_id and PromptServer is not None:
            try:
                # Build table rows for metadata
                metadata_rows = ""
                if metadata:
                    for key, value in metadata.items():
                        # Format value - handle special cases
                        if isinstance(value, dict):
                            formatted_value = "<pre>" + "\n".join([f"{k}: {v}" for k, v in value.items()]) + "</pre>"
                        elif isinstance(value, (list, tuple)):
                            formatted_value = "<pre>" + "\n".join([str(item) for item in value]) + "</pre>"
                        else:
                            formatted_value = str(value)
                        
                        metadata_rows += f"<tr><td><b>{key}</b></td><td>{formatted_value}</td></tr>"
                
                PromptServer.instance.send_progress_text(
                    f"<details>"
                    f"<summary><b>Metadata</b></summary>"
                    f"<table border='0' cellpadding='3'>"
                    f"<tr><td colspan='2'><b>Metadata</b></td></tr>"
                    f"{metadata_rows if metadata else '<tr><td>No metadata found</td></tr>'}"
                    f"</table>"
                    f"</details>", 
                    unique_id
                )
            except Exception as e:
                print(f"Error displaying metadata: {e}")
                pass

        lora = {
            "path": lora_path,
            "strength": strength,
            "name": lora.split(".")[0],
            "blocks": blocks.get("selected_blocks", {}),
            "layer_filter": blocks.get("layer_filter", ""),
            "low_mem_load": low_mem_load,
            "merge_loras": merge_loras,
        }
        if prev_lora is not None:
            loras_list.extend(prev_lora)

        loras_list.append(lora)
        return (loras_list,)
    
class WanVideoLoraSelectMulti:
    @classmethod
    def INPUT_TYPES(s):
        lora_files = folder_paths.get_filename_list("loras")
        lora_files = ["none"] + lora_files  # Add "none" as the first option
        return {
            "required": {
               "lora_0": (lora_files, {"default": "none"}),
                "strength_0": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
                "lora_1": (lora_files, {"default": "none"}),
                "strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
                "lora_2": (lora_files, {"default": "none"}),
                "strength_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
                "lora_3": (lora_files, {"default": "none"}),
                "strength_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
                "lora_4": (lora_files, {"default": "none"}),
                "strength_4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
            },
            "optional": {
                "prev_lora":("WANVIDLORA", {"default": None, "tooltip": "For loading multiple LoRAs"}),
                "blocks":("SELECTEDBLOCKS", ),
                "low_mem_load": ("BOOLEAN", {"default": False, "tooltip": "Load the LORA model with less VRAM usage, slower loading"}),
                "merge_loras": ("BOOLEAN", {"default": True, "tooltip": "Merge LoRAs into the model, otherwise they are loaded on the fly. Always enabled for GGUF and scaled fp8 models. This affects ALL LoRAs, not just the current one"}),

            }
        }

    RETURN_TYPES = ("WANVIDLORA",)
    RETURN_NAMES = ("lora", )
    FUNCTION = "getlorapath"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Select a LoRA model from ComfyUI/models/loras"

    def getlorapath(self, lora_0, strength_0, lora_1, strength_1, lora_2, strength_2, 
                lora_3, strength_3, lora_4, strength_4, blocks={}, prev_lora=None, 
                low_mem_load=False, merge_loras=True):
        loras_list = []

        strength_0 = round(strength_0, 4)
        strength_1 = round(strength_1, 4)
        strength_2 = round(strength_2, 4)
        strength_3 = round(strength_3, 4)
        strength_4 = round(strength_4, 4)
        
        if prev_lora is not None:
            loras_list.extend(prev_lora)
        
        # Process each LoRA
        lora_inputs = [
            (lora_0, strength_0), 
            (lora_1, strength_1), 
            (lora_2, strength_2), 
            (lora_3, strength_3), 
            (lora_4, strength_4)
        ]
        
        for lora_name, strength in lora_inputs:
            # Skip if the LoRA is empty
            if not lora_name or lora_name == "none":
                continue
                
            lora = {
                "path": folder_paths.get_full_path("loras", lora_name),
                "strength": strength,
                "name": lora_name.split(".")[0],
                "blocks": blocks.get("selected_blocks", {}),
                "layer_filter": blocks.get("layer_filter", ""),
                "low_mem_load": low_mem_load,
                "merge_loras": merge_loras,
            }
            
            loras_list.append(lora)
        
        return (loras_list,)
    
class WanVideoVACEModelSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vace_model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' VACE model to use when not using model that has it included"}),
            },
        }

    RETURN_TYPES = ("VACEPATH",)
    RETURN_NAMES = ("vace_model", )
    FUNCTION = "getvacepath"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "VACE model to use when not using model that has it included, loaded from 'ComfyUI/models/diffusion_models'"

    def getvacepath(self, vace_model):
        vace_model = {
            "path": folder_paths.get_full_path("diffusion_models", vace_model),
        }
        return (vace_model,)

class WanVideoLoraBlockEdit:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {}
        argument = ("BOOLEAN", {"default": True})

        for i in range(40):
            arg_dict["blocks.{}.".format(i)] = argument

        return {"required": arg_dict, "optional": {"layer_filter": ("STRING", {"default": "", "multiline": True})}}

    RETURN_TYPES = ("SELECTEDBLOCKS", )
    RETURN_NAMES = ("blocks", )
    OUTPUT_TOOLTIPS = ("The modified lora model",)
    FUNCTION = "select"

    CATEGORY = "WanVideoWrapper"

    def select(self, layer_filter=[], **kwargs):
        selected_blocks = {k: v for k, v in kwargs.items() if v is True and isinstance(v, bool)}
        print("Selected blocks LoRA: ", selected_blocks)
        selected = {
            "selected_blocks": selected_blocks,
            "layer_filter": [x.strip() for x in layer_filter.split(",")]
        }
        return (selected,)

#region Model loading
class WanVideoModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("unet_gguf") + folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),

            "base_precision": (["fp32", "bf16", "fp16", "fp16_fast"], {"default": "bf16"}),
            "quantization": (["disabled", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "fp8_e4m3fn_fast_no_ffn", "fp8_e4m3fn_scaled"], {"default": "disabled", "tooltip": "optional quantization method"}),
            "load_device": (["main_device", "offload_device"], {"default": "main_device", "tooltip": "Initial device to load the model to, NOT recommended with the larger models unless you have 48GB+ VRAM"}),
            },
            "optional": {
                "attention_mode": ([
                    "sdpa",
                    "flash_attn_2",
                    "flash_attn_3",
                    "sageattn",
                    "flex_attention",
                    "radial_sage_attention",
                    ], {"default": "sdpa"}),
                "compile_args": ("WANCOMPILEARGS", ),
                "block_swap_args": ("BLOCKSWAPARGS", ),
                "lora": ("WANVIDLORA", {"default": None}),
                "vram_management_args": ("VRAM_MANAGEMENTARGS", {"default": None, "tooltip": "Alternative offloading method from DiffSynth-Studio, more aggressive in reducing memory use than block swapping, but can be slower"}),
                "vace_model": ("VACEPATH", {"default": None, "tooltip": "VACE model to use when not using model that has it included"}),
                "fantasytalking_model": ("FANTASYTALKINGMODEL", {"default": None, "tooltip": "FantasyTalking model https://github.com/Fantasy-AMAP"}),
                "multitalk_model": ("MULTITALKMODEL", {"default": None, "tooltip": "Multitalk model"}),
            }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"

    def loadmodel(self, model, base_precision, load_device,  quantization,
                  compile_args=None, attention_mode="sdpa", block_swap_args=None, lora=None, vram_management_args=None, vace_model=None, fantasytalking_model=None, multitalk_model=None):
        assert not (vram_management_args is not None and block_swap_args is not None), "Can't use both block_swap_args and vram_management_args at the same time"
        
        lora_low_mem_load = merge_loras = False
        if lora is not None:
            for l in lora:
                lora_low_mem_load = l.get("low_mem_load", False)
                merge_loras = l.get("merge_loras", True)

        transformer = None
        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()
        manual_offloading = True
        if "sage" in attention_mode:
            try:
                from sageattention import sageattn
            except Exception as e:
                raise ValueError(f"Can't import SageAttention: {str(e)}")

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        gguf = False
        if model.endswith(".gguf"):
            if quantization != "disabled":
                raise ValueError("Quantization should be disabled when loading GGUF models.")
            quantization = "gguf"
            gguf = True
            merge_loras = False

                
        manual_offloading = True
        transformer_load_device = device if load_device == "main_device" else offload_device
        
        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[base_precision]
        
        if base_precision == "fp16_fast":
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                torch.backends.cuda.matmul.allow_fp16_accumulation = True
            else:
                raise ValueError("torch.backends.cuda.matmul.allow_fp16_accumulation is not available in this version of torch, requires torch 2.7.0.dev2025 02 26 nightly minimum currently")
        else:
            try:
                if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                    torch.backends.cuda.matmul.allow_fp16_accumulation = False
            except:
                pass
 

        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)
      
        if not gguf:
            sd = load_torch_file(model_path, device=transformer_load_device, safe_load=True)
        else:
            from diffusers.models.model_loading_utils import load_gguf_checkpoint
            sd = load_gguf_checkpoint(model_path)
        
        if quantization == "disabled":
            if "scaled_fp8" in sd:
                quantization = "fp8_e4m3fn_scaled"
            else:
                for k, v in sd.items():
                    if isinstance(v, torch.Tensor):
                        if v.dtype == torch.float8_e4m3fn:
                            quantization = "fp8_e4m3fn"
                            break
                        elif v.dtype == torch.float8_e5m2:
                            quantization = "fp8_e5m2"
                            break

        if "scaled_fp8" in sd and quantization != "fp8_e4m3fn_scaled":
            raise ValueError("The model is a scaled fp8 model, please set quantization to 'fp8_e4m3fn_scaled'")

        if merge_loras and "scaled" in quantization:
            raise ValueError("scaled models currently do not support merging LoRAs, please disable merging or use a non-scaled model")

        if "vace_blocks.0.after_proj.weight" in sd and not "patch_embedding.weight" in sd:
            raise ValueError("You are attempting to load a VACE module as a WanVideo model, instead you should use the vace_model input and matching T2V base model")

        if vace_model is not None:
            if gguf:
                if not vace_model["path"].endswith(".gguf"):
                    raise ValueError("With GGUF main model the VACE module must also be a GGUF quantized, if the main model already has VACE included, you can disconnect the VACE module loader")
                vace_sd = load_gguf_checkpoint(model_path)
            else:
                vace_sd = load_torch_file(vace_model["path"], device=transformer_load_device, safe_load=True)
            sd.update(vace_sd)

        first_key = next(iter(sd))
        if first_key.startswith("model.diffusion_model."):
            new_sd = {}
            for key, value in sd.items():
                new_key = key.replace("model.diffusion_model.", "", 1)
                new_sd[new_key] = value
            sd = new_sd
        elif first_key.startswith("model."):
            new_sd = {}
            for key, value in sd.items():
                new_key = key.replace("model.", "", 1)
                new_sd[new_key] = value
            sd = new_sd
        if not "patch_embedding.weight" in sd:
            raise ValueError("Invalid WanVideo model selected")
        dim = sd["patch_embedding.weight"].shape[0]
        in_features = sd["blocks.0.self_attn.k.weight"].shape[1]
        out_features = sd["blocks.0.self_attn.k.weight"].shape[0]
        in_channels = sd["patch_embedding.weight"].shape[1]
        log.info(f"Detected model in_channels: {in_channels}")
        ffn_dim = sd["blocks.0.ffn.0.bias"].shape[0]
        ffn2_dim = sd["blocks.0.ffn.2.weight"].shape[1]

        if not "text_embedding.0.weight" in sd:
            model_type = "no_cross_attn" #minimaxremover
        elif "model_type.Wan2_1-FLF2V-14B-720P" in sd or "img_emb.emb_pos" in sd or "flf2v" in model.lower():
            model_type = "fl2v"
        elif in_channels in [36, 48]:
            model_type = "i2v"
        elif in_channels == 16:
            model_type = "t2v"
        elif "control_adapter.conv.weight" in sd:
            model_type = "t2v"

        num_heads = 40 if dim == 5120 else 12
        num_layers = 40 if dim == 5120 else 30

        vace_layers, vace_in_dim = None, None
        if "vace_blocks.0.after_proj.weight" in sd:
            if in_channels != 16:
                raise ValueError("VACE only works properly with T2V models.")
            model_type = "t2v"
            if dim == 5120:
                vace_layers = [0, 5, 10, 15, 20, 25, 30, 35]
            else:
                vace_layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
            vace_in_dim = 96

        log.info(f"Model type: {model_type}, num_heads: {num_heads}, num_layers: {num_layers}")

        teacache_coefficients_map = {
            "1_3B": {
                "e": [2.39676752e+03, -1.31110545e+03, 2.01331979e+02, -8.29855975e+00, 1.37887774e-01],
                "e0": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            },
            "14B": {
                "e": [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404],
                "e0": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            },
            "i2v_480": {
                "e": [-3.02331670e+02, 2.23948934e+02, -5.25463970e+01, 5.87348440e+00, -2.01973289e-01],
                "e0": [2.57151496e+05, -3.54229917e+04, 1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            },
            "i2v_720":{
                "e": [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683],
                "e0": [8.10705460e+03, 2.13393892e+03, -3.72934672e+02, 1.66203073e+01, -4.17769401e-02],
            },
        }

        magcache_ratios_map = {
            "1_3B": np.array([1.0]*2+[1.0124, 1.02213, 1.00166, 1.0041, 0.99791, 1.00061, 0.99682, 0.99762, 0.99634, 0.99685, 0.99567, 0.99586, 0.99416, 0.99422, 0.99578, 0.99575, 0.9957, 0.99563, 0.99511, 0.99506, 0.99535, 0.99531, 0.99552, 0.99549, 0.99541, 0.99539, 0.9954, 0.99536, 0.99489, 0.99485, 0.99518, 0.99514, 0.99484, 0.99478, 0.99481, 0.99479, 0.99415, 0.99413, 0.99419, 0.99416, 0.99396, 0.99393, 0.99388, 0.99386, 0.99349, 0.99349, 0.99309, 0.99304, 0.9927, 0.9927, 0.99228, 0.99226, 0.99171, 0.9917, 0.99137, 0.99135, 0.99068, 0.99063, 0.99005, 0.99003, 0.98944, 0.98942, 0.98849, 0.98849, 0.98758, 0.98757, 0.98644, 0.98643, 0.98504, 0.98503, 0.9836, 0.98359, 0.98202, 0.98201, 0.97977, 0.97978, 0.97717, 0.97718, 0.9741, 0.97411, 0.97003, 0.97002, 0.96538, 0.96541, 0.9593, 0.95933, 0.95086, 0.95089, 0.94013, 0.94019, 0.92402, 0.92414, 0.90241, 0.9026, 0.86821, 0.86868, 0.81838, 0.81939]),
            "14B": np.array([1.0]*2+[1.02504, 1.03017, 1.00025, 1.00251, 0.9985, 0.99962, 0.99779, 0.99771, 0.9966, 0.99658, 0.99482, 0.99476, 0.99467, 0.99451, 0.99664, 0.99656, 0.99434, 0.99431, 0.99533, 0.99545, 0.99468, 0.99465, 0.99438, 0.99434, 0.99516, 0.99517, 0.99384, 0.9938, 0.99404, 0.99401, 0.99517, 0.99516, 0.99409, 0.99408, 0.99428, 0.99426, 0.99347, 0.99343, 0.99418, 0.99416, 0.99271, 0.99269, 0.99313, 0.99311, 0.99215, 0.99215, 0.99218, 0.99215, 0.99216, 0.99217, 0.99163, 0.99161, 0.99138, 0.99135, 0.98982, 0.9898, 0.98996, 0.98995, 0.9887, 0.98866, 0.98772, 0.9877, 0.98767, 0.98765, 0.98573, 0.9857, 0.98501, 0.98498, 0.9838, 0.98376, 0.98177, 0.98173, 0.98037, 0.98035, 0.97678, 0.97677, 0.97546, 0.97543, 0.97184, 0.97183, 0.96711, 0.96708, 0.96349, 0.96345, 0.95629, 0.95625, 0.94926, 0.94929, 0.93964, 0.93961, 0.92511, 0.92504, 0.90693, 0.90678, 0.8796, 0.87945, 0.86111, 0.86189]),
            "i2v_480": np.array([1.0]*2+[0.98783, 0.98993, 0.97559, 0.97593, 0.98311, 0.98319, 0.98202, 0.98225, 0.9888, 0.98878, 0.98762, 0.98759, 0.98957, 0.98971, 0.99052, 0.99043, 0.99383, 0.99384, 0.98857, 0.9886, 0.99065, 0.99068, 0.98845, 0.98847, 0.99057, 0.99057, 0.98957, 0.98961, 0.98601, 0.9861, 0.98823, 0.98823, 0.98756, 0.98759, 0.98808, 0.98814, 0.98721, 0.98724, 0.98571, 0.98572, 0.98543, 0.98544, 0.98157, 0.98165, 0.98411, 0.98413, 0.97952, 0.97953, 0.98149, 0.9815, 0.9774, 0.97742, 0.97825, 0.97826, 0.97355, 0.97361, 0.97085, 0.97087, 0.97056, 0.97055, 0.96588, 0.96587, 0.96113, 0.96124, 0.9567, 0.95681, 0.94961, 0.94969, 0.93973, 0.93988, 0.93217, 0.93224, 0.91878, 0.91896, 0.90955, 0.90954, 0.92617, 0.92616]),
            "i2v_720": np.array([1.0]*2+[0.99428, 0.99498, 0.98588, 0.98621, 0.98273, 0.98281, 0.99018, 0.99023, 0.98911, 0.98917, 0.98646, 0.98652, 0.99454, 0.99456, 0.9891, 0.98909, 0.99124, 0.99127, 0.99102, 0.99103, 0.99215, 0.99212, 0.99515, 0.99515, 0.99576, 0.99572, 0.99068, 0.99072, 0.99097, 0.99097, 0.99166, 0.99169, 0.99041, 0.99042, 0.99201, 0.99198, 0.99101, 0.99101, 0.98599, 0.98603, 0.98845, 0.98844, 0.98848, 0.98851, 0.98862, 0.98857, 0.98718, 0.98719, 0.98497, 0.98497, 0.98264, 0.98263, 0.98389, 0.98393, 0.97938, 0.9794, 0.97535, 0.97536, 0.97498, 0.97499, 0.973, 0.97301, 0.96827, 0.96828, 0.96261, 0.96263, 0.95335, 0.9534, 0.94649, 0.94655, 0.93397, 0.93414, 0.91636, 0.9165, 0.89088, 0.89109, 0.8679, 0.86768]),
        }

        model_variant = "14B" #default to this
        if model_type == "i2v" or model_type == "fl2v":
            if "480" in model or "fun" in model.lower() or "a2" in model.lower() or "540" in model: #just a guess for the Fun model for now...
                model_variant = "i2v_480"
            elif "720" in model:
                model_variant = "i2v_720"
        elif model_type == "t2v":
            model_variant = "14B"
            
        if dim == 1536:
            model_variant = "1_3B"
        log.info(f"Model variant detected: {model_variant}")
        
        TRANSFORMER_CONFIG= {
            "dim": dim,
            "in_features": in_features,
            "out_features": out_features,
            "ffn_dim": ffn_dim,
            "ffn2_dim": ffn2_dim,
            "eps": 1e-06,
            "freq_dim": 256,
            "in_dim": in_channels,
            "model_type": model_type,
            "out_dim": 16,
            "text_len": 512,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "attention_mode": attention_mode,
            "rope_func": "comfy",
            "main_device": device,
            "offload_device": offload_device,
            "teacache_coefficients": teacache_coefficients_map[model_variant],
            "magcache_ratios": magcache_ratios_map[model_variant],
            "vace_layers": vace_layers,
            "vace_in_dim": vace_in_dim,
            "inject_sample_info": True if "fps_embedding.weight" in sd else False,
            "add_ref_conv": True if "ref_conv.weight" in sd else False,
            "in_dim_ref_conv": sd["ref_conv.weight"].shape[1] if "ref_conv.weight" in sd else None,
            "add_control_adapter": True if "control_adapter.conv.weight" in sd else False,
        }

        with init_empty_weights():
            transformer = WanModel(**TRANSFORMER_CONFIG)
        transformer.eval()

        #ReCamMaster
        if "blocks.0.cam_encoder.weight" in sd:
            log.info("ReCamMaster model detected, patching model...")
            import torch.nn as nn
            for block in transformer.blocks:
                block.cam_encoder = nn.Linear(12, dim)
                block.projector = nn.Linear(dim, dim)
                block.cam_encoder.weight.data.zero_()
                block.cam_encoder.bias.data.zero_()
                block.projector.weight = nn.Parameter(torch.eye(dim))
                block.projector.bias = nn.Parameter(torch.zeros(dim))

        # FantasyTalking https://github.com/Fantasy-AMAP
        if fantasytalking_model is not None:
            log.info("FantasyTalking model detected, patching model...")
            context_dim = fantasytalking_model["sd"]["proj_model.proj.weight"].shape[0]
            import torch.nn as nn
            for block in transformer.blocks:
                block.cross_attn.k_proj = nn.Linear(context_dim, dim, bias=False)
                block.cross_attn.v_proj = nn.Linear(context_dim, dim, bias=False)
            sd.update(fantasytalking_model["sd"])
        if multitalk_model is not None:
            # init audio module
            from .multitalk.multitalk import SingleStreamMultiAttention
            from .wanvideo.modules.model import WanRMSNorm, WanLayerNorm
            norm_input_visual = True #dunno what this is
               
            for block in transformer.blocks:
                block.audio_cross_attn = SingleStreamMultiAttention(
                        dim=dim,
                        encoder_hidden_states_dim=768,
                        num_heads=num_heads,
                        qk_norm=False,
                        qkv_bias=True,
                        eps=transformer.eps,
                        norm_layer=WanRMSNorm,
                        class_range=24,
                        class_interval=4,
                        attention_mode=attention_mode,
                    )
                block.norm_x = WanLayerNorm(dim, transformer.eps, elementwise_affine=True) if norm_input_visual else nn.Identity()
            log.info("MultiTalk model detected, patching model...")
            
            sd.update(multitalk_model["sd"])

        
        # Additional cond latents
        if "add_conv_in.weight" in sd:
            def zero_module(module):
                for p in module.parameters():
                    torch.nn.init.zeros_(p)
                return module
            inner_dim = sd["add_conv_in.weight"].shape[0]
            add_cond_in_dim = sd["add_conv_in.weight"].shape[1]
            attn_cond_in_dim = sd["attn_conv_in.weight"].shape[1]
            transformer.add_conv_in = torch.nn.Conv3d(add_cond_in_dim, inner_dim, kernel_size=transformer.patch_size, stride=transformer.patch_size)
            transformer.add_proj = zero_module(torch.nn.Linear(inner_dim, inner_dim))
            transformer.attn_conv_in = torch.nn.Conv3d(attn_cond_in_dim, inner_dim, kernel_size=transformer.patch_size, stride=transformer.patch_size)
        
        comfy_model = WanVideoModel(
            WanVideoModelConfig(base_dtype),
            model_type=comfy.model_base.ModelType.FLOW,
            device=device,
        )
        
        if not gguf:
            if "fp8_e4m3fn" in quantization:
                dtype = torch.float8_e4m3fn
            elif quantization == "fp8_e5m2":
                dtype = torch.float8_e5m2
            else:
                dtype = base_dtype
            params_to_keep = {"norm", "head", "bias", "time_in", "vector_in", "patch_embedding", "time_", "img_emb", "modulation", "text_embedding", "adapter", "add"}
            if "scaled" in quantization:
                params_to_keep = {"patch_embedding", "modulation","norm", "bias"}
            #if lora is not None:
            #    transformer_load_device = device
            if not lora_low_mem_load:
                log.info("Using accelerate to load and assign model weights to device...")
                param_count = sum(1 for _ in transformer.named_parameters())
                pbar = ProgressBar(param_count)
                for name, param in tqdm(transformer.named_parameters(), 
                        desc=f"Loading transformer parameters to {transformer_load_device}", 
                        total=param_count,
                        leave=True):
                    dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype
                    if "patch_embedding" in name:
                        dtype_to_use = torch.float32
                    set_module_tensor_to_device(transformer, name, device=transformer_load_device, dtype=dtype_to_use, value=sd[name])
                    pbar.update(1)              

        comfy_model.diffusion_model = transformer
        comfy_model.load_device = transformer_load_device
        
        patcher = comfy.model_patcher.ModelPatcher(comfy_model, device, offload_device)
        patcher.model.is_patched = False

        control_lora = False
        
        if lora is not None:
            for l in lora:
                log.info(f"Loading LoRA: {l['name']} with strength: {l['strength']}")
                lora_path = l["path"]
                lora_strength = l["strength"]
                lora_sd = load_torch_file(lora_path, safe_load=True)
                if "dwpose_embedding.0.weight" in lora_sd: #unianimate
                    from .unianimate.nodes import update_transformer
                    log.info("Unianimate LoRA detected, patching model...")
                    transformer = update_transformer(transformer, lora_sd)

                lora_sd = standardize_lora_key_format(lora_sd)
                if l["blocks"]:
                    lora_sd = filter_state_dict_by_blocks(lora_sd, l["blocks"], l.get("layer_filter", []))

                #spacepxl's control LoRA patch
                # for key in lora_sd.keys():
                #     print(key)
                
                if "diffusion_model.patch_embedding.lora_A.weight" in lora_sd:
                    log.info("Control-LoRA detected, patching model...")
                    control_lora = True

                    in_cls = transformer.patch_embedding.__class__ # nn.Conv3d
                    old_in_dim = transformer.in_dim # 16
                    new_in_dim = lora_sd["diffusion_model.patch_embedding.lora_A.weight"].shape[1]
                    assert new_in_dim == 32
                    
                    new_in = in_cls(
                        new_in_dim,
                        transformer.patch_embedding.out_channels,
                        transformer.patch_embedding.kernel_size,
                        transformer.patch_embedding.stride,
                        transformer.patch_embedding.padding,
                    ).to(device=device, dtype=torch.float32)
                    
                    new_in.weight.zero_()
                    new_in.bias.zero_()
                    
                    new_in.weight[:, :old_in_dim].copy_(transformer.patch_embedding.weight)
                    new_in.bias.copy_(transformer.patch_embedding.bias)
                    
                    transformer.patch_embedding = new_in
                    transformer.expanded_patch_embedding = new_in
                    transformer.register_to_config(in_dim=new_in_dim)

                patcher, _ = load_lora_for_models(patcher, None, lora_sd, lora_strength, 0)
                
                del lora_sd
            
            if not gguf and not "scaled" in quantization and merge_loras:
                log.info("Patching LoRA to the model...")
                patcher = apply_lora(patcher, device, transformer_load_device, params_to_keep=params_to_keep, dtype=dtype, base_dtype=base_dtype, state_dict=sd, low_mem_load=lora_low_mem_load)
        
        if gguf:
            #from diffusers.quantizers.gguf.utils import _replace_with_gguf_linear, GGUFParameter
            from .gguf.gguf import _replace_with_gguf_linear, GGUFParameter
            log.info("Using GGUF to load and assign model weights to device...")
            param_count = sum(1 for _ in transformer.named_parameters())
            
            out_features = sd["blocks.0.self_attn.k.weight"].shape[1]
        
            patcher.model.diffusion_model = _replace_with_gguf_linear(patcher.model.diffusion_model, base_dtype, sd, patches=patcher.patches)
            pbar = ProgressBar(param_count)
            for name, param in tqdm(patcher.model.diffusion_model.named_parameters(), 
                    desc=f"Loading transformer parameters to {transformer_load_device}", 
                    total=param_count,
                    leave=True):
                #print(name, param.dtype, param.device, param.shape)
                if isinstance(param, GGUFParameter):
                    dtype_to_use = torch.uint8
                elif "patch_embedding" in name:
                    dtype_to_use = torch.float32
                else:
                    dtype_to_use = base_dtype
                set_module_tensor_to_device(patcher.model.diffusion_model, name, device=transformer_load_device, dtype=dtype_to_use, value=sd[name])
                pbar.update(1)
            #for name, param in transformer.named_parameters():
            #    print(name, param.dtype, param.device, param.shape)
            #patcher.load(device, full_load=True)

        patcher.model.is_patched = True

        
        
        if "fast" in quantization:
            if not merge_loras:
                raise ValueError("FP8 fast quantization requires LoRAs to be merged into the model, please set merge_loras=True in the LoRA input")
            from .fp8_optimization import convert_fp8_linear
            if quantization == "fp8_e4m3fn_fast_no_ffn":
                params_to_keep.update({"ffn"})
            print(params_to_keep)
            convert_fp8_linear(patcher.model.diffusion_model, base_dtype, params_to_keep=params_to_keep)
        
        if "scaled" in quantization:
            log.info("Using FP8 scaled linear quantization")
            from .fp8_optimization import convert_fp8_scaled_linear
            convert_fp8_scaled_linear(patcher.model.diffusion_model, sd, base_dtype, params_to_keep=params_to_keep, patches=patcher.patches)
        elif not merge_loras and not gguf:
            log.info("LoRAs will be applied at runtime")
            from .fp8_optimization import convert_linear_with_lora
            convert_linear_with_lora(patcher.model.diffusion_model, base_dtype, patches=patcher.patches)

        del sd

        if multitalk_model is not None:
            transformer.audio_proj = multitalk_model["proj_model"]

        if vram_management_args is not None:
            if gguf:
                raise ValueError("GGUF models don't support vram management")
            from .diffsynth.vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
            from .wanvideo.modules.model import WanLayerNorm, WanRMSNorm

            total_params_in_model = sum(p.numel() for p in patcher.model.diffusion_model.parameters())
            log.info(f"Total number of parameters in the loaded model: {total_params_in_model}")

            offload_percent = vram_management_args["offload_percent"]
            offload_params = int(total_params_in_model * offload_percent)
            params_to_keep = total_params_in_model - offload_params
            log.info(f"Selected params to offload: {offload_params}")
        
            enable_vram_management(
                patcher.model.diffusion_model,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    WanLayerNorm: AutoWrappedModule,
                    WanRMSNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device=offload_device,
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=base_dtype,
                    computation_device=device,
                ),
                max_num_param=params_to_keep,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device=offload_device,
                    onload_dtype=dtype,
                    onload_device=offload_device,
                    computation_dtype=base_dtype,
                    computation_device=device,
                ),
                compile_args = compile_args,
            )

        #compile
        if compile_args is not None and vram_management_args is None:
            torch._dynamo.config.cache_size_limit = compile_args["dynamo_cache_size_limit"]
            try:
                if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'config'):
                    torch._dynamo.config.recompile_limit = compile_args["dynamo_recompile_limit"]
            except Exception as e:
                log.warning(f"Could not set recompile_limit: {e}")
            if compile_args["compile_transformer_blocks_only"]:
                for i, block in enumerate(patcher.model.diffusion_model.blocks):
                    patcher.model.diffusion_model.blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
                if vace_layers is not None:
                    for i, block in enumerate(patcher.model.diffusion_model.vace_blocks):
                        patcher.model.diffusion_model.vace_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
            else:
                patcher.model.diffusion_model = torch.compile(patcher.model.diffusion_model, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])        
        
        if load_device == "offload_device" and patcher.model.diffusion_model.device != offload_device:
            log.info(f"Moving diffusion model from {patcher.model.diffusion_model.device} to {offload_device}")
            patcher.model.diffusion_model.to(offload_device)
            gc.collect()
            mm.soft_empty_cache()

        patcher.model["dtype"] = base_dtype
        patcher.model["base_path"] = model_path
        patcher.model["model_name"] = model
        patcher.model["manual_offloading"] = manual_offloading
        patcher.model["quantization"] = quantization
        patcher.model["auto_cpu_offload"] = True if vram_management_args is not None else False
        patcher.model["control_lora"] = control_lora

        if 'transformer_options' not in patcher.model_options:
            patcher.model_options['transformer_options'] = {}
        patcher.model_options["transformer_options"]["block_swap_args"] = block_swap_args   

        for model in mm.current_loaded_models:
            if model._model() == patcher:
                mm.current_loaded_models.remove(model)            

        return (patcher,)
    
#region load VAE

class WanVideoVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("vae"), {"tooltip": "These models are loaded from 'ComfyUI/models/vae'"}),
            },
            "optional": {
                "precision": (["fp16", "fp32", "bf16"],
                    {"default": "bf16"}
                ),
            }
        }

    RETURN_TYPES = ("WANVAE",)
    RETURN_NAMES = ("vae", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Loads Wan VAE model from 'ComfyUI/models/vae'"

    def loadmodel(self, model_name, precision):
        from .wanvideo.wan_video_vae import WanVideoVAE

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        #with open(os.path.join(script_directory, 'configs', 'hy_vae_config.json')) as f:
        #    vae_config = json.load(f)
        model_path = folder_paths.get_full_path("vae", model_name)
        vae_sd = load_torch_file(model_path, safe_load=True)

        has_model_prefix = any(k.startswith("model.") for k in vae_sd.keys())
        if not has_model_prefix:
            vae_sd = {f"model.{k}": v for k, v in vae_sd.items()}
        
        vae = WanVideoVAE(dtype=dtype)
        vae.load_state_dict(vae_sd)
        vae.eval()
        vae.to(device = offload_device, dtype = dtype)
            

        return (vae,)

class WanVideoTinyVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("vae_approx"), {"tooltip": "These models are loaded from 'ComfyUI/models/vae_approx'"}),
            },
            "optional": {
                "precision": (["fp16", "fp32", "bf16"], {"default": "fp16"}), 
                "parallel": ("BOOLEAN", {"default": False, "tooltip": "uses more memory but is faster"}),
            }
        }

    RETURN_TYPES = ("WANVAE",)
    RETURN_NAMES = ("vae", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Loads Wan VAE model from 'ComfyUI/models/vae'"

    def loadmodel(self, model_name, precision, parallel=False):
        from .taehv import TAEHV

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        model_path = folder_paths.get_full_path("vae_approx", model_name)
        vae_sd = load_torch_file(model_path, safe_load=True)
        
        vae = TAEHV(vae_sd, parallel=parallel)
       
        vae.to(device = offload_device, dtype = dtype)

        return (vae,)

class LoadWanVideoT5TextEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("text_encoders"), {"tooltip": "These models are loaded from 'ComfyUI/models/text_encoders'"}),
                "precision": (["fp32", "bf16"],
                    {"default": "bf16"}
                ),
            },
            "optional": {
                "load_device": (["main_device", "offload_device"], {"default": "offload_device"}),
                "quantization": (['disabled', 'fp8_e4m3fn'], {"default": 'disabled', "tooltip": "optional quantization method"}),
            }
        }

    RETURN_TYPES = ("WANTEXTENCODER",)
    RETURN_NAMES = ("wan_t5_model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Loads Wan text_encoder model from 'ComfyUI/models/LLM'"

    def loadmodel(self, model_name, precision, load_device="offload_device", quantization="disabled"):
        text_encoder_load_device = device if load_device == "main_device" else offload_device

        tokenizer_path = os.path.join(script_directory, "configs", "T5_tokenizer")

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        model_path = folder_paths.get_full_path("text_encoders", model_name)
        sd = load_torch_file(model_path, safe_load=True)
        
        if "token_embedding.weight" not in sd and "shared.weight" not in sd:
            raise ValueError("Invalid T5 text encoder model, this node expects the 'umt5-xxl' model")
        if "scaled_fp8" in sd:
            raise ValueError("Invalid T5 text encoder model, fp8 scaled is not supported by this node")

        # Convert state dict keys from T5 format to the expected format
        if "shared.weight" in sd:
            log.info("Converting T5 text encoder model to the expected format...")
            converted_sd = {}
            
            for key, value in sd.items():
                # Handle encoder block patterns
                if key.startswith('encoder.block.'):
                    parts = key.split('.')
                    block_num = parts[2]
                    
                    # Self-attention components
                    if 'layer.0.SelfAttention' in key:
                        if key.endswith('.k.weight'):
                            new_key = f"blocks.{block_num}.attn.k.weight"
                        elif key.endswith('.o.weight'):
                            new_key = f"blocks.{block_num}.attn.o.weight"
                        elif key.endswith('.q.weight'):
                            new_key = f"blocks.{block_num}.attn.q.weight"
                        elif key.endswith('.v.weight'):
                            new_key = f"blocks.{block_num}.attn.v.weight"
                        elif 'relative_attention_bias' in key:
                            new_key = f"blocks.{block_num}.pos_embedding.embedding.weight"
                        else:
                            new_key = key
                    
                    # Layer norms
                    elif 'layer.0.layer_norm' in key:
                        new_key = f"blocks.{block_num}.norm1.weight"
                    elif 'layer.1.layer_norm' in key:
                        new_key = f"blocks.{block_num}.norm2.weight"
                    
                    # Feed-forward components
                    elif 'layer.1.DenseReluDense' in key:
                        if 'wi_0' in key:
                            new_key = f"blocks.{block_num}.ffn.gate.0.weight"
                        elif 'wi_1' in key:
                            new_key = f"blocks.{block_num}.ffn.fc1.weight"
                        elif 'wo' in key:
                            new_key = f"blocks.{block_num}.ffn.fc2.weight"
                        else:
                            new_key = key
                    else:
                        new_key = key
                elif key == "shared.weight":
                    new_key = "token_embedding.weight"
                elif key == "encoder.final_layer_norm.weight":
                    new_key = "norm.weight"
                else:
                    new_key = key
                converted_sd[new_key] = value
            sd = converted_sd

        T5_text_encoder = T5EncoderModel(
            text_len=512,
            dtype=dtype,
            device=text_encoder_load_device,
            state_dict=sd,
            tokenizer_path=tokenizer_path,
            quantization=quantization
        )
        text_encoder = {
            "model": T5_text_encoder,
            "dtype": dtype,
        }
        
        return (text_encoder,)
    
class LoadWanVideoClipTextEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("clip_vision") + folder_paths.get_filename_list("text_encoders"), {"tooltip": "These models are loaded from 'ComfyUI/models/clip_vision'"}),
                 "precision": (["fp16", "fp32", "bf16"],
                    {"default": "fp16"}
                ),
            },
            "optional": {
                "load_device": (["main_device", "offload_device"], {"default": "offload_device"}),
            }
        }

    RETURN_TYPES = ("CLIP_VISION",) 
    RETURN_NAMES = ("wan_clip_vision", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Loads Wan clip_vision model from 'ComfyUI/models/clip_vision'"

    def loadmodel(self, model_name, precision, load_device="offload_device"):
        text_encoder_load_device = device if load_device == "main_device" else offload_device

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        model_path = folder_paths.get_full_path("clip_vision", model_name)
        # We also support legacy setups where the model is in the text_encoders folder
        if model_path is None:
            model_path = folder_paths.get_full_path("text_encoders", model_name)
        sd = load_torch_file(model_path, safe_load=True)
        if "log_scale" not in sd:
            raise ValueError("Invalid CLIP model, this node expectes the 'open-clip-xlm-roberta-large-vit-huge-14' model")

        clip_model = CLIPModel(dtype=dtype, device=device, state_dict=sd)
        clip_model.model.to(text_encoder_load_device)
        del sd
        
        return (clip_model,)

NODE_CLASS_MAPPINGS = {
    "WanVideoModelLoader": WanVideoModelLoader,
    "WanVideoVAELoader": WanVideoVAELoader,
    "WanVideoLoraSelect": WanVideoLoraSelect,
    "WanVideoLoraBlockEdit": WanVideoLoraBlockEdit,
    "WanVideoTinyVAELoader": WanVideoTinyVAELoader,
    "WanVideoVACEModelSelect": WanVideoVACEModelSelect,
    "WanVideoLoraSelectMulti": WanVideoLoraSelectMulti,
    "WanVideoBlockSwap": WanVideoBlockSwap,
    "WanVideoVRAMManagement": WanVideoVRAMManagement,
    "WanVideoTorchCompileSettings": WanVideoTorchCompileSettings,
    "LoadWanVideoT5TextEncoder": LoadWanVideoT5TextEncoder,
    "LoadWanVideoClipTextEncoder": LoadWanVideoClipTextEncoder,
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoModelLoader": "WanVideo Model Loader",
    "WanVideoVAELoader": "WanVideo VAE Loader",
    "WanVideoLoraSelect": "WanVideo Lora Select",
    "WanVideoLoraBlockEdit": "WanVideo Lora Block Edit",
    "WanVideoTinyVAELoader": "WanVideo Tiny VAE Loader",
    "WanVideoVACEModelSelect": "WanVideo VACE Module Select",
    "WanVideoLoraSelectMulti": "WanVideo Lora Select Multi",
    "WanVideoBlockSwap": "WanVideo Block Swap",
    "WanVideoVRAMManagement": "WanVideo VRAM Management",
    "WanVideoTorchCompileSettings": "WanVideo Torch Compile Settings",
    "LoadWanVideoT5TextEncoder": "WanVideo T5 Text Encoder Loader",
    "LoadWanVideoClipTextEncoder": "WanVideo CLIP Text Encoder Loader",
    }