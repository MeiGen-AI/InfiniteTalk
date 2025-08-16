import os
import torch
import gc
from ..utils import log

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

import comfy.model_management as mm
from comfy.utils import load_torch_file
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))


class DownloadAndLoadWav2VecModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                    "facebook/wav2vec2-base-960h", 
                    "TencentGameMate/chinese-wav2vec2-base"
                    ],
                ),

            "base_precision": (["fp32", "bf16", "fp16"], {"default": "fp16"}),
            "load_device": (["main_device", "offload_device"], {"default": "main_device", "tooltip": "Initial device to load the model to, NOT recommended with the larger models unless you have 48GB+ VRAM"}),
            },
        }
    

    RETURN_TYPES = ("WAV2VECMODEL",)
    RETURN_NAMES = ("wav2vec_model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"

    def loadmodel(self, model, base_precision, load_device):
        from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
        from ..multitalk.wav2vec2 import Wav2Vec2Model as MultiTalkWav2Vec2Model
        
        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[base_precision]
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        if load_device == "offload_device":
            transfomer_load_device = offload_device
        else:
            transfomer_load_device = device

        model_path = os.path.join(folder_paths.models_dir, "transformers", model)
        if not os.path.exists(model_path):
            log.info(f"Downloading Qwen model to: {model_path}")
            from huggingface_hub import snapshot_download
            ignore_patterns = None
            if model == "facebook/wav2vec2-base-960h":
                ignore_patterns = ["*.bin", "*.h5"]
            elif model == "TencentGameMate/chinese-wav2vec2-base":
                ignore_patterns = ["*.pt"]
            snapshot_download(
                repo_id=model,
                ignore_patterns=ignore_patterns,
                local_dir=model_path,
                local_dir_use_symlinks=False,
            )

        if model == "facebook/wav2vec2-base-960h":
            wav2vec_processor = Wav2Vec2Processor.from_pretrained(model_path)
            wav2vec = Wav2Vec2Model.from_pretrained(model_path).to(base_dtype).to(transfomer_load_device).eval()
        elif model == "TencentGameMate/chinese-wav2vec2-base":
            wav2vec = MultiTalkWav2Vec2Model.from_pretrained(model_path).to(base_dtype).to(transfomer_load_device).eval()
            wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path, local_files_only=True)

        wav2vec_processor_model = {
            "processor": wav2vec_processor if model == "facebook/wav2vec2-base-960h" else None,
            "feature_extractor": wav2vec_feature_extractor if model == "TencentGameMate/chinese-wav2vec2-base" else None,
            "model": wav2vec,
            "dtype": base_dtype,
            "model_type": model,
        }

        return (wav2vec_processor_model,)

class FantasyTalkingModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),

            "base_precision": (["fp32", "bf16", "fp16"], {"default": "fp16"}),
            },
        }

    RETURN_TYPES = ("FANTASYTALKINGMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"

    def loadmodel(self, model, base_precision):
        from .model import FantasyTalkingAudioConditionModel

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[base_precision]
        
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)
        sd = load_torch_file(model_path, device=offload_device, safe_load=True)

        with init_empty_weights():
            fantasytalking_proj_model = FantasyTalkingAudioConditionModel(audio_in_dim=768, audio_proj_dim=2048)
        #fantasytalking_proj_model.load_state_dict(sd, strict=False)

        for name, param in fantasytalking_proj_model.named_parameters():
            set_module_tensor_to_device(fantasytalking_proj_model, name, device=offload_device, dtype=base_dtype, value=sd[name])

        fantasytalking = {
            "proj_model": fantasytalking_proj_model,
            "sd": sd,
        }

        return (fantasytalking,)
    
class FantasyTalkingWav2VecEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "wav2vec_model": ("WAV2VECMODEL",),
            "fantasytalking_model": ("FANTASYTALKINGMODEL",),
            "audio": ("AUDIO",),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 1000, "step": 1}),
            "fps": ("FLOAT", {"default": 23.0, "min": 1.0, "max": 60.0, "step": 0.1}),
            "audio_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Strength of the audio conditioning"}),
            "audio_cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "When not 1.0, an extra model pass without audio conditioning is done: slower inference but more motion is allowed"}),
            },
        }

    RETURN_TYPES = ("FANTASYTALKING_EMBEDS", )
    RETURN_NAMES = ("fantasytalking_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, wav2vec_model, fantasytalking_model, fps, num_frames, audio_scale, audio_cfg_scale, audio):
        import torchaudio

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = wav2vec_model["dtype"]
        wav2vec = wav2vec_model["model"]
        wav2vec_processor = wav2vec_model["processor"]
        audio_proj_model = fantasytalking_model["proj_model"]

        sr = 16000

        audio_input = audio["waveform"]
        sample_rate = audio["sample_rate"]
        if sample_rate != sr:
            audio_input = torchaudio.functional.resample(audio_input, sample_rate, sr)
        audio_input = audio_input[0][0]

        start_time = 0
        end_time = num_frames / fps

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        try:
            audio_segment = audio_input[start_sample:end_sample]
        except:
            audio_segment = audio_input

        print("audio_segment.shape", audio_segment.shape)

        input_values = wav2vec_processor(
            audio_segment.numpy(), sampling_rate=sr, return_tensors="pt"
        ).input_values.to(dtype).to(device)

        wav2vec.to(device)
        audio_features = wav2vec(input_values).last_hidden_state
        wav2vec.to(offload_device)

        audio_proj_model.proj_model.to(device)
        audio_proj_fea = audio_proj_model.get_proj_fea(audio_features)
        pos_idx_ranges = audio_proj_model.split_audio_sequence(
            audio_proj_fea.size(1), num_frames=num_frames
        )
        audio_proj_split, audio_context_lens = audio_proj_model.split_tensor_with_padding(
            audio_proj_fea, pos_idx_ranges, expand_length=4
        )  # [b,21,9+8,768]
        audio_proj_model.proj_model.to(offload_device)
        mm.soft_empty_cache()

        out = {
            "audio_proj": audio_proj_split,
            "audio_context_lens": audio_context_lens,
            "audio_scale": audio_scale,
            "audio_cfg_scale": audio_cfg_scale
            }
    
        return (out,)


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadWav2VecModel": DownloadAndLoadWav2VecModel,
    "FantasyTalkingModelLoader": FantasyTalkingModelLoader,
    "FantasyTalkingWav2VecEmbeds": FantasyTalkingWav2VecEmbeds,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadWav2VecModel": "(Down)load Wav2Vec Model",
    "FantasyTalkingModelLoader": "FantasyTalking Model Loader",
    "FantasyTalkingWav2VecEmbeds": "FantasyTalking Wav2Vec Embeds",
    }
