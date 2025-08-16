import folder_paths
from comfy import model_management as mm
from comfy.utils import load_torch_file, common_upscale
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
import torch
from ..utils import log


class MultiTalkModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),

            "base_precision": (["fp32", "bf16", "fp16"], {"default": "fp16"}),
            },
        }

    RETURN_TYPES = ("MULTITALKMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"

    def loadmodel(self, model, base_precision):
        from .multitalk import AudioProjModel

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[base_precision]
        
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)
        sd = load_torch_file(model_path, device=offload_device, safe_load=True)

        audio_proj_keys = [k for k in sd.keys() if "audio_proj" in k]
        audio_proj_sd = {k.replace("audio_proj.", ""): sd.pop(k) for k in audio_proj_keys}

        audio_window=5
        intermediate_dim=512
        output_dim=768
        context_tokens=32
        vae_scale=4
        norm_output_audio = True

        with init_empty_weights():
            multitalk_proj_model = AudioProjModel(
                    seq_len=audio_window,
                    seq_len_vf=audio_window+vae_scale-1,
                    intermediate_dim=intermediate_dim,
                    output_dim=output_dim,
                    context_tokens=context_tokens,
                    norm_output_audio=norm_output_audio,
            )
        #fantasytalking_proj_model.load_state_dict(sd, strict=False)

        for name, param in multitalk_proj_model.named_parameters():
            set_module_tensor_to_device(multitalk_proj_model, name, device=offload_device, dtype=base_dtype, value=audio_proj_sd[name])

        multitalk = {
            "proj_model": multitalk_proj_model,
            "sd": sd,
        }

        return (multitalk,)
    

def loudness_norm(audio_array, sr=16000, lufs=-23):
    try:
        import pyloudnorm
    except:
        raise ImportError("pyloudnorm package is not installed")
    meter = pyloudnorm.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyloudnorm.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio
    
class MultiTalkWav2VecEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "wav2vec_model": ("WAV2VECMODEL",),
            "audio_1": ("AUDIO",),
            "normalize_loudness": ("BOOLEAN", {"default": True}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 1000, "step": 1}),
            "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 60.0, "step": 0.1}),
            "audio_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Strength of the audio conditioning"}),
            "audio_cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "When not 1.0, an extra model pass without audio conditioning is done: slower inference but more motion is allowed"}),
            "multi_audio_type": (["para", "add"], {"default": "para", "tooltip": "'para' overlay speakers in parallel, 'add' concatenate sequentially"}),
        },
            "optional" : {
                "audio_2": ("AUDIO",),
                "audio_3": ("AUDIO",),
                "audio_4": ("AUDIO",),
                "ref_target_masks": ("MASK", {"tooltip": "Per-speaker semantic mask(s) in pixel space. Supply one mask per speaker (plus optional background) to guide mouth assignment"}),
            }
        }

    RETURN_TYPES = ("MULTITALK_EMBEDS", "AUDIO", )
    RETURN_NAMES = ("multitalk_embeds", "audio", )
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, wav2vec_model, normalize_loudness, fps, num_frames, audio_1, audio_scale, audio_cfg_scale, multi_audio_type, audio_2=None, audio_3=None, audio_4=None, ref_target_masks=None):
        model_type = wav2vec_model["model_type"]
        if not "tencent" in model_type.lower():
            raise ValueError("Only tencent wav2vec models supported by MultiTalk")
        import torchaudio
        import numpy as np
        from einops import rearrange

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = wav2vec_model["dtype"]
        wav2vec = wav2vec_model["model"]
        wav2vec_feature_extractor = wav2vec_model["feature_extractor"]

        sr = 16000

        audio_inputs = [audio_1, audio_2, audio_3, audio_4]
        audio_inputs = [a for a in audio_inputs if a is not None]

        multitalk_audio_features = []
        seq_lengths = []
        audio_outputs = []  # for debugging / optional saving – choose first as return

        for audio in audio_inputs:
            audio_input = audio["waveform"]
            sample_rate = audio["sample_rate"]

            if sample_rate != 16000:
                audio_input = torchaudio.functional.resample(audio_input, sample_rate, sr)
            audio_input = audio_input[0][0]

            start_time = 0
            end_time = num_frames / fps

            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            try:
                audio_segment = audio_input[start_sample:end_sample]
            except Exception:
                audio_segment = audio_input

            audio_segment = audio_segment.numpy()

            if normalize_loudness:
                audio_segment = loudness_norm(audio_segment, sr=sr)

            audio_feature = np.squeeze(
                wav2vec_feature_extractor(audio_segment, sampling_rate=sr).input_values
            )

            audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
            audio_feature = audio_feature.unsqueeze(0)

            # audio encoder
            audio_duration = len(audio_segment) / sr
            video_length = audio_duration * fps

            wav2vec.to(device)
            embeddings = wav2vec(audio_feature.to(dtype), seq_len=int(video_length), output_hidden_states=True)
            wav2vec.to(offload_device)

            if len(embeddings) == 0:
                print("Fail to extract audio embedding for one speaker")
                continue

            audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
            audio_emb = rearrange(audio_emb, "b s d -> s b d")

            multitalk_audio_features.append(audio_emb.cpu().detach())
            seq_lengths.append(audio_emb.shape[0])

            waveform_tensor = torch.from_numpy(audio_segment).float().cpu().unsqueeze(0).unsqueeze(0)  # (B, C, N)
            audio_outputs.append({"waveform": waveform_tensor, "sample_rate": sr})

        log.info("[MultiTalk] --- Raw speaker lengths (samples) ---")
        for idx, ao in enumerate(audio_outputs):
            log.info(f"  speaker {idx+1}: {ao['waveform'].shape[-1]} samples (shape: {ao['waveform'].shape})")

        # Pad / combine depending on multi_audio_type
        if len(multitalk_audio_features) > 1:
            if multi_audio_type == "para":
                max_len = max(seq_lengths)
                padded = []
                for emb in multitalk_audio_features:
                    if emb.shape[0] < max_len:
                        pad = torch.zeros(max_len - emb.shape[0], *emb.shape[1:], dtype=emb.dtype)
                        emb = torch.cat([emb, pad], dim=0)
                    padded.append(emb)
                multitalk_audio_features = padded
            elif multi_audio_type == "add":
                total_len = sum(seq_lengths)
                full_list = []
                offset = 0
                for emb, length in zip(multitalk_audio_features, seq_lengths):
                    full = torch.zeros(total_len, *emb.shape[1:], dtype=emb.dtype)
                    full[offset:offset+length] = emb
                    full_list.append(full)
                    offset += length
                multitalk_audio_features = full_list

        # fallback
        if len(multitalk_audio_features) == 0:
            raise RuntimeError("No valid audio embeddings extracted, please check inputs")

        multitalk_embeds = {
            "audio_features": multitalk_audio_features,
            "audio_scale": audio_scale,
            "audio_cfg_scale": audio_cfg_scale,
            "ref_target_masks": ref_target_masks
        }

        if len(audio_outputs) == 1: # single speaker
            out_audio = audio_outputs[0]
        else: # multi speaker
            if multi_audio_type == "para":
                # Overlay speakers in parallel – mix waveforms to same length (max len)
                max_len = max([a["waveform"].shape[-1] for a in audio_outputs])
                mixed = torch.zeros(1, 1, max_len, dtype=audio_outputs[0]["waveform"].dtype)
                for a in audio_outputs:
                    w = a["waveform"]
                    if w.shape[-1] < max_len:
                        w = torch.nn.functional.pad(w, (0, max_len - w.shape[-1]))
                    mixed += w
                out_audio = {"waveform": mixed, "sample_rate": sr}
            else:  # "add" – sequential concatenate with silent padding for other speakers
                total_len = sum([a["waveform"].shape[-1] for a in audio_outputs])
                mixed = torch.zeros(1, 1, total_len, dtype=audio_outputs[0]["waveform"].dtype)
                offset = 0
                for a in audio_outputs:
                    w = a["waveform"]
                    mixed[:, :, offset:offset + w.shape[-1]] += w
                    offset += w.shape[-1]
                out_audio = {"waveform": mixed, "sample_rate": sr}

        # Debug: log final mixed audio length and mode
        total_samples_raw = sum([ao["waveform"].shape[-1] for ao in audio_outputs])
        log.info(f"[MultiTalk] total raw duration = {total_samples_raw/sr:.3f}s")
        log.info(f"[MultiTalk] multi_audio_type={multi_audio_type} | final waveform shape={out_audio['waveform'].shape} | length={out_audio['waveform'].shape[-1]} samples | seconds={out_audio['waveform'].shape[-1]/sr:.3f}s (expected {'sum' if multi_audio_type=='add' else 'max'} of raw)")

        return (multitalk_embeds, out_audio)


class WanVideoImageToVideoMultiTalk:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae": ("WANVAE",),
            "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 29048, "step": 8, "tooltip": "Height of the image to encode"}),
            "frame_window_size": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "motion_frame": ("INT", {"default": 25, "min": 1, "max": 10000, "step": 1, "tooltip": "Driven frame length used in the long video generation."}),
            "force_offload": ("BOOLEAN", {"default": True}),
            "colormatch": (
            [   
                'disabled',
                'mkl',
                'hm', 
                'reinhard', 
                'mvgd', 
                'hm-mvgd-hm', 
                'hm-mkl-hm',
            ], {
               "default": 'disabled'
            }),
            },
            "optional": {
                "start_image": ("IMAGE", {"tooltip": "Image to encode"}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
                "clip_embeds": ("WANVIDIMAGE_CLIPEMBEDS", {"tooltip": "Clip vision encoded image"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, vae, width, height, frame_window_size, motion_frame, force_offload, colormatch, start_image=None, tiled_vae=False, clip_embeds=None):

        H = height
        W = width
        VAE_STRIDE = (4, 8, 8)
        
        num_frames = ((frame_window_size - 1) // 4) * 4 + 1

        # Resize and rearrange the input image dimensions
        if start_image is not None:
            resized_start_image = common_upscale(start_image.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            resized_start_image = resized_start_image * 2 - 1
            resized_start_image = resized_start_image.unsqueeze(0)
        
        target_shape = (16, (num_frames - 1) // VAE_STRIDE[0] + 1,
                        height // VAE_STRIDE[1],
                        width // VAE_STRIDE[2])

        image_embeds = {
            "multitalk_sampling": True,
            "multitalk_start_image": resized_start_image if start_image is not None else None,
            "num_frames": num_frames,
            "motion_frame": motion_frame,
            "target_h": H,
            "target_w": W,
            "tiled_vae": tiled_vae,
            "force_offload": force_offload,
            "vae": vae,
            "target_shape": target_shape,
            "clip_context": clip_embeds.get("clip_embeds", None) if clip_embeds is not None else None,
            "colormatch": colormatch
        }

        return (image_embeds,)
    
NODE_CLASS_MAPPINGS = {
    "MultiTalkModelLoader": MultiTalkModelLoader,
    "MultiTalkWav2VecEmbeds": MultiTalkWav2VecEmbeds,
    "WanVideoImageToVideoMultiTalk": WanVideoImageToVideoMultiTalk    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiTalkModelLoader": "MultiTalk Model Loader",
    "MultiTalkWav2VecEmbeds": "MultiTalk Wav2Vec Embeds",
    "WanVideoImageToVideoMultiTalk": "WanVideo Image To Video MultiTalk"
}