# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import json
import math
import importlib
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial
from PIL import Image

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from diffusers.models.modeling_utils import no_init_weights, ContextManagers
import accelerate

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.multitalk_model import WanModel, WanLayerNorm, WanRMSNorm
from .modules.t5 import T5EncoderModel, T5LayerNorm, T5RelativeEmbedding
from .modules.vae import WanVAE, CausalConv3d, RMS_norm, Upsample
from .utils.multitalk_utils import MomentumBuffer, adaptive_projected_guidance, match_and_blend_colors
from src.vram_management import AutoWrappedQLinear, AutoWrappedLinear, AutoWrappedModule, enable_vram_management
from wan.utils.utils import convert_video_to_h264, extract_specific_frames, get_video_codec
from wan.wan_lora import WanLoraWrapper

from safetensors.torch import load_file
from optimum.quanto import quantize, freeze, qint8,requantize
import optimum.quanto.nn.qlinear as qlinear
from .utils.timing import timed, timing_enabled

def torch_gc():
    # NOTE: torch_gc() can be surprisingly expensive (and may dominate non-forward time).
    # Aggregate its cost into global timing stats when enabled.
    with timed(
        "torch_gc",
        key="runtime/torch_gc",
        log=False,
        sync_cuda=False,
        rank0_only=True,
    ):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def to_param_dtype_fp32only(model, param_dtype):
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if param.dtype == torch.float32 and param.__class__.__name__ not in ['WeightQBytesTensor']:
                param.data = param.data.to(param_dtype)
        for name, buf in module.named_buffers(recurse=False):
            if buf.dtype == torch.float32 and buf.__class__.__name__ not in ['WeightQBytesTensor']:
                module._buffers[name] = buf.to(param_dtype)
                
def resize_and_centercrop(cond_image, target_size):
        """
        Resize image or tensor to the target size without padding.
        """

        # Get the original size
        if isinstance(cond_image, torch.Tensor):
            _, orig_h, orig_w = cond_image.shape
        else:
            orig_h, orig_w = cond_image.height, cond_image.width

        target_h, target_w = target_size
        
        # Calculate the scaling factor for resizing
        scale_h = target_h / orig_h
        scale_w = target_w / orig_w
        
        # Compute the final size
        scale = max(scale_h, scale_w)
        final_h = math.ceil(scale * orig_h)
        final_w = math.ceil(scale * orig_w)
        
        # Resize
        if isinstance(cond_image, torch.Tensor):
            if len(cond_image.shape) == 3:
                cond_image = cond_image[None]
            resized_tensor = nn.functional.interpolate(cond_image, size=(final_h, final_w), mode='nearest').contiguous() 
            # crop
            cropped_tensor = transforms.functional.center_crop(resized_tensor, target_size) 
            cropped_tensor = cropped_tensor.squeeze(0)
        else:
            resized_image = cond_image.resize((final_w, final_h), resample=Image.BILINEAR)
            resized_image = np.array(resized_image)
            # tensor and crop
            resized_tensor = torch.from_numpy(resized_image)[None, ...].permute(0, 3, 1, 2).contiguous()
            cropped_tensor = transforms.functional.center_crop(resized_tensor, target_size)
            cropped_tensor = cropped_tensor[:, :, None, :, :] 

        return cropped_tensor


def timestep_transform(
    t,
    shift=5.0,
    num_timesteps=1000,
):
    t = t / num_timesteps
    # shift the timestep based on ratio
    new_t = shift * t / (1 + (shift - 1) * t)
    new_t = new_t * num_timesteps
    return new_t



class InfiniteTalkPipeline:

    def __init__(
        self,
        config,
        checkpoint_dir,
        quant_dir=None,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
        num_timesteps=1000,
        use_timestep_transform=True,
        lora_dir=None,
        lora_scales=None,
        quant = None,
        dit_path = None,
        infinitetalk_dir=None,
        enable_torch_compile=False,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            quant (`str`, *optional*, defaults to None):
                Quantization type, must be 'int8' or 'fp8'.
        """
        if quant is not None and quant not in ("int8", "fp8"):
            raise ValueError("quant must be 'int8', 'fp8', or None(default fp32 model)")
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)

        with timed(
            "[InfiniteTalkPipeline] load T5 text encoder",
            key="pipeline.init/load T5",
            log=True,
            rank0_only=True,
        ):
            logging.info("[InfiniteTalkPipeline] [1/5] Loading T5 text encoder ...")
            self.text_encoder = T5EncoderModel(
                text_len=config.text_len,
                dtype=config.t5_dtype,
                device=torch.device('cpu'),
                checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
                tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
                shard_fn=shard_fn if t5_fsdp else None,
                quant=quant,
                quant_dir=os.path.dirname(quant_dir) if quant_dir is not None else None,
            )
            logging.info("[InfiniteTalkPipeline] [1/5] T5 loaded.")

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        with timed(
            "[InfiniteTalkPipeline] load VAE",
            key="pipeline.init/load VAE",
            log=True,
            rank0_only=True,
        ):
            logging.info("[InfiniteTalkPipeline] [2/5] Loading VAE ...")
            self.vae = WanVAE(
                vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
                device=self.device,
            )
            logging.info("[InfiniteTalkPipeline] [2/5] VAE loaded.")

        with timed(
            "[InfiniteTalkPipeline] load CLIP",
            key="pipeline.init/load CLIP",
            log=True,
            rank0_only=True,
        ):
            logging.info("[InfiniteTalkPipeline] [3/5] Loading CLIP ...")
            self.clip = CLIPModel(
                dtype=config.clip_dtype,
                device=self.device,
                checkpoint_path=os.path.join(checkpoint_dir, config.clip_checkpoint),
                tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer),
            )
            logging.info("[InfiniteTalkPipeline] [3/5] CLIP loaded.")

        logging.info(f"Creating WanModel from {checkpoint_dir}")

        if quant is not None:
            with timed(
                "[InfiniteTalkPipeline] load quantized WanModel",
                key="pipeline.init/load quantized model",
                log=True,
                rank0_only=True,
            ):
                logging.info(f"Loading Quantized MultiTalk from {quant_dir}")
                with torch.device('meta'):
                    wan_config = json.load(open(os.path.join(checkpoint_dir, "config.json")))
                    self.model = WanModel(weight_init=False, **wan_config)
                    torch_gc()
                model_state_dict = load_file(quant_dir)
                map_json_path = os.path.join(quant_dir.replace('safetensors', 'json'))
                self.model.init_freqs()
                with open(map_json_path, "r") as f:
                    quantization_map = json.load(f)
                requantize(self.model, model_state_dict, quantization_map, device='cpu')
        else:
            if dit_path is None:
                with timed(
                    "[InfiniteTalkPipeline] init WanModel + load weights",
                    key="pipeline.init/init model + load weights",
                    log=True,
                    rank0_only=True,
                ):
                    init_contexts = [no_init_weights()]
                    init_contexts.append(accelerate.init_empty_weights())
                    wan_config = json.load(open(os.path.join(checkpoint_dir, "config.json")))
                    self.model = WanModel(weight_init=False, **wan_config).to(dtype=self.param_dtype)
                    weight_files = [
                        f"{checkpoint_dir}/diffusion_pytorch_model-00001-of-00007.safetensors",
                        f"{checkpoint_dir}/diffusion_pytorch_model-00002-of-00007.safetensors",
                        f"{checkpoint_dir}/diffusion_pytorch_model-00003-of-00007.safetensors",
                        f"{checkpoint_dir}/diffusion_pytorch_model-00004-of-00007.safetensors",
                        f"{checkpoint_dir}/diffusion_pytorch_model-00005-of-00007.safetensors",
                        f"{checkpoint_dir}/diffusion_pytorch_model-00006-of-00007.safetensors",
                        f"{checkpoint_dir}/diffusion_pytorch_model-00007-of-00007.safetensors",
                        f"{infinitetalk_dir}",
                    ]
                    merged_state_dict = {}
                    logging.info(
                        f"[InfiniteTalkPipeline] [4/5] Loading diffusion weights ({len(weight_files)} files) ..."
                    )
                    for idx, weight_file in enumerate(weight_files, start=1):
                        with timed(
                            f"[InfiniteTalkPipeline] load weights file {idx}/{len(weight_files)}",
                            key="pipeline.init/load weights file",
                            log=False,  # avoid per-file spam; aggregated only
                            rank0_only=True,
                        ):
                            logging.info(
                                f"[InfiniteTalkPipeline]   - loading {idx}/{len(weight_files)}: {os.path.basename(weight_file)}"
                            )
                            sd = load_file(weight_file)
                            merged_state_dict.update(sd)
                    with timed(
                        "[InfiniteTalkPipeline] apply merged state_dict",
                        key="pipeline.init/apply state_dict",
                        log=True,
                        rank0_only=True,
                    ):
                        logging.info("[InfiniteTalkPipeline] [4/5] All weights loaded, applying state_dict ...")
                        self.model.load_state_dict(merged_state_dict)
                        logging.info("[InfiniteTalkPipeline] [4/5] state_dict applied.")
                
            else:
                with timed(
                    "[InfiniteTalkPipeline] init WanModel from dit_path",
                    key="pipeline.init/init model from dit_path",
                    log=True,
                    rank0_only=True,
                ):
                    init_contexts = [no_init_weights()]
                    init_contexts.append(accelerate.init_empty_weights())
                    with ContextManagers(init_contexts):
                        wan_config = json.load(open(os.path.join(checkpoint_dir, "config.json")))
                        self.model = WanModel(weight_init=False, **wan_config)
                    checkpoint_weights = torch.load(dit_path, map_location='cpu')
                    self.model.load_state_dict(checkpoint_weights['state_dict'])
                    logging.info(f"loading infinitetalk weights {checkpoint_dir}")
            
        self.model.eval().requires_grad_(False)
        
        to_param_dtype_fp32only(self.model, self.param_dtype)
        if lora_dir is not None and quant is None :
            with timed(
                "[InfiniteTalkPipeline] load/apply LoRA",
                key="pipeline.init/lora",
                log=True,
                rank0_only=True,
            ):
                lora_wrapper = WanLoraWrapper(self.model)
                for lora_path, lora_scale in zip(lora_dir, lora_scales):
                    lora_name = lora_wrapper.load_lora(lora_path)
                    lora_wrapper.apply_lora(
                        lora_name, lora_scale, param_dtype=self.param_dtype, device=self.device
                    )


    

        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False
        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_dit_forward_multitalk,
                usp_attn_forward_multitalk,
                usp_crossattn_multi_forward_multitalk
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward_multitalk, block.self_attn)
                block.audio_cross_attn.forward = types.MethodType(
                    usp_crossattn_multi_forward_multitalk, block.audio_cross_attn)
            self.model.forward = types.MethodType(usp_dit_forward_multitalk, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            with timed(
                "[InfiniteTalkPipeline] shard DiT with FSDP",
                key="pipeline.init/fsdp shard",
                log=True,
                rank0_only=True,
            ):
                self.model = shard_fn(self.model)
        else:
            if not init_on_cpu:
                with timed(
                    "[InfiniteTalkPipeline] move DiT to device",
                    key="pipeline.init/model.to(device)",
                    log=True,
                    rank0_only=True,
                ):
                    self.model.to(self.device)
        
        self.sample_neg_prompt = config.sample_neg_prompt
        self.num_timesteps = num_timesteps
        self.use_timestep_transform = use_timestep_transform

        self.cpu_offload = False
        self.model_names = ["model"]
        self.vram_management = False

        self.enable_torch_compile = enable_torch_compile
        if enable_torch_compile:
            torch._dynamo.config.capture_scalar_outputs = True
            logging.info("Enabling torch compile for wan model, mode: max-autotune-no-cudagraphs.")
            self.model.compile(mode="max-autotune-no-cudagraphs", fullgraph=False, dynamic=None)

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timesteps = timesteps.float() / self.num_timesteps
        timesteps = timesteps.view(timesteps.shape + (1,) * (len(noise.shape)-1))

        return (1 - timesteps) * original_samples + timesteps * noise

    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.model.parameters())).dtype
        enable_vram_management(
            self.model,
            module_map={
                qlinear.QLinear: AutoWrappedQLinear,
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                WanLayerNorm: AutoWrappedModule,
                WanRMSNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.param_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.param_dtype,
                computation_device=self.device,
            ),
        )
        self.enable_cpu_offload()

    def enable_cpu_offload(self):
        self.cpu_offload = True
    
    def load_models_to_device(self, loadmodel_names=[]):
        with timed(
            "load_models_to_device",
            key="runtime/cpu_offload/load_models_to_device",
            log=False,
            rank0_only=True,
        ):
            # only load models to device if cpu_offload is enabled
            if not self.cpu_offload:
                return
            # offload the unneeded models to cpu
            for model_name in self.model_names:
                if model_name not in loadmodel_names:
                    model = getattr(self, model_name)

                    if not isinstance(model, nn.Module):
                        model = model.model

                    if model is not None:
                        if (
                            hasattr(model, "vram_management_enabled")
                            and model.vram_management_enabled
                        ):
                            for module in model.modules():
                                if hasattr(module, "offload"):
                                    module.offload()
                        else:
                            model.cpu()
            # load the needed models to device
            for model_name in loadmodel_names:
                model = getattr(self, model_name)
                if not isinstance(model, nn.Module):
                    model = model.model
                if model is not None:
                    if (
                        hasattr(model, "vram_management_enabled")
                        and model.vram_management_enabled
                    ):
                        for module in model.modules():
                            if hasattr(module, "onload"):
                                module.onload()
                    else:
                        model.to(self.device)
            # fresh the cuda cache
            torch.cuda.empty_cache()

   
    def generate_infinitetalk(self,
                 input_data,
                 size_buckget='infinitetalk-480',
                 motion_frame=25,
                 frame_num=81,
                 shift=5.0,
                 sampling_steps=40,
                 text_guide_scale=5.0,
                 audio_guide_scale=4.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 max_frames_num=1000,
                 face_scale=0.05,
                 progress=True,
                 color_correction_strength=0.0,
                 extra_args=None):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
        """

        # init teacache
        if extra_args.use_teacache:
            self.model.teacache_init(
                sample_steps=sampling_steps,
                teacache_thresh=extra_args.teacache_thresh,
                model_scale=extra_args.size,
            )
        else:
            self.model.disable_teacache()

        timing_on = timing_enabled(extra_args)
        if timing_on:
            # For non-generate entrypoints, still allow env-based enablement downstream.
            os.environ["INFINI_PRINT_TIMING"] = "1"
        _tp = "pipeline.generate_infinitetalk/"

        input_prompt = input_data['prompt']
        cond_file_path = input_data['cond_video']
        with timed(
            "preprocess: codec check/convert + extract first frame",
            enabled=timing_on,
            key=_tp + "preprocess/codec+first_frame",
            log=True,
            rank0_only=True,
        ):
            codec = get_video_codec(cond_file_path)
            if codec == 'av1':
                output_video_path = 'tmp/' + '_input_h264.mp4'
                print(f"Converting {cond_file_path} from AV1 to H.264...")
                convert_video_to_h264(cond_file_path, output_video_path)
                print(f"Conversion complete! Saved as {output_video_path}")
                cond_file_path = output_video_path
            else:
                print("No conversion needed.")
            cond_image = extract_specific_frames(cond_file_path, 0)
        # cond_image = Image.fromarray(cond_image)
        
        
        # decide a proper size
        with timed(
            "preprocess: choose bucket + resize/normalize",
            enabled=timing_on,
            key=_tp + "preprocess/bucket+resize",
            log=True,
            rank0_only=True,
        ):
            bucket_config_module = importlib.import_module("wan.utils.multitalk_utils")
            if size_buckget == 'infinitetalk-480':
                bucket_config = getattr(bucket_config_module, 'ASPECT_RATIO_627')
            elif size_buckget == 'infinitetalk-720':
                bucket_config = getattr(bucket_config_module, 'ASPECT_RATIO_960')

            src_h, src_w = cond_image.height, cond_image.width
            ratio = src_h / src_w
            closest_bucket = sorted(list(bucket_config.keys()), key=lambda x: abs(float(x)-ratio))[0]
            target_h, target_w = bucket_config[closest_bucket][0]
            cond_image = resize_and_centercrop(cond_image, (target_h, target_w))
            cond_image = cond_image / 255
            cond_image = (cond_image - 0.5) * 2 # normalization
            cond_image = cond_image.to(self.device)  # 1 C 1 H W

        # Store the original image for color reference if strength > 0
        original_color_reference = None
        if color_correction_strength > 0.0:
            original_color_reference = cond_image.clone()


        # read audio embeddings
        audio_embedding_path_1 = input_data['cond_audio']['person1']
        if len(input_data['cond_audio']) == 1:
            HUMAN_NUMBER = 1
            audio_embedding_path_2 = None
        else:
            HUMAN_NUMBER = 2
            audio_embedding_path_2 = input_data['cond_audio']['person2']

        
        with timed(
            "load audio embeddings (torch.load .pt)",
            enabled=timing_on,
            key=_tp + "io/load_audio_embeddings",
            log=True,
            rank0_only=True,
        ):
            full_audio_embs = []        
            audio_embedding_paths = [audio_embedding_path_1, audio_embedding_path_2]
            for human_idx in range(HUMAN_NUMBER):   
                audio_embedding_path = audio_embedding_paths[human_idx]
                if not os.path.exists(audio_embedding_path):
                    continue
                full_audio_emb = torch.load(audio_embedding_path)
                if torch.isnan(full_audio_emb).any():
                    continue
                if full_audio_emb.shape[0] <= frame_num:
                    continue
                full_audio_embs.append(full_audio_emb) 
        
        assert len(full_audio_embs) == HUMAN_NUMBER, f"Aduio file not exists or length not satisfies frame nums."

        # preprocess text embedding
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        with timed(
            "text encode (T5)",
            enabled=timing_on,
            key=_tp + "text/t5_encode",
            log=True,
            rank0_only=True,
        ):
            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)
                context, context_null = self.text_encoder([input_prompt, n_prompt], self.device)
                if offload_model:
                    self.text_encoder.model.cpu()
            else:
                context = self.text_encoder([input_prompt], torch.device('cpu'))
                context_null = self.text_encoder([n_prompt], torch.device('cpu'))
                context = [t.to(self.device) for t in context]
                context_null = [t.to(self.device) for t in context_null]

        torch_gc()
        # prepare params for video generation
        indices = (torch.arange(2 * 2 + 1) - 2) * 1 
        clip_length = frame_num
        is_first_clip = True
        arrive_last_frame = False
        cur_motion_frames_num = 1
        audio_start_idx = 0
        audio_end_idx = audio_start_idx + clip_length
        gen_video_list = []
        torch_gc()

        # set random seed and init noise
        seed = seed if seed >= 0 else random.randint(0, 99999999)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        if extra_args.enable_torch_deterministic:
            torch.backends.cudnn.deterministic = True
            logging.info("torch deterministic is enabled")
        else:
            logging.info("torch deterministic is disabled")

        # start video generation iteratively
        while True:
            with timed(
                "iter: slice audio embeddings window + to(device)",
                enabled=timing_on,
                key=_tp + "iter/audio_window",
                log=False,
                rank0_only=True,
            ):
                audio_embs = []
                # split audio with window size
                for human_idx in range(HUMAN_NUMBER):   
                    center_indices = torch.arange(
                        audio_start_idx,
                        audio_end_idx,
                        1,
                    ).unsqueeze(
                        1
                    ) + indices.unsqueeze(0)
                    center_indices = torch.clamp(center_indices, min=0, max=full_audio_embs[human_idx].shape[0]-1)
                    audio_emb = full_audio_embs[human_idx][center_indices][None,...].to(self.device)
                    audio_embs.append(audio_emb)
                audio_embs = torch.concat(audio_embs, dim=0).to(self.param_dtype)
                # torch_gc()

            h, w = cond_image.shape[-2], cond_image.shape[-1]
            lat_h, lat_w = h // self.vae_stride[1], w // self.vae_stride[2]
            max_seq_len = ((frame_num - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
                self.patch_size[1] * self.patch_size[2])
            max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size



            noise = torch.randn(
                16, (frame_num - 1) // 4 + 1,
                lat_h,
                lat_w,
                dtype=torch.float32,
                device=self.device) 

            # get mask
            msk = torch.ones(1, frame_num, lat_h, lat_w, device=self.device)
            msk[:, 1:] = 0
            msk = torch.concat([
                torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
            ],
                            dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
            msk = msk.transpose(1, 2).to(self.param_dtype) # B 4 T H W

            with torch.no_grad():
                # get clip embedding
                with timed(
                    "clip visual forward",
                    enabled=timing_on,
                    key=_tp + "clip/visual_forward",
                    log=False,
                    rank0_only=True,
                ):
                    self.clip.model.to(self.device)
                    clip_context = self.clip.visual(cond_image[:, :, -1:, :, :]).to(self.param_dtype) 
                    if offload_model:
                        self.clip.model.cpu()
                    # torch_gc()

                # zero padding and vae encode
                with timed(
                    "vae encode (padding frames)",
                    enabled=timing_on,
                    key=_tp + "vae/encode_padding",
                    log=False,
                    rank0_only=True,
                ):
                    video_frames = torch.zeros(1, cond_image.shape[1], frame_num-cond_image.shape[2], target_h, target_w).to(self.device)
                    padding_frames_pixels_values = torch.concat([cond_image, video_frames], dim=2)
                    y = self.vae.encode(padding_frames_pixels_values) 
                    y = torch.stack(y).to(self.param_dtype) # B C T H W
                    cur_motion_frames_latent_num = int(1 + (cur_motion_frames_num-1) // 4)

                with timed(
                    "vae encode (motion frames)",
                    enabled=timing_on,
                    key=_tp + "vae/encode_motion",
                    log=False,
                    rank0_only=True,
                ):
                    if is_first_clip:
                        latent_motion_frames = self.vae.encode(cond_image)[0]
                    else:
                        latent_motion_frames = self.vae.encode(cond_frame)[0]

                y = torch.concat([msk, y], dim=1) # B 4+C T H W
                # torch_gc()
            

            # construct human mask
            human_masks = []
            if HUMAN_NUMBER==1:
                background_mask = torch.ones([src_h, src_w])
                human_mask1 = torch.ones([src_h, src_w])
                human_mask2 = torch.ones([src_h, src_w])
                human_masks = [human_mask1, human_mask2, background_mask]
            elif HUMAN_NUMBER==2:
                if 'bbox' in input_data:
                    assert len(input_data['bbox']) == len(input_data['cond_audio']), f"The number of target bbox should be the same with cond_audio"
                    background_mask = torch.zeros([src_h, src_w])
                    for _, person_bbox in input_data['bbox'].items():
                        x_min, y_min, x_max, y_max = person_bbox
                        human_mask = torch.zeros([src_h, src_w])
                        human_mask[int(x_min):int(x_max), int(y_min):int(y_max)] = 1
                        background_mask += human_mask
                        human_masks.append(human_mask)
                else:
                    x_min, x_max = int(src_h * face_scale), int(src_h * (1 - face_scale))
                    background_mask = torch.zeros([src_h, src_w])
                    background_mask = torch.zeros([src_h, src_w])
                    human_mask1 = torch.zeros([src_h, src_w])
                    human_mask2 = torch.zeros([src_h, src_w])
                    lefty_min, lefty_max = int((src_w//2) * face_scale), int((src_w//2) * (1 - face_scale))
                    righty_min, righty_max = int((src_w//2) * face_scale + (src_w//2)), int((src_w//2) * (1 - face_scale) + (src_w//2))
                    human_mask1[x_min:x_max, lefty_min:lefty_max] = 1
                    human_mask2[x_min:x_max, righty_min:righty_max] = 1
                    background_mask += human_mask1
                    background_mask += human_mask2
                    human_masks = [human_mask1, human_mask2]
                background_mask = torch.where(background_mask > 0, torch.tensor(0), torch.tensor(1))
                human_masks.append(background_mask)

            ref_target_masks = torch.stack(human_masks, dim=0).to(self.device)
            # resize and centercrop for ref_target_masks 
            ref_target_masks = resize_and_centercrop(ref_target_masks, (target_h, target_w))

            _, _, _,lat_h, lat_w = y.shape
            ref_target_masks = F.interpolate(ref_target_masks.unsqueeze(0), size=(lat_h, lat_w), mode='nearest').squeeze() 
            ref_target_masks = (ref_target_masks > 0) 
            ref_target_masks = ref_target_masks.float().to(self.device)

            # torch_gc()

            @contextmanager
            def noop_no_sync():
                yield

            no_sync = getattr(self.model, 'no_sync', noop_no_sync)

            # evaluation mode
            with torch.no_grad(), no_sync():
                
                # prepare timesteps
                with timed(
                    "prepare timesteps",
                    enabled=timing_on,
                    key=_tp + "sampling/prepare_timesteps",
                    log=False,
                    rank0_only=True,
                ):
                    timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
                    timesteps.append(0.)
                    timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
                    if self.use_timestep_transform:
                        timesteps = [timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps) for t in timesteps]
                
                # sample videos
                latent = noise

                # prepare condition and uncondition configs
                arg_c = {
                    'context': [context],
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': y,
                    'audio': audio_embs,
                    'ref_target_masks': ref_target_masks
                }


                arg_null_text = {
                    'context': [context_null],
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': y,
                    'audio': audio_embs,
                    'ref_target_masks': ref_target_masks
                }

                arg_null_audio = {
                    'context': [context],
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': y,
                    'audio': torch.zeros_like(audio_embs)[-1:],
                    'ref_target_masks': ref_target_masks
                }


                arg_null = {
                    'context': [context_null],
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'y': y,
                    'audio': torch.zeros_like(audio_embs)[-1:],
                    'ref_target_masks': ref_target_masks
                }

                # torch_gc()
                if not self.vram_management:
                    self.model.to(self.device)
                else:
                    self.load_models_to_device(["model"])
                
                # injecting motion frames
                if not is_first_clip:
                    latent_motion_frames = latent_motion_frames.to(latent.dtype).to(self.device)
                    motion_add_noise = torch.randn_like(latent_motion_frames).contiguous()
                    add_latent = self.add_noise(latent_motion_frames, motion_add_noise, timesteps[0])
                    _, T_m, _, _ = add_latent.shape
                    latent[:, :T_m] = add_latent

                # infer with APG
                # refer https://arxiv.org/abs/2410.02416   
                if extra_args.use_apg:  
                    text_momentumbuffer  = MomentumBuffer(extra_args.apg_momentum) 
                    audio_momentumbuffer = MomentumBuffer(extra_args.apg_momentum) 


                progress_wrap = partial(tqdm, total=len(timesteps)-1) if progress else (lambda x: x)
                for i in progress_wrap(range(len(timesteps)-1)):
                    timestep = timesteps[i]
                    latent[:, :cur_motion_frames_latent_num] = latent_motion_frames
                    latent_model_input = [latent.to(self.device)]

                    # inference with CFG strategy
                    with timed(
                        "dit forward: cond",
                        enabled=timing_on,
                        key=_tp + "forward/cond",
                        log=False,
                        rank0_only=True,
                    ):
                        noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c)[0] 
                    # torch_gc()

                    if math.isclose(text_guide_scale, 1.0):
                        with timed(
                            "dit forward: drop_audio",
                            enabled=timing_on,
                            key=_tp + "forward/drop_audio",
                            log=False,
                            rank0_only=True,
                        ):
                            noise_pred_drop_audio = self.model(
                                latent_model_input, t=timestep, **arg_null_audio)[0]  
                        # torch_gc()
                    else:
                        with timed(
                            "dit forward: drop_text",
                            enabled=timing_on,
                            key=_tp + "forward/drop_text",
                            log=False,
                            rank0_only=True,
                        ):
                            noise_pred_drop_text = self.model(
                                latent_model_input, t=timestep, **arg_null_text)[0] 
                        # torch_gc()
                        with timed(
                            "dit forward: uncond",
                            enabled=timing_on,
                            key=_tp + "forward/uncond",
                            log=False,
                            rank0_only=True,
                        ):
                            noise_pred_uncond = self.model(
                                latent_model_input, t=timestep, **arg_null)[0]  
                        # torch_gc()

                    if extra_args.use_apg:
                        # correct update direction
                        if math.isclose(text_guide_scale, 1.0):
                            diff_uncond_audio  = noise_pred_cond - noise_pred_drop_audio
                            noise_pred = noise_pred_cond + (audio_guide_scale - 1)* adaptive_projected_guidance(diff_uncond_audio, 
                                                                                            noise_pred_cond, 
                                                                                            momentum_buffer=audio_momentumbuffer, 
                                                                                            norm_threshold=extra_args.apg_norm_threshold)
                        else:
                            diff_uncond_text  = noise_pred_cond - noise_pred_drop_text
                            diff_uncond_audio = noise_pred_drop_text - noise_pred_uncond
                            noise_pred = noise_pred_cond + (text_guide_scale - 1) * adaptive_projected_guidance(diff_uncond_text, 
                                                                                                                noise_pred_cond, 
                                                                                                                momentum_buffer=text_momentumbuffer, 
                                                                                                                norm_threshold=extra_args.apg_norm_threshold) \
                                + (audio_guide_scale - 1) * adaptive_projected_guidance(diff_uncond_audio, 
                                                                                            noise_pred_cond, 
                                                                                            momentum_buffer=audio_momentumbuffer, 
                                                                                            norm_threshold=extra_args.apg_norm_threshold)
                    else:
                        # vanilla CFG strategy
                        if math.isclose(text_guide_scale, 1.0):
                            noise_pred = noise_pred_drop_audio + audio_guide_scale* (noise_pred_cond - noise_pred_drop_audio)  
                        else:
                            noise_pred = noise_pred_uncond + text_guide_scale * (
                                noise_pred_cond - noise_pred_drop_text) + \
                                audio_guide_scale * (noise_pred_drop_text - noise_pred_uncond)  
                    noise_pred = -noise_pred  

                    # update latent
                    dt = timesteps[i] - timesteps[i + 1]
                    dt = dt / self.num_timesteps
                    latent = latent + noise_pred * dt[:, None, None, None]

                    # injecting motion frames
                    if not is_first_clip:
                        latent_motion_frames = latent_motion_frames.to(latent.dtype).to(self.device)
                        motion_add_noise = torch.randn_like(latent_motion_frames).contiguous()
                        add_latent = self.add_noise(latent_motion_frames, motion_add_noise, timesteps[i+1])
                        _, T_m, _, _ = add_latent.shape
                        latent[:, :T_m] = add_latent

                    latent[:, :cur_motion_frames_latent_num] = latent_motion_frames
                    x0 = [latent.to(self.device)] 
                    del latent_model_input, timestep
                
                if offload_model: 
                    if not self.vram_management:
                        self.model.cpu()
                # torch_gc()

                with timed(
                    "vae decode",
                    enabled=timing_on,
                    key=_tp + "vae/decode",
                    log=False,
                    rank0_only=True,
                ):
                    videos = self.vae.decode(x0)
            
            # cache generated samples
            videos = torch.stack(videos).cpu() # B C T H W
            # >>> START OF COLOR CORRECTION STEP <<<
            if color_correction_strength > 0.0 and original_color_reference is not None:
                with timed(
                    "color correction",
                    enabled=timing_on,
                    key=_tp + "post/color_correction",
                    log=False,
                    rank0_only=True,
                ):
                    videos = match_and_blend_colors(videos, original_color_reference, color_correction_strength)
            # >>> END OF COLOR CORRECTION STEP <<<

            if is_first_clip:
                gen_video_list.append(videos)
            else:
                gen_video_list.append(videos[:, :, cur_motion_frames_num:])

            # decide whether is done
            if arrive_last_frame: break

            # update next condition frames
            is_first_clip = False
            cur_motion_frames_num = motion_frame

            cond_frame = videos[:, :, -cur_motion_frames_num:].to(torch.float32).to(self.device)
            audio_start_idx += (frame_num - cur_motion_frames_num)
            audio_end_idx = audio_start_idx + clip_length

            cond_image = extract_specific_frames(cond_file_path, audio_start_idx)
            # cond_image = Image.fromarray(cond_image)
            cond_image = resize_and_centercrop(cond_image, (target_h, target_w))
            cond_image = cond_image / 255
            cond_image = (cond_image - 0.5) * 2 # normalization
            cond_image = cond_image.to(self.device)  # 1 C 1 H W

            # Repeat audio emb
            if audio_end_idx >= min(max_frames_num, len(full_audio_embs[0])):
                arrive_last_frame = True
                miss_lengths = []
                source_frames = []
                for human_inx in range(HUMAN_NUMBER):
                    source_frame = len(full_audio_embs[human_inx])
                    source_frames.append(source_frame)
                    if audio_end_idx >= len(full_audio_embs[human_inx]):
                        miss_length   = audio_end_idx - len(full_audio_embs[human_inx]) + 3 
                        add_audio_emb = torch.flip(full_audio_embs[human_inx][-1*miss_length:], dims=[0])
                        full_audio_embs[human_inx] = torch.cat([full_audio_embs[human_inx], add_audio_emb], dim=0)
                        miss_lengths.append(miss_length)
                    else:
                        miss_lengths.append(0)

            
            if max_frames_num <= frame_num: break
            
            # torch_gc()
            if offload_model:    
                torch.cuda.synchronize()
            if dist.is_initialized():
                with timed(
                    "dist.barrier (per-iter end)",
                    enabled=timing_on,
                    key=_tp + "dist/barrier_iter_end",
                    log=False,
                    rank0_only=True,
                ):
                    dist.barrier()
        
        gen_video_samples = torch.cat(gen_video_list, dim=2)[:, :, :int(max_frames_num)] 
        gen_video_samples = gen_video_samples.to(torch.float32)
        if max_frames_num > frame_num and sum(miss_lengths) > 0:
            # split video frames
            # gen_video_samples = gen_video_samples[:, :, :-1*miss_lengths[0]]
            gen_video_samples = gen_video_samples[:, :, :full_audio_emb.shape[0]]
        
        if dist.is_initialized():
            with timed(
                "dist.barrier (final)",
                enabled=timing_on,
                key=_tp + "dist/barrier_final",
                log=False,
                rank0_only=True,
            ):
                dist.barrier()

        del noise, latent
        torch_gc()

        return gen_video_samples[0] if self.rank == 0 else None
    

   
