import os
import torch
import gc
from ..utils import log, print_memory, fourier_filter, optimized_scale
import math
from tqdm import tqdm

from ..wanvideo.modules.model import rope_params
from ..wanvideo.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from ..wanvideo.schedulers.scheduling_flow_match_lcm import FlowMatchLCMScheduler

from einops import rearrange

from ..enhance_a_video.globals import disable_enhance

import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar, common_upscale
from comfy.clip_vision import clip_preprocess, ClipVisionModel
from comfy.cli_args import args, LatentPreviewMethod

script_directory = os.path.dirname(os.path.abspath(__file__))

def generate_timestep_matrix(
        num_frames,
        step_template,
        base_num_frames,
        ar_step=5,
        num_pre_ready=0,
        casual_block_size=1,
        shrink_interval_with_mask=False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple]]:
        step_matrix, step_index = [], []
        update_mask, valid_interval = [], []
        num_iterations = len(step_template) + 1
        num_frames_block = num_frames // casual_block_size
        base_num_frames_block = base_num_frames // casual_block_size
        if base_num_frames_block < num_frames_block:
            infer_step_num = len(step_template)
            gen_block = base_num_frames_block
            min_ar_step = infer_step_num / gen_block
            assert ar_step >= min_ar_step, f"ar_step should be at least {math.ceil(min_ar_step)} in your setting"
        # print(num_frames, step_template, base_num_frames, ar_step, num_pre_ready, casual_block_size, num_frames_block, base_num_frames_block)
        step_template = torch.cat(
            [
                torch.tensor([999], dtype=torch.int64, device=step_template.device),
                step_template.long(),
                torch.tensor([0], dtype=torch.int64, device=step_template.device),
            ]
        )  # to handle the counter in row works starting from 1
        pre_row = torch.zeros(num_frames_block, dtype=torch.long)
        if num_pre_ready > 0:
            pre_row[: num_pre_ready // casual_block_size] = num_iterations

        while torch.all(pre_row >= (num_iterations - 1)) == False:
            new_row = torch.zeros(num_frames_block, dtype=torch.long)
            for i in range(num_frames_block):
                if i == 0 or pre_row[i - 1] >= (
                    num_iterations - 1
                ):  # the first frame or the last frame is completely denoised
                    new_row[i] = pre_row[i] + 1
                else:
                    new_row[i] = new_row[i - 1] - ar_step
            new_row = new_row.clamp(0, num_iterations)

            update_mask.append(
                (new_row != pre_row) & (new_row != num_iterations)
            )  # False: no need to update， True: need to update
            step_index.append(new_row)
            step_matrix.append(step_template[new_row])
            pre_row = new_row

        # for long video we split into several sequences, base_num_frames is set to the model max length (for training)
        terminal_flag = base_num_frames_block
        if shrink_interval_with_mask:
            idx_sequence = torch.arange(num_frames_block, dtype=torch.int64)
            update_mask = update_mask[0]
            update_mask_idx = idx_sequence[update_mask]
            last_update_idx = update_mask_idx[-1].item()
            terminal_flag = last_update_idx + 1
        # for i in range(0, len(update_mask)):
        for curr_mask in update_mask:
            if terminal_flag < num_frames_block and curr_mask[terminal_flag]:
                terminal_flag += 1
            valid_interval.append((max(terminal_flag - base_num_frames_block, 0), terminal_flag))

        step_update_mask = torch.stack(update_mask, dim=0)
        step_index = torch.stack(step_index, dim=0)
        step_matrix = torch.stack(step_matrix, dim=0)

        if casual_block_size > 1:
            step_update_mask = step_update_mask.unsqueeze(-1).repeat(1, 1, casual_block_size).flatten(1).contiguous()
            step_index = step_index.unsqueeze(-1).repeat(1, 1, casual_block_size).flatten(1).contiguous()
            step_matrix = step_matrix.unsqueeze(-1).repeat(1, 1, casual_block_size).flatten(1).contiguous()
            valid_interval = [(s * casual_block_size, e * casual_block_size) for s, e in valid_interval]

        return step_matrix, step_index, step_update_mask, valid_interval

#region Sampler
class WanVideoDiffusionForcingSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "text_embeds": ("WANVIDEOTEXTEMBEDS", ),
                "image_embeds": ("WANVIDIMAGE_EMBEDS", ),
                "addnoise_condition": ("INT", {"default": 10, "min": 0, "max": 1000, "tooltip": "Improves consistency in long video generation"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "steps": ("INT", {"default": 30, "min": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Moves the model to the offload device after sampling"}),
                "scheduler": (["unipc", "unipc/beta", "euler", "euler/beta", "lcm", "lcm/beta"],
                    {
                        "default": 'unipc'
                    }),
            },
            "optional": {
                "samples": ("LATENT", {"tooltip": "init Latents to use for video2video process"} ),
                "prefix_samples": ("LATENT", {"tooltip": "prefix latents"} ),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cache_args": ("CACHEARGS", ),
                "slg_args": ("SLGARGS", ),
                "rope_function": (["default", "comfy"], {"default": "comfy", "tooltip": "Comfy's RoPE implementation doesn't use complex numbers and can thus be compiled, that should be a lot faster when using torch.compile"}),
                "experimental_args": ("EXPERIMENTALARGS", ),
                "unianimate_poses": ("UNIANIMATE_POSE", ),
            }
        }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, model, text_embeds, image_embeds, shift, fps, steps, addnoise_condition, cfg, seed, scheduler, 
        force_offload=True, samples=None, prefix_samples=None, denoise_strength=1.0, slg_args=None, rope_function="default", cache_args=None, teacache_args=None, 
        experimental_args=None, unianimate_poses=None):
        #assert not (context_options and teacache_args), "Context options cannot currently be used together with teacache."
        patcher = model
        model = model.model
        transformer = model.diffusion_model
        dtype = model["dtype"]
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        steps = int(steps/denoise_strength)

        timesteps = None
        if 'unipc' in scheduler:
            sample_scheduler = FlowUniPCMultistepScheduler(shift=shift)
            sample_scheduler.set_timesteps(steps, device=device, shift=shift, use_beta_sigmas=('beta' in scheduler))
        elif 'euler' in scheduler:
            sample_scheduler = FlowMatchEulerDiscreteScheduler(shift=shift, use_beta_sigmas=(scheduler == 'euler/beta'))
            sample_scheduler.set_timesteps(steps, device=device)
        elif 'lcm' in scheduler:
            sample_scheduler = FlowMatchLCMScheduler(shift=shift, use_beta_sigmas=(scheduler == 'lcm/beta'))
            sample_scheduler.set_timesteps(steps, device=device) 
        
        
        init_timesteps = sample_scheduler.timesteps
        
        if denoise_strength < 1.0:
            steps = int(steps * denoise_strength)
            timesteps = timesteps[-(steps + 1):] 
        
        seed_g = torch.Generator(device=torch.device("cpu"))
        seed_g.manual_seed(seed)
       
        clip_fea, clip_fea_neg = None, None
        vace_data, vace_context, vace_scale = None, None, None

        image_cond = image_embeds.get("image_embeds", None)

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
        
        latent_video_length = noise.shape[1]  
        seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * noise.shape[1])

        
               
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

        latents = noise.to(device)
        
        fps_embeds = None
        if hasattr(transformer, "fps_embedding"):
            fps = round(fps, 2)
            log.info(f"Model has fps embedding, using {fps} fps")
            fps_embeds = [fps]
            fps_embeds = [0 if i == 16 else 1 for i in fps_embeds]

        prefix_video = prefix_samples["samples"].to(noise) if prefix_samples is not None else None
        prefix_video_latent_length = prefix_video.shape[2] if prefix_video is not None else 0
        if prefix_video is not None:
            log.info(f"Prefix video of length: {prefix_video_latent_length}")
            latents[:, :prefix_video_latent_length] = prefix_video[0]
        #base_num_frames = (base_num_frames - 1) // 4 + 1 if base_num_frames is not None else latent_video_length
        base_num_frames=latent_video_length

        ar_step = 0
        causal_block_size = 1
        step_matrix, _, step_update_mask, valid_interval = generate_timestep_matrix(
                latent_video_length, init_timesteps, base_num_frames, ar_step, prefix_video_latent_length, causal_block_size
            )
        
        sample_schedulers = []
        for _ in range(latent_video_length):
            if 'unipc' in scheduler:
                sample_scheduler = FlowUniPCMultistepScheduler(shift=shift)
                sample_scheduler.set_timesteps(steps, device=device, shift=shift, use_beta_sigmas=('beta' in scheduler))
            elif 'euler' in scheduler:
                sample_scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
                sample_scheduler.set_timesteps(steps, device=device)
            elif 'lcm' in scheduler:
                sample_scheduler = FlowMatchLCMScheduler(shift=shift, use_beta_sigmas=(scheduler == 'lcm/beta'))
                sample_scheduler.set_timesteps(steps, device=device) 
            
            sample_schedulers.append(sample_scheduler)
        sample_schedulers_counter = [0] * latent_video_length

        unianim_data = None
        if unianimate_poses is not None:
            transformer.dwpose_embedding.to(device)
            transformer.randomref_embedding_pose.to(device)
            dwpose_data = unianimate_poses["pose"]
            dwpose_data = transformer.dwpose_embedding(
                (torch.cat([dwpose_data[:,:,:1].repeat(1,1,3,1,1), dwpose_data], dim=2)
                    ).to(device)).to(model["dtype"])
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
        
        disable_enhance() #not sure if this can work, disabling for now to avoid errors if it's enabled by another sampler

        freqs = None
        transformer.rope_embedder.k = None
        transformer.rope_embedder.num_frames = None
        if rope_function=="comfy":
            transformer.rope_embedder.k = 0
            transformer.rope_embedder.num_frames = latent_video_length
        else:
            d = transformer.dim // transformer.num_heads
            freqs = torch.cat([
                rope_params(1024, d - 4 * (d // 6), L_test=latent_video_length, k=0),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
            dim=1)

        if not isinstance(cfg, list):
            cfg = [cfg] * (steps +1)

        log.info(f"Seq len: {seq_len}")
           
        pbar = ProgressBar(steps)

        if args.preview_method in [LatentPreviewMethod.Auto, LatentPreviewMethod.Latent2RGB]: #default for latent2rgb
            from latent_preview import prepare_callback
        else:
            from ..latent_preview import prepare_callback #custom for tiny VAE previews
        callback = prepare_callback(patcher, steps)

        #blockswap init        
        transformer_options = patcher.model_options.get("transformer_options", None)
        if transformer_options is not None:
            block_swap_args = transformer_options.get("block_swap_args", None)

        if block_swap_args is not None:
            transformer.use_non_blocking = block_swap_args.get("use_non_blocking", True)
            for name, param in transformer.named_parameters():
                if "block" not in name:
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
        transformer.enable_teacache = transformer.enable_magcache = False
        if teacache_args is not None: #for backward compatibility on old workflows
            cache_args = teacache_args
        if cache_args is not None:            
            transformer.cache_device = cache_args["cache_device"]
            if cache_args["cache_type"] == "TeaCache":
                log.info(f"TeaCache: Using cache device: {transformer.cache_device}")
                transformer.teacache_state.clear_all()
                transformer.enable_teacache = True
                transformer.rel_l1_thresh = cache_args["rel_l1_thresh"]
                transformer.teacache_start_step = cache_args["start_step"]
                transformer.teacache_end_step = len(init_timesteps)-1 if cache_args["end_step"] == -1 else cache_args["end_step"]
                transformer.teacache_use_coefficients = cache_args["use_coefficients"]
                transformer.teacache_mode = cache_args["mode"]
            elif cache_args["cache_type"] == "MagCache":
                log.info(f"MagCache: Using cache device: {transformer.cache_device}")
                transformer.magcache_state.clear_all()
                transformer.enable_magcache = True
                transformer.magcache_start_step = cache_args["start_step"]
                transformer.magcache_end_step = len(init_timesteps)-1 if cache_args["end_step"] == -1 else cache_args["end_step"]
                transformer.magcache_thresh = cache_args["magcache_thresh"]
                transformer.magcache_K = cache_args["magcache_K"]

        if slg_args is not None:
            transformer.slg_blocks = slg_args["blocks"]
            transformer.slg_start_percent = slg_args["start_percent"]
            transformer.slg_end_percent = slg_args["end_percent"]
        else:
            transformer.slg_blocks = None

        self.teacache_state = [None, None]
        self.teacache_state_source = [None, None]
        self.teacache_states_context = []


        use_cfg_zero_star, use_fresca = False, False
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
                             vace_data=None, unianim_data=None, teacache_state=None):
            with torch.autocast(device_type=mm.get_autocast_device(device), dtype=dtype, enabled=("fp8" in model["quantization"])):

                if use_cfg_zero_star and (idx <= zero_star_steps) and use_zero_init:
                    return latent_model_input*0, None

                nonlocal patcher
                current_step_percentage = idx / len(init_timesteps)
                control_lora_enabled = False
                
                image_cond_input = image_cond
    
                base_params = {
                    'seq_len': seq_len,
                    'device': device,
                    'freqs': freqs,
                    't': timestep,
                    'current_step': idx,
                    'control_lora_enabled': control_lora_enabled,
                    'vace_data': vace_data,
                    'unianim_data': unianim_data,
                    'fps_embeds': fps_embeds,
                    "nag_params": text_embeds.get("nag_params", {}),
                    "nag_context": text_embeds.get("nag_prompt_embeds", None),
                }

                batch_size = 1

                if not math.isclose(cfg_scale, 1.0) and len(positive_embeds) > 1:
                    negative_embeds = negative_embeds * len(positive_embeds)

                
                #cond
                noise_pred_cond, teacache_state_cond = transformer(
                    [z], context=positive_embeds, y=[image_cond_input] if image_cond_input is not None else None,
                    clip_fea=clip_fea, is_uncond=False, current_step_percentage=current_step_percentage,
                    pred_id=teacache_state[0] if teacache_state else None,
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
                    return noise_pred_cond, [teacache_state_cond]
                #uncond
                noise_pred_uncond, teacache_state_uncond = transformer(
                    [z], context=negative_embeds, clip_fea=clip_fea_neg if clip_fea_neg is not None else clip_fea,
                    y=[image_cond_input] if image_cond_input is not None else None, 
                    is_uncond=True, current_step_percentage=current_step_percentage,
                    pred_id=teacache_state[1] if teacache_state else None,
                    **base_params
                )
                noise_pred_uncond = noise_pred_uncond[0].to(intermediate_device)
            
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
                

                return noise_pred, [teacache_state_cond, teacache_state_uncond]

        log.info(f"Sampling {(latent_video_length-1) * 4 + 1} frames at {latents.shape[3]*8}x{latents.shape[2]*8} with {steps} steps")

        intermediate_device = device

        #clear memory before sampling
        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass

        #region main loop start
        for i, timestep_i in enumerate(tqdm(step_matrix)):
            update_mask_i = step_update_mask[i]
            valid_interval_i = valid_interval[i]
            valid_interval_start, valid_interval_end = valid_interval_i
            timestep = timestep_i[None, valid_interval_start:valid_interval_end].clone()
            latent_model_input = latents[:, valid_interval_start:valid_interval_end, :, :].clone()
            if addnoise_condition > 0 and valid_interval_start < prefix_video_latent_length:
                noise_factor = 0.001 * addnoise_condition
                timestep_for_noised_condition = addnoise_condition
                latent_model_input[:, valid_interval_start:prefix_video_latent_length] = (
                    latent_model_input[:, valid_interval_start:prefix_video_latent_length] * (1.0 - noise_factor)
                    + torch.randn_like(latent_model_input[:, valid_interval_start:prefix_video_latent_length])
                    * noise_factor
                )
                timestep[:, valid_interval_start:prefix_video_latent_length] = timestep_for_noised_condition


            #print("timestep", timestep)
            noise_pred, self.teacache_state = predict_with_cfg(
                latent_model_input.to(dtype), 
                cfg[i], 
                text_embeds["prompt_embeds"], 
                text_embeds["negative_prompt_embeds"], 
                timestep, i, image_cond, clip_fea, unianim_data=unianim_data, vace_data=vace_data,
                teacache_state=self.teacache_state)
            
            for idx in range(valid_interval_start, valid_interval_end):
                if update_mask_i[idx].item():
                    latents[:, idx] = sample_schedulers[idx].step(
                        noise_pred[:, idx - valid_interval_start],
                        timestep_i[idx],
                        latents[:, idx],
                        return_dict=False,
                        generator=seed_g,
                    )[0]
                    sample_schedulers_counter[idx] += 1

            x0 = latents.unsqueeze(0)
            if callback is not None:
                callback_latent = (latent_model_input - noise_pred.to(timestep_i[idx].device) * timestep_i[idx] / 1000).detach().permute(1,0,2,3)
                callback(i, callback_latent, None, steps)
            else:
                pbar.update(1)

        if teacache_args is not None:
            states = transformer.teacache_state.states
            state_names = {
                0: "conditional",
                1: "unconditional"
            }
            for pred_id, state in states.items():
                name = state_names.get(pred_id, f"prediction_{pred_id}")
                if 'skipped_steps' in state:
                    log.info(f"TeaCache skipped: {len(state['skipped_steps'])} {name} steps: {state['skipped_steps']}")
            transformer.teacache_state.clear_all()

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

        return ({
            "samples": x0.cpu(),
            }, )

NODE_CLASS_MAPPINGS = {
    "WanVideoDiffusionForcingSampler": WanVideoDiffusionForcingSampler,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoDiffusionForcingSampler": "WanVideo Diffusion Forcing Sampler",
    }
