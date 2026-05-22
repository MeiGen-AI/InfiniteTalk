# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import json
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image
import subprocess

import wan
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import str2bool, is_video, split_wav_librosa
from wan.utils.multitalk_utils import save_video_ffmpeg
from kokoro import KPipeline
from transformers import Wav2Vec2FeatureExtractor
from src.audio_analysis.wav2vec2 import Wav2Vec2Model
from wan.utils.segvideo import shot_detect


import librosa
import pyloudnorm as pyln
import numpy as np
from einops import rearrange
import soundfile as sf
import re


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 40

    if args.sample_shift is None:
        if args.size == 'infinitetalk-480':
            args.sample_shift = 7
        elif args.size == 'infinitetalk-720':
            args.sample_shift = 11
        else:
            raise NotImplementedError(f'Not supported size')

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, 99999999)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="infinitetalk-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="infinitetalk-480",
        choices=list(SIZE_CONFIGS.keys()),
        help="The buckget size of the generated video. The aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=81,
        help="How many frames to be generated in one clip. The number should be 4n+1"
    )
    parser.add_argument(
        "--max_frame_num",
        type=int,
        default=1000,
        help="The max frame lenght of the generated video."
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the Wan checkpoint directory.")
    parser.add_argument(
        "--infinitetalk_dir",
        type=str,
        default=None,
        help="The path to the InfiniteTalk checkpoint directory.")
    parser.add_argument(
        "--quant_dir",
        type=str,
        default=None,
        help="The path to the Wan quant checkpoint directory.")
    parser.add_argument(
        "--wav2vec_dir",
        type=str,
        default=None,
        help="The path to the wav2vec checkpoint directory.")
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="The path to the Wan checkpoint directory.")
    parser.add_argument(
        "--lora_dir",
        type=str,
        nargs='+',
        default=None,
        help="The paths to the LoRA checkpoint files."
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        nargs='+',
        default=[1.2],
        help="Controls how much to influence the outputs with the LoRA parameters. Accepts multiple float values."
    )
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--audio_save_dir",
        type=str,
        default='save_audio',
        help="The path to save the audio embedding.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--input_json",
        type=str,
        default='examples.json',
        help="[meta file] The condition path to generate the video.")
    parser.add_argument(
        "--motion_frame",
        type=int,
        default=9,
        help="Driven frame length used in the mode of long video genration.")
    parser.add_argument(
        "--mode",
        type=str,
        default="clip",
        choices=['clip', 'streaming'],
        help="clip: generate one video chunk, streaming: long video generation")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_text_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale for text control.")
    parser.add_argument(
        "--sample_audio_guide_scale",
        type=float,
        default=4.0,
        help="Classifier free guidance scale for audio control.")
    parser.add_argument(
        "--num_persistent_param_in_dit",
        type=int,
        default=None,
        required=False,
        help="Maximum parameter quantity retained in video memory, small number to reduce VRAM required",
    )
    parser.add_argument(
        "--audio_mode",
        type=str,
        default="localfile",
        choices=['localfile', 'tts'],
        help="localfile: audio from local wav file, tts: audio from TTS")
    parser.add_argument(
        "--use_teacache",
        action="store_true",
        default=False,
        help="Enable teacache for video generation."
    )
    parser.add_argument(
        "--teacache_thresh",
        type=float,
        default=0.2,
        help="Threshold for teacache."
    )
    parser.add_argument(
        "--use_apg",
        action="store_true",
        default=False,
        help="Enable adaptive projected guidance for video generation (APG)."
    )
    parser.add_argument(
        "--apg_momentum",
        type=float,
        default=-0.75,
        help="Momentum used in adaptive projected guidance (APG)."
    )
    parser.add_argument(
        "--apg_norm_threshold",
        type=float,
        default=55,
        help="Norm threshold used in adaptive projected guidance (APG)."
    )
    parser.add_argument(
        "--color_correction_strength",
        type=float,
        default=1.0,
        help="strength for color correction [0.0 -- 1.0]."
    )
    parser.add_argument(
        "--scene_seg",
        action="store_true",
        default=False,
        help="Enable scene segmentation for input video."
    )
    parser.add_argument(
        "--quant",
        type=str,
        default=None,
        help="Quantization type, must be 'int8' or 'fp8'."
    )
    
    args = parser.parse_args()

    _validate_args(args)

    return args

def custom_init(device, wav2vec):    
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True).to(device)
    audio_encoder.feature_extractor._freeze_parameters()
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec, local_files_only=True)
    return wav2vec_feature_extractor, audio_encoder

def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio

def audio_prepare_multi(left_path, right_path, audio_type, sample_rate=16000):

    if not (left_path=='None' or right_path=='None'):
        human_speech_array1 = audio_prepare_single(left_path)
        human_speech_array2 = audio_prepare_single(right_path)
    elif left_path=='None':
        human_speech_array2 = audio_prepare_single(right_path)
        human_speech_array1 = np.zeros(human_speech_array2.shape[0])
    elif right_path=='None':
        human_speech_array1 = audio_prepare_single(left_path)
        human_speech_array2 = np.zeros(human_speech_array1.shape[0])

    if audio_type=='para':
        new_human_speech1 = human_speech_array1
        new_human_speech2 = human_speech_array2
    elif audio_type=='add':
        new_human_speech1 = np.concatenate([human_speech_array1[: human_speech_array1.shape[0]], np.zeros(human_speech_array2.shape[0])]) 
        new_human_speech2 = np.concatenate([np.zeros(human_speech_array1.shape[0]), human_speech_array2[:human_speech_array2.shape[0]]])
    sum_human_speechs = new_human_speech1 + new_human_speech2
    return new_human_speech1, new_human_speech2, sum_human_speechs

def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device='cpu'):
    audio_duration = len(speech_array) / sr
    video_length = audio_duration * 25 # Assume the video fps is 25

    # wav2vec_feature_extractor
    audio_feature = np.squeeze(
        wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
    audio_feature = audio_feature.unsqueeze(0)

    # audio encoder
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

    if len(embeddings) == 0:
        print("Fail to extract audio embedding")
        return None

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")

    audio_emb = audio_emb.cpu().detach()
    return audio_emb

def extract_audio_from_video(filename, sample_rate):
    raw_audio_path = filename.split('/')[-1].split('.')[0]+'.wav'
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i",
        str(filename),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "2",
        str(raw_audio_path),
    ]
    subprocess.run(ffmpeg_command, check=True)
    human_speech_array, sr = librosa.load(raw_audio_path, sr=sample_rate)
    human_speech_array = loudness_norm(human_speech_array, sr)
    os.remove(raw_audio_path)

    return human_speech_array

def audio_prepare_single(audio_path, sample_rate=16000):
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in ['.mp4', '.mov', '.avi', '.mkv']:
        human_speech_array = extract_audio_from_video(audio_path, sample_rate)
        return human_speech_array
    else:
        human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
        human_speech_array = loudness_norm(human_speech_array, sr)
        return human_speech_array

def process_tts_single(text, save_dir, voice1):    
    s1_sentences = []

    pipeline = KPipeline(lang_code='a', repo_id='weights/Kokoro-82M')

    voice_tensor = torch.load(voice1, weights_only=True)
    generator = pipeline(
        text, voice=voice_tensor, # <= change voice here
        speed=1, split_pattern=r'\n+'
    )
    audios = []
    for i, (gs, ps, audio) in enumerate(generator):
        audios.append(audio)
    audios = torch.concat(audios, dim=0)
    s1_sentences.append(audios)
    s1_sentences = torch.concat(s1_sentences, dim=0)
    save_path1 =f'{save_dir}/s1.wav'
    sf.write(save_path1, s1_sentences, 24000) # save each audio file
    s1, _ = librosa.load(save_path1, sr=16000)
    return s1, save_path1
    
   

def process_tts_multi(text, save_dir, voice1, voice2):
    pattern = r'\(s(\d+)\)\s*(.*?)(?=\s*\(s\d+\)|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    s1_sentences = []
    s2_sentences = []

    pipeline = KPipeline(lang_code='a', repo_id='weights/Kokoro-82M')
    for idx, (speaker, content) in enumerate(matches):
        if speaker == '1':
            voice_tensor = torch.load(voice1, weights_only=True)
            generator = pipeline(
                content, voice=voice_tensor, # <= change voice here
                speed=1, split_pattern=r'\n+'
            )
            audios = []
            for i, (gs, ps, audio) in enumerate(generator):
                audios.append(audio)
            audios = torch.concat(audios, dim=0)
            s1_sentences.append(audios)
            s2_sentences.append(torch.zeros_like(audios))
        elif speaker == '2':
            voice_tensor = torch.load(voice2, weights_only=True)
            generator = pipeline(
                content, voice=voice_tensor, # <= change voice here
                speed=1, split_pattern=r'\n+'
            )
            audios = []
            for i, (gs, ps, audio) in enumerate(generator):
                audios.append(audio)
            audios = torch.concat(audios, dim=0)
            s2_sentences.append(audios)
            s1_sentences.append(torch.zeros_like(audios))
    
    s1_sentences = torch.concat(s1_sentences, dim=0)
    s2_sentences = torch.concat(s2_sentences, dim=0)
    sum_sentences = s1_sentences + s2_sentences
    save_path1 =f'{save_dir}/s1.wav'
    save_path2 =f'{save_dir}/s2.wav'
    save_path_sum = f'{save_dir}/sum.wav'
    sf.write(save_path1, s1_sentences, 24000) # save each audio file
    sf.write(save_path2, s2_sentences, 24000)
    sf.write(save_path_sum, sum_sentences, 24000)

    s1, _ = librosa.load(save_path1, sr=16000)
    s2, _ = librosa.load(save_path2, sr=16000)
    # sum, _ = librosa.load(save_path_sum, sr=16000)
    return s1, s2, save_path_sum

def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    # TODO: use prompt refine
    # if args.use_prompt_extend:
    #     if args.prompt_extend_method == "dashscope":
    #         prompt_expander = DashScopePromptExpander(
    #             model_name=args.prompt_extend_model,
    #             is_vl="i2v" in args.task or "flf2v" in args.task)
    #     elif args.prompt_extend_method == "local_qwen":
    #         prompt_expander = QwenPromptExpander(
    #             model_name=args.prompt_extend_model,
    #             is_vl="i2v" in args.task,
    #             device=rank)
    #     else:
    #         raise NotImplementedError(
    #             f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    assert args.task == "infinitetalk-14B", 'You should choose infinitetalk in args.task.'
    

    logging.info("Creating infinitetalk pipeline.")
    wan_i2v = wan.InfiniteTalkPipeline(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        quant_dir=args.quant_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp, 
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),  
        t5_cpu=args.t5_cpu,
        lora_dir=args.lora_dir,
        lora_scales=args.lora_scale,
        quant=args.quant,
        dit_path=args.dit_path,
        infinitetalk_dir=args.infinitetalk_dir
    )
    if args.num_persistent_param_in_dit is not None:
        wan_i2v.vram_management = True
        wan_i2v.enable_vram_management(
            num_persistent_param_in_dit=args.num_persistent_param_in_dit
        )
    
    generated_list = []
    with open(args.input_json, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
        
    wav2vec_feature_extractor, audio_encoder= custom_init('cpu', args.wav2vec_dir)
    args.audio_save_dir = os.path.join(args.audio_save_dir, input_data['cond_video'].split('/')[-1].split('.')[0])
    os.makedirs(args.audio_save_dir,exist_ok=True)
    
    conds_list = []

    if args.scene_seg and is_video(input_data['cond_video']):
        time_list, cond_list = shot_detect(input_data['cond_video'], args.audio_save_dir)
        if len(time_list)==0:
            conds_list.append([input_data['cond_video']])
            conds_list.append([input_data['cond_audio']['person1']])
            if len(input_data['cond_audio'])==2:
                conds_list.append([input_data['cond_audio']['person2']])
        else:
            audio1_list = split_wav_librosa(input_data['cond_audio']['person1'], time_list, args.audio_save_dir)
            conds_list.append(cond_list)
            conds_list.append(audio1_list)
            if len(input_data['cond_audio'])==2:
                audio2_list = split_wav_librosa(input_data['cond_audio']['person2'], time_list, args.audio_save_dir)
                conds_list.append(audio2_list)
    else:
        conds_list.append([input_data['cond_video']])
        conds_list.append([input_data['cond_audio']['person1']])
        if len(input_data['cond_audio'])==2:
            conds_list.append([input_data['cond_audio']['person2']])

    if len(input_data['cond_audio'])==2:
        new_human_speech1, new_human_speech2, sum_human_speechs = audio_prepare_multi(input_data['cond_audio']['person1'], input_data['cond_audio']['person2'], input_data['audio_type'])
        sum_audio = os.path.join(args.audio_save_dir, 'sum_all.wav')
        sf.write(sum_audio, sum_human_speechs, 16000)
        input_data['video_audio'] = sum_audio
    else:
        human_speech = audio_prepare_single(input_data['cond_audio']['person1'])
        sum_audio = os.path.join(args.audio_save_dir, 'sum_all.wav')
        sf.write(sum_audio, human_speech, 16000)
        input_data['video_audio'] = sum_audio
    logging.info("Generating video ...")
        
    for idx, items in enumerate(zip(*conds_list)):
        print(items)
        input_clip = {}
        input_clip['prompt'] = input_data['prompt']
        input_clip['cond_video'] = items[0]

        if 'audio_type' in input_data:
            input_clip['audio_type'] = input_data['audio_type']
        if 'bbox' in input_data:
            input_clip['bbox'] = input_data['bbox']
        cond_audio = {}
        if args.audio_mode=='localfile':
            if len(input_data['cond_audio'])==2:
                new_human_speech1, new_human_speech2, sum_human_speechs = audio_prepare_multi(items[1], items[2], input_data['audio_type'])
                audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder)
                audio_embedding_2 = get_embedding(new_human_speech2, wav2vec_feature_extractor, audio_encoder)
                emb1_path = os.path.join(args.audio_save_dir, '1.pt')
                emb2_path = os.path.join(args.audio_save_dir, '2.pt')
                sum_audio = os.path.join(args.audio_save_dir, 'sum.wav')
                sf.write(sum_audio, sum_human_speechs, 16000)
                if rank == 0:
                    torch.save(audio_embedding_1, emb1_path)
                    torch.save(audio_embedding_2, emb2_path)
                cond_audio['person1'] = emb1_path
                cond_audio['person2'] = emb2_path
                input_clip['video_audio'] = sum_audio
                v_length = audio_embedding_1.shape[0]
            elif len(input_data['cond_audio'])==1:
                human_speech = audio_prepare_single(items[1])
                audio_embedding = get_embedding(human_speech, wav2vec_feature_extractor, audio_encoder)
                emb_path = os.path.join(args.audio_save_dir, '1.pt')
                sum_audio = os.path.join(args.audio_save_dir, 'sum.wav')
                sf.write(sum_audio, human_speech, 16000)
                if rank == 0:
                    torch.save(audio_embedding, emb_path)
                cond_audio['person1'] = emb_path
                input_clip['video_audio'] = sum_audio
                v_length = audio_embedding.shape[0]
        
        input_clip['cond_audio'] = cond_audio
                    
        video = wan_i2v.generate_infinitetalk(
            input_clip,
            size_buckget=args.size,
            motion_frame=args.motion_frame,
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sampling_steps=args.sample_steps,
            text_guide_scale=args.sample_text_guide_scale,
            audio_guide_scale=args.sample_audio_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            max_frames_num=args.frame_num if args.mode == 'clip' else args.max_frame_num,
            color_correction_strength = args.color_correction_strength,
            extra_args=args,
            )
        
        generated_list.append(video)

    if rank == 0:
        
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = input_clip['prompt'].replace(" ", "_").replace("/",
                                                                        "_")[:50]
            args.save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}"
        
        sum_video = torch.cat(generated_list, dim=1)
        save_video_ffmpeg(sum_video, args.save_file, [input_data['video_audio']], high_quality_save=False)
   
    logging.info(f"Saving generated video to {args.save_file}.mp4")  
    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
