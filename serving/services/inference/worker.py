import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from ..distributed_utils import DistributedManager
from .infinitetalk_audio import audio_prepare_single, custom_init, get_embedding, save_wav_16k
from ...utils.timing import timed_stage, timing_enabled, timing_summary_lines


def _round_to_4n_plus_1(frames: int, min_frames: int = 5) -> int:
    frames = max(frames, min_frames)
    # nearest <= frames that satisfies 4n+1
    return (max((frames - 1) // 4, 1) * 4) + 1


def _default_shift_for_size(size: str | None) -> float | None:
    if size == "infinitetalk-480":
        return 7.0
    if size == "infinitetalk-720":
        return 11.0
    return None


class TorchrunInferenceWorker:
    def __init__(self):
        self.rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.dist_manager = DistributedManager()
        self.processing = False

        self.pipeline = None
        self.wav2vec_feature_extractor = None
        self.audio_encoder = None

        # defaults (can be overridden by request fields)
        self.default_size = "infinitetalk-480"
        self.default_motion_frame = 9
        self.default_sample_shift = None
        self.default_text_guide_scale = 5.0
        self.default_audio_guide_scale = 4.0
        self.default_offload_model = True
        self.default_max_frame_num = 1000
        self.default_use_teacache = False
        self.default_teacache_thresh = 0.2

    def init(self, args) -> bool:
        try:
            t0 = time.perf_counter()
            self._timing_on = timing_enabled(args)
            try:
                import torch  # local import for better startup ergonomics
            except ModuleNotFoundError as e:
                raise RuntimeError("Missing dependency: torch. Please install PyTorch before starting serving.") from e

            import wan  # local import (requires project deps)
            from wan.configs import WAN_CONFIGS

            # Ensure Wan's internal `logging.info(...)` is visible.
            import logging

            if not logging.getLogger().handlers:
                logging.basicConfig(
                    level=logging.INFO,
                    format=f"[%(asctime)s][rank={self.rank}] %(levelname)s: %(message)s",
                )

            if self.world_size > 1:
                logger.info(f"Rank {self.rank}: [1/4] 初始化分布式进程组 ...")
                if not self.dist_manager.init_process_group():
                    raise RuntimeError("Failed to initialize distributed process group")
                logger.info(f"Rank {self.rank}: [1/4] 分布式初始化完成，用时 {time.perf_counter() - t0:.1f}s")
            else:
                self.dist_manager.rank = 0
                self.dist_manager.world_size = 1
                self.dist_manager.device = "cuda:0" if torch.cuda.is_available() else "cpu"
                self.dist_manager.is_initialized = False

            # Parse required model args
            ckpt_dir = getattr(args, "ckpt_dir", None)
            wav2vec_dir = getattr(args, "wav2vec_dir", None)
            infinitetalk_dir = getattr(args, "infinitetalk_dir", None)
            quant_dir = getattr(args, "quant_dir", None)
            quant = getattr(args, "quant", None)
            dit_path = getattr(args, "dit_path", None)
            ulysses_size = int(getattr(args, "ulysses_size", 1) or 1)
            ring_size = int(getattr(args, "ring_size", 1) or 1)

            if not ckpt_dir or not wav2vec_dir or not infinitetalk_dir:
                raise RuntimeError("--ckpt_dir, --wav2vec_dir, --infinitetalk_dir are required")

            # Initialize xfuser model parallel groups when USP (ulysses/ring) is enabled.
            # This mirrors `generate_infinitetalk.py` behavior and avoids:
            # AssertionError: pipeline model parallel group is not initialized
            if ulysses_size > 1 or ring_size > 1:
                if self.world_size <= 1:
                    raise RuntimeError("ulysses_size/ring_size > 1 requires torchrun (WORLD_SIZE > 1)")
                if ulysses_size * ring_size != self.world_size:
                    raise RuntimeError(
                        f"ulysses_size*ring_size must equal WORLD_SIZE. "
                        f"Got {ulysses_size=} {ring_size=} {self.world_size=}"
                    )

                t_mp = time.perf_counter()
                logger.info(
                    f"Rank {self.rank}: 初始化 xfuser model-parallel groups "
                    f"(ulysses_size={ulysses_size}, ring_size={ring_size}) ..."
                )
                import torch.distributed as dist
                from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel

                init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
                initialize_model_parallel(
                    sequence_parallel_degree=dist.get_world_size(),
                    ring_degree=ring_size,
                    ulysses_degree=ulysses_size,
                )
                logger.info(f"Rank {self.rank}: xfuser 并行组初始化完成，用时 {time.perf_counter() - t_mp:.1f}s")

            self.default_size = getattr(args, "size", self.default_size) or self.default_size
            self.default_motion_frame = int(getattr(args, "motion_frame", self.default_motion_frame) or self.default_motion_frame)
            self.default_text_guide_scale = float(getattr(args, "sample_text_guide_scale", self.default_text_guide_scale) or self.default_text_guide_scale)
            self.default_audio_guide_scale = float(getattr(args, "sample_audio_guide_scale", self.default_audio_guide_scale) or self.default_audio_guide_scale)
            self.default_offload_model = bool(getattr(args, "offload_model", self.default_offload_model))
            self.default_max_frame_num = int(getattr(args, "max_frame_num", self.default_max_frame_num) or self.default_max_frame_num)
            self.default_use_teacache = bool(getattr(args, "use_teacache", self.default_use_teacache))
            self.default_teacache_thresh = float(getattr(args, "teacache_thresh", self.default_teacache_thresh))

            # shift default depends on size if not specified
            self.default_sample_shift = getattr(args, "sample_shift", None)
            if self.default_sample_shift is None:
                if self.default_size == "infinitetalk-480":
                    self.default_sample_shift = 7
                elif self.default_size == "infinitetalk-720":
                    self.default_sample_shift = 11

            cfg = WAN_CONFIGS["infinitetalk-14B"]

            # Init wav2vec on CPU (shared for all requests)
            t_w2v = time.perf_counter()
            logger.info(f"Rank {self.rank}: [2/4] 加载 wav2vec 到 CPU ...")
            self.wav2vec_feature_extractor, self.audio_encoder = custom_init("cpu", wav2vec_dir)
            logger.info(f"Rank {self.rank}: [2/4] wav2vec 加载完成，用时 {time.perf_counter() - t_w2v:.1f}s")

            # Init InfiniteTalk pipeline (GPU)
            t_pipe = time.perf_counter()
            logger.info(
                f"Rank {self.rank}: [3/4] 加载 InfiniteTalk 主模型权重到 GPU（最耗时阶段）..."
            )
            device_id = self.rank if torch.cuda.is_available() else 0
            self.pipeline = wan.InfiniteTalkPipeline(
                config=cfg,
                checkpoint_dir=ckpt_dir,
                quant_dir=quant_dir,
                device_id=device_id,
                rank=self.rank,
                t5_fsdp=getattr(args, "t5_fsdp", False),
                dit_fsdp=getattr(args, "dit_fsdp", False),
                use_usp=bool(ulysses_size > 1 or ring_size > 1),
                t5_cpu=getattr(args, "t5_cpu", False),
                lora_dir=getattr(args, "lora_dir", None),
                lora_scales=getattr(args, "lora_scale", None),
                quant=quant,
                dit_path=dit_path,
                infinitetalk_dir=infinitetalk_dir,
                enable_torch_compile=getattr(args, "enable_torch_compile", False),
            )
            logger.info(f"Rank {self.rank}: [3/4] 主模型加载完成，用时 {time.perf_counter() - t_pipe:.1f}s")

            num_persistent = getattr(args, "num_persistent_param_in_dit", None)
            if num_persistent is not None:
                t_vram = time.perf_counter()
                logger.info(f"Rank {self.rank}: [4/4] 启用 VRAM 管理 ...")
                self.pipeline.vram_management = True
                self.pipeline.enable_vram_management(num_persistent_param_in_dit=num_persistent)
                logger.info(f"Rank {self.rank}: [4/4] VRAM 管理启用完成，用时 {time.perf_counter() - t_vram:.1f}s")

            logger.info(
                f"Rank {self.rank}/{self.world_size - 1} initialization completed, total {time.perf_counter() - t0:.1f}s"
            )
            return True

        except Exception as e:
            logger.exception(f"Rank {self.rank} initialization failed: {str(e)}")
            return False

    async def process_request(self, task_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        has_error = False
        error_msg = ""

        try:
            import torch  # local import
            from wan.utils.multitalk_utils import save_video_ffmpeg

            if self.world_size > 1 and self.rank == 0:
                task_data = self.dist_manager.broadcast_task_data(task_data)

            # Extract inputs (LightX2V-compatible fields)
            task_id = task_data["task_id"]
            prompt = task_data.get("prompt", "")
            cond_media_path = task_data.get("image_path", "")
            audio_path = task_data.get("audio_path", "")
            save_result_path = task_data.get("save_result_path", "")

            if not cond_media_path:
                raise RuntimeError("image_path is required (can be image or video)")
            if not audio_path:
                raise RuntimeError("audio_path is required for InfiniteTalk")
            if not save_result_path:
                raise RuntimeError("save_result_path is required")

            # Map controls
            sampling_steps = int(task_data.get("infer_steps", 40))
            seed = int(task_data.get("seed", 42))
            fps = int(task_data.get("target_fps", 25) or 25)

            # InfiniteTalk-specific overrides (optional)
            size = task_data.get("size") or self.default_size
            motion_frame = int(task_data.get("motion_frame") or self.default_motion_frame)
            # sample_shift: request-body override first; otherwise default follows requested size; else fall back to server default.
            if "sample_shift" in task_data and task_data.get("sample_shift") is not None:
                shift = float(task_data["sample_shift"])
            else:
                shift = float(_default_shift_for_size(size) or self.default_sample_shift)
            text_guide_scale = float(task_data.get("text_guide_scale") or self.default_text_guide_scale)
            audio_guide_scale = float(task_data.get("audio_guide_scale") or self.default_audio_guide_scale)
            # NOTE:
            # - `max_frame_num` is the ONLY supported knob for output length.
            # - The actual produced frames will be capped by audio length internally (25 fps default).
            max_frames_num = int(task_data.get("max_frame_num") or self.default_max_frame_num)

            # Decide first chunk length (frame_num) and streaming total length (max_frames_num).
            # InfiniteTalk's internal chunk is typically 81 frames; we keep it fixed here.
            frame_num = _round_to_4n_plus_1(min(81, max_frames_num))
            mode_streaming = max_frames_num > frame_num

            output_path = Path(save_result_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_base = str(output_path.with_suffix(""))  # save_video_ffmpeg appends .mp4

            work_dir = output_path.parent / f"{task_id}_work"
            work_dir.mkdir(parents=True, exist_ok=True)

            timing_on = bool(getattr(self, "_timing_on", False)) and self.rank == 0
            timing_stats = {} if timing_on else None

            # Prepare audio embedding (CPU)
            with timed_stage("audio_prepare_single()", enabled=timing_on, stats=timing_stats):
                speech = audio_prepare_single(audio_path, sample_rate=16000, tmp_dir=str(work_dir))
            with timed_stage("wav2vec get_embedding()", enabled=timing_on, stats=timing_stats):
                audio_emb = get_embedding(
                    speech, self.wav2vec_feature_extractor, self.audio_encoder, sr=16000, device="cpu"
                )

            emb_path = str(work_dir / "1.pt")
            with timed_stage("torch.save(audio_emb)", enabled=timing_on, stats=timing_stats):
                torch.save(audio_emb, emb_path)

            # Also ensure we have a wav for mux/cropping
            wav_path = str(work_dir / "audio_16k.wav")
            with timed_stage("save_wav_16k()", enabled=timing_on, stats=timing_stats):
                save_wav_16k(speech, wav_path, sr=16000)

            input_data = {
                "prompt": prompt,
                "cond_video": cond_media_path,
                "cond_audio": {"person1": emb_path},
                "video_audio": wav_path,
            }

            # Run pipeline (GPU heavy)
            extra_args = type("ExtraArgs", (), {})()
            # APG/TeaCache are opt-in; keep off by default.
            extra_args.use_teacache = bool(task_data.get("use_teacache", self.default_use_teacache))
            extra_args.teacache_thresh = float(task_data.get("teacache_thresh", self.default_teacache_thresh))
            extra_args.size = size
            extra_args.use_apg = bool(task_data.get("use_apg", False))
            extra_args.apg_momentum = float(task_data.get("apg_momentum", -0.75))
            extra_args.apg_norm_threshold = float(task_data.get("apg_norm_threshold", 55))
            extra_args.enable_torch_deterministic = bool(task_data.get("enable_torch_deterministic", False))

            with timed_stage("pipeline.generate_infinitetalk()", enabled=timing_on, stats=timing_stats, key="generate_infinitetalk"):
                video_tensor = self.pipeline.generate_infinitetalk(
                    input_data,
                    size_buckget=size,
                    motion_frame=motion_frame,
                    frame_num=frame_num,
                    shift=shift,
                    sampling_steps=sampling_steps,
                    text_guide_scale=text_guide_scale,
                    audio_guide_scale=audio_guide_scale,
                    seed=seed,
                    offload_model=self.default_offload_model,
                    max_frames_num=max_frames_num if mode_streaming else frame_num,
                    color_correction_strength=float(task_data.get("color_correction_strength", 0.0)),
                    extra_args=extra_args,
                )

            if self.world_size > 1:
                self.dist_manager.barrier()

            if self.rank == 0:
                # Save mp4 with audio
                with timed_stage("save_video_ffmpeg()", enabled=timing_on, stats=timing_stats):
                    save_video_ffmpeg(video_tensor, output_base, [wav_path], fps=fps, high_quality_save=False)

                if timing_on:
                    for line in timing_summary_lines(timing_stats or {}):
                        print(line, flush=True)
                    timing_path = str(output_path.with_suffix(output_path.suffix + ".timing.json"))
                    try:
                        with open(timing_path, "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "task_id": task_id,
                                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "timing": timing_stats or {},
                                },
                                f,
                                ensure_ascii=False,
                                indent=2,
                            )
                        print(f"[TIMING] wrote: {timing_path}", flush=True)
                    except Exception as e:
                        logger.warning(f"Failed to write timing json: {e}")

        except Exception as e:
            has_error = True
            error_msg = str(e)
            logger.exception(f"Rank {self.rank} inference failed: {error_msg}")

        if self.world_size > 1:
            self.dist_manager.barrier()

        if self.rank == 0:
            if has_error:
                return {"task_id": task_data.get("task_id", "unknown"), "status": "failed", "error": error_msg, "message": f"Inference failed: {error_msg}"}
            return {
                "task_id": task_data["task_id"],
                "status": "success",
                "save_result_path": task_data.get("video_path", Path(task_data["save_result_path"]).name),
                "message": "Inference completed",
            }
        return None

    async def worker_loop(self):
        while True:
            task_data = None
            try:
                task_data = self.dist_manager.broadcast_task_data()
                if task_data is None:
                    logger.info(f"Rank {self.rank} received stop signal")
                    break
                await self.process_request(task_data)
            except Exception as e:
                error_str = str(e)
                if "Connection closed by peer" in error_str or "Connection reset by peer" in error_str:
                    logger.info(f"Rank {self.rank} detected master process shutdown, exiting worker loop")
                    break
                logger.error(f"Rank {self.rank} worker loop error: {error_str}")
                if self.world_size > 1 and task_data is not None:
                    try:
                        self.dist_manager.barrier()
                    except Exception as barrier_error:
                        logger.warning(f"Rank {self.rank} barrier failed, exiting: {barrier_error}")
                        break
                continue

    def cleanup(self):
        self.dist_manager.cleanup()

