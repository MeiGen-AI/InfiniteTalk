# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.nn as nn
from einops import rearrange, repeat
from ..utils.multitalk_utils import RotaryPositionalEmbedding1D, normalize_and_scale, split_token_counts_and_frame_ids
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
import xformers.ops

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import sys
import warnings

__all__ = [
    'flash_attention',
    'attention',
]


_WARN_ONCE_KEYS = set()


def _warn(msg: str):
    print(f"[ATTN][WARN] {msg}", file=sys.stderr, flush=True)


def _warn_once(key: str, msg: str):
    if key in _WARN_ONCE_KEYS:
        return
    _WARN_ONCE_KEYS.add(key)
    # Use direct stderr print to avoid warnings filters / tqdm stderr refresh hiding the message.
    print(f"[ATTN][ONCE] {msg}", file=sys.stderr, flush=True)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    q_in, k_in, v_in = q, k, v

    def _sdpa_fallback():
        # Slower but keeps inference runnable.
        if q_lens is not None or k_lens is not None:
            _warn_once(
                "sdpa_padding_mask_disabled",
                "Padding mask is disabled when using scaled_dot_product_attention fallback. "
                "It can have a significant impact on performance.",
            )

        q_sdpa = q_in.transpose(1, 2).to(dtype)
        k_sdpa = k_in.transpose(1, 2).to(dtype)
        v_sdpa = v_in.transpose(1, 2).to(dtype)
        out = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=None,
            is_causal=causal,
            dropout_p=dropout_p,
        )
        return out.transpose(1, 2).contiguous().to(q_in.dtype)

    def _fa3_varlen(q_packed, k_packed, v_packed, q_lens_local, k_lens_local, lq_local, lk_local, b_local):
        """
        flash_attn_interface.flash_attn_varlen_func wrappers can differ across packages.
        This adapter tries a few common output layouts and reshapes to [B, Lq, H, D].
        """
        cu_q = torch.cat([q_lens_local.new_zeros([1]), q_lens_local]).cumsum(0, dtype=torch.int32).to(
            q_packed.device, non_blocking=True
        )
        cu_k = torch.cat([k_lens_local.new_zeros([1]), k_lens_local]).cumsum(0, dtype=torch.int32).to(
            q_packed.device, non_blocking=True
        )

        out = flash_attn_interface.flash_attn_varlen_func(
            q=q_packed,
            k=k_packed,
            v=v_packed,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq_local,
            max_seqlen_k=lk_local,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )

        x = out[0] if isinstance(out, (tuple, list)) else out

        # Expected "packed" layout is typically [total_q, H, D]
        total_q = int(q_lens_local.sum().item())
        if x.dim() == 4 and x.shape[0] == b_local and x.shape[1] == lq_local:
            return x  # already [B, Lq, H, D]
        if x.dim() == 3 and x.shape[0] == total_q:
            return x.unflatten(0, (b_local, lq_local))
        if x.dim() == 3 and x.shape[1] == total_q:
            # [H, total_q, D] -> [total_q, H, D]
            return x.permute(1, 0, 2).contiguous().unflatten(0, (b_local, lq_local))

        raise RuntimeError(f"Unexpected FA3 varlen output shape: {tuple(x.shape)} (total_q={total_q}, b={b_local}, lq={lq_local})")

    # If a specific flash-attn version is requested but unavailable, degrade gracefully.
    if version == 2 and not FLASH_ATTN_2_AVAILABLE:
        if FLASH_ATTN_3_AVAILABLE:
            _warn_once("fa2_missing_use_fa3", "Flash attention 2 is not available, use flash attention 3 instead.")
            version = 3
        else:
            return _sdpa_fallback()
    if version == 3 and not FLASH_ATTN_3_AVAILABLE:
        if FLASH_ATTN_2_AVAILABLE:
            _warn_once("fa3_missing_use_fa2", "Flash attention 3 is not available, use flash attention 2 instead.")
            version = 2
        else:
            return _sdpa_fallback()
    if version is None and not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        return _sdpa_fallback()

    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    # `version` has been normalized above; keep this branch for safety if called with odd configs.
    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        _warn_once("fa3_missing_safety", "Flash attention 3 is not available, use flash attention 2 instead.")

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        try:
            x = _fa3_varlen(q, k, v, q_lens, k_lens, lq, lk, b)
        except Exception as e:
            _warn_once("fa3_failed_sdpa", f"Flash attention 3 failed, falling back to SDPA: {e}")
            return _sdpa_fallback()
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            _warn_once(
                "sdpa_padding_mask_disabled",
                "Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.",
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
    

class SingleStreamAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        encoder_hidden_states_dim: int,
        num_heads: int,
        qkv_bias: bool,
        qk_norm: bool,
        norm_layer: nn.Module,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.encoder_hidden_states_dim = encoder_hidden_states_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qk_norm = qk_norm

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim,eps=eps) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kv_linear = nn.Linear(encoder_hidden_states_dim, dim * 2, bias=qkv_bias)

        self.add_q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.add_k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

    @torch.compiler.disable
    def forward(self, x: torch.Tensor, encoder_hidden_states: torch.Tensor, shape=None, enable_sp=False, kv_seq=None) -> torch.Tensor:
       
        N_t, N_h, N_w = shape
        if not enable_sp:
            x = rearrange(x, "B (N_t S) C -> (B N_t) S C", N_t=N_t)

        # get q for hidden_state
        B, N, C = x.shape
        q = self.q_linear(x)
        q_shape = (B, N, self.num_heads, self.head_dim)
        q = q.view(q_shape).permute((0, 2, 1, 3))

        if self.qk_norm:
            q = self.q_norm(q)
        
        # get kv from encoder_hidden_states
        _, N_a, _ = encoder_hidden_states.shape
        encoder_kv = self.kv_linear(encoder_hidden_states)
        encoder_kv_shape = (B, N_a, 2, self.num_heads, self.head_dim)
        encoder_kv = encoder_kv.view(encoder_kv_shape).permute((2, 0, 3, 1, 4)) 
        encoder_k, encoder_v = encoder_kv.unbind(0)

        if self.qk_norm:
            encoder_k = self.add_k_norm(encoder_k)


        q = rearrange(q, "B H M K -> B M H K")
        encoder_k = rearrange(encoder_k, "B H M K -> B M H K")
        encoder_v = rearrange(encoder_v, "B H M K -> B M H K")

        if enable_sp:
            # context parallel
            sp_size = get_sequence_parallel_world_size()
            sp_rank = get_sequence_parallel_rank()
            visual_seqlen, _ = split_token_counts_and_frame_ids(N_t, N_h * N_w, sp_size, sp_rank)
            assert kv_seq is not None, f"kv_seq should not be None."
            attn_bias = xformers.ops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(visual_seqlen, kv_seq)
        else:
            attn_bias = None
        try:
            x = xformers.ops.memory_efficient_attention(
                q, encoder_k, encoder_v, attn_bias=attn_bias, op=None
            )
            _warn_once(
                "xformers_mea_ok",
                "xFormers memory_efficient_attention: OK (has been used at least once).",
            )
        except NotImplementedError:
            # xFormers CUDA ops are unavailable in some environments (e.g., wheel built for a different torch/python).
            # Fall back to PyTorch SDPA / flash-attn wrapper.
            if attn_bias is not None:
                _warn_once(
                    "xformers_sp_attn_bias_dropped",
                    "xFormers operator not available; falling back to SDPA without attn_bias. "
                    "This may be incorrect for sequence-parallel mode.",
                )
            _warn_once(
                "xformers_mea_fallback",
                "xFormers memory_efficient_attention: NOT available -> falling back to PyTorch attention.",
            )
            x = attention(q, encoder_k, encoder_v, dropout_p=0.0, causal=False, dtype=q.dtype)
        x = rearrange(x, "B M H K -> B H M K") 

        # linear transform
        x_output_shape = (B, N, C)
        x = x.transpose(1, 2) 
        x = x.reshape(x_output_shape) 
        x = self.proj(x)
        x = self.proj_drop(x)

        if not enable_sp:
            # reshape x to origin shape
            x = rearrange(x, "(B N_t) S C -> B (N_t S) C", N_t=N_t)

        return x

class SingleStreamMutiAttention(SingleStreamAttention):
    def __init__(
        self,
        dim: int,
        encoder_hidden_states_dim: int,
        num_heads: int,
        qkv_bias: bool,
        qk_norm: bool,
        norm_layer: nn.Module,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        eps: float = 1e-6,
        class_range: int = 24,
        class_interval: int = 4,
    ) -> None:
        super().__init__(
            dim=dim,
            encoder_hidden_states_dim=encoder_hidden_states_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            norm_layer=norm_layer,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            eps=eps,
        )
        self.class_interval = class_interval
        self.class_range = class_range
        self.rope_h1  = (0, self.class_interval)
        self.rope_h2  = (self.class_range - self.class_interval, self.class_range)
        self.rope_bak = int(self.class_range // 2)

        self.rope_1d = RotaryPositionalEmbedding1D(self.head_dim)

    @torch.compiler.disable
    def forward(self, 
                x: torch.Tensor, 
                encoder_hidden_states: torch.Tensor, 
                shape=None, 
                x_ref_attn_map=None,
                human_num=None) -> torch.Tensor:
        
        encoder_hidden_states = encoder_hidden_states.squeeze(0)
        if human_num == 1:
            return super().forward(x, encoder_hidden_states, shape)

        N_t, _, _ = shape 
        x = rearrange(x, "B (N_t S) C -> (B N_t) S C", N_t=N_t) 

        # get q for hidden_state
        B, N, C = x.shape
        q = self.q_linear(x) 
        q_shape = (B, N, self.num_heads, self.head_dim) 
        q = q.view(q_shape).permute((0, 2, 1, 3))

        if self.qk_norm:
            q = self.q_norm(q)

  
        max_values = x_ref_attn_map.max(1).values[:, None, None] 
        min_values = x_ref_attn_map.min(1).values[:, None, None] 
        max_min_values = torch.cat([max_values, min_values], dim=2)

        human1_max_value, human1_min_value = max_min_values[0, :, 0].max(), max_min_values[0, :, 1].min()
        human2_max_value, human2_min_value = max_min_values[1, :, 0].max(), max_min_values[1, :, 1].min()

        human1 = normalize_and_scale(x_ref_attn_map[0], (human1_min_value, human1_max_value), (self.rope_h1[0], self.rope_h1[1]))
        human2 = normalize_and_scale(x_ref_attn_map[1], (human2_min_value, human2_max_value), (self.rope_h2[0], self.rope_h2[1]))
        back   = torch.full((x_ref_attn_map.size(1),), self.rope_bak, dtype=human1.dtype).to(human1.device)
        max_indices = x_ref_attn_map.argmax(dim=0)
        normalized_map = torch.stack([human1, human2, back], dim=1)
        normalized_pos = normalized_map[range(x_ref_attn_map.size(1)), max_indices] # N 

        q = rearrange(q, "(B N_t) H S C -> B H (N_t S) C", N_t=N_t)
        q = self.rope_1d(q, normalized_pos)
        q = rearrange(q, "B H (N_t S) C -> (B N_t) H S C", N_t=N_t)

        _, N_a, _ = encoder_hidden_states.shape 
        encoder_kv = self.kv_linear(encoder_hidden_states) 
        encoder_kv_shape = (B, N_a, 2, self.num_heads, self.head_dim)
        encoder_kv = encoder_kv.view(encoder_kv_shape).permute((2, 0, 3, 1, 4)) 
        encoder_k, encoder_v = encoder_kv.unbind(0) 

        if self.qk_norm:
            encoder_k = self.add_k_norm(encoder_k)

        
        per_frame = torch.zeros(N_a, dtype=encoder_k.dtype).to(encoder_k.device)
        per_frame[:per_frame.size(0)//2] = (self.rope_h1[0] + self.rope_h1[1]) / 2
        per_frame[per_frame.size(0)//2:] = (self.rope_h2[0] + self.rope_h2[1]) / 2
        encoder_pos = torch.concat([per_frame]*N_t, dim=0)
        encoder_k = rearrange(encoder_k, "(B N_t) H S C -> B H (N_t S) C", N_t=N_t)
        encoder_k = self.rope_1d(encoder_k, encoder_pos)
        encoder_k = rearrange(encoder_k, "B H (N_t S) C -> (B N_t) H S C", N_t=N_t)

 
        q = rearrange(q, "B H M K -> B M H K")
        encoder_k = rearrange(encoder_k, "B H M K -> B M H K")
        encoder_v = rearrange(encoder_v, "B H M K -> B M H K")
        try:
            x = xformers.ops.memory_efficient_attention(q, encoder_k, encoder_v, attn_bias=None, op=None)
            _warn_once(
                "xformers_mea_ok",
                "xFormers memory_efficient_attention: OK (has been used at least once).",
            )
        except NotImplementedError:
            _warn_once(
                "xformers_mea_fallback",
                "xFormers memory_efficient_attention: NOT available -> falling back to PyTorch attention.",
            )
            x = attention(q, encoder_k, encoder_v, dropout_p=0.0, causal=False, dtype=q.dtype)
        x = rearrange(x, "B M H K -> B H M K")

        # linear transform
        x_output_shape = (B, N, C)
        x = x.transpose(1, 2) 
        x = x.reshape(x_output_shape) 
        x = self.proj(x) 
        x = self.proj_drop(x)

        # reshape x to origin shape
        x = rearrange(x, "(B N_t) S C -> B (N_t S) C", N_t=N_t) 

        return x