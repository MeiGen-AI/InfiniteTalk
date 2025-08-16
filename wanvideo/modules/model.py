# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import repeat, rearrange
from ...enhance_a_video.enhance import get_feta_scores

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention, BlockMask
    create_block_mask = torch.compile(create_block_mask)
    flex_attention = torch.compile(flex_attention)
except:
    BlockMask = create_block_mask = flex_attention = None
    pass
try:
    from ..radial_attention.attn_mask import RadialSpargeSageAttn, RadialSpargeSageAttnDense, MaskMap
except:
    pass

from .attention import attention
import numpy as np
__all__ = ['WanModel']

from tqdm import tqdm
import gc
import comfy.model_management as mm
from ...utils import log, get_module_memory_mb
from ...cache_methods.cache_methods import TeaCacheState, MagCacheState, EasyCacheState, relative_l1_distance
from ...multitalk.multitalk import get_attn_map_with_target

from comfy.ldm.flux.math import apply_rope as apply_rope_comfy

def apply_rope_comfy_chunked(xq, xk, freqs_cis, num_chunks=4):
    seq_dim = 1
    
    # Initialize output tensors
    xq_out = torch.empty_like(xq)
    xk_out = torch.empty_like(xk)
    
    # Calculate chunks
    seq_len = xq.shape[seq_dim]
    chunk_sizes = [seq_len // num_chunks + (1 if i < seq_len % num_chunks else 0) 
                  for i in range(num_chunks)]
    
    # First pass: process xq completely
    start_idx = 0
    for size in chunk_sizes:
        end_idx = start_idx + size
        
        slices = [slice(None)] * len(xq.shape)
        slices[seq_dim] = slice(start_idx, end_idx)
        
        freq_slices = [slice(None)] * len(freqs_cis.shape)
        if seq_dim < len(freqs_cis.shape):
            freq_slices[seq_dim] = slice(start_idx, end_idx)
        freqs_chunk = freqs_cis[tuple(freq_slices)]
        
        xq_chunk = xq[tuple(slices)]
        xq_chunk_ = xq_chunk.to(dtype=freqs_cis.dtype).reshape(*xq_chunk.shape[:-1], -1, 1, 2)
        xq_out[tuple(slices)] = (freqs_chunk[..., 0] * xq_chunk_[..., 0] + 
                                freqs_chunk[..., 1] * xq_chunk_[..., 1]).reshape(*xq_chunk.shape).type_as(xq)
        
        del xq_chunk, xq_chunk_, freqs_chunk
        start_idx = end_idx
    
    # Second pass: process xk completely
    start_idx = 0
    for size in chunk_sizes:
        end_idx = start_idx + size
        
        slices = [slice(None)] * len(xk.shape)
        slices[seq_dim] = slice(start_idx, end_idx)
        
        freq_slices = [slice(None)] * len(freqs_cis.shape)
        if seq_dim < len(freqs_cis.shape):
            freq_slices[seq_dim] = slice(start_idx, end_idx)
        freqs_chunk = freqs_cis[tuple(freq_slices)]
        
        xk_chunk = xk[tuple(slices)]
        xk_chunk_ = xk_chunk.to(dtype=freqs_cis.dtype).reshape(*xk_chunk.shape[:-1], -1, 1, 2)
        xk_out[tuple(slices)] = (freqs_chunk[..., 0] * xk_chunk_[..., 0] + 
                                freqs_chunk[..., 1] * xk_chunk_[..., 1]).reshape(*xk_chunk.shape).type_as(xk)
        
        del xk_chunk, xk_chunk_, freqs_chunk
        start_idx = end_idx
    
    return xq_out, xk_out

def rope_riflex(pos, dim, theta, L_test, k, temporal):
    assert dim % 2 == 0
    if mm.is_device_mps(pos.device) or mm.is_intel_xpu() or mm.is_directml_enabled():
        device = torch.device("cpu")
    else:
        device = pos.device

    scale = torch.linspace(0, (dim - 2) / dim, steps=dim//2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta**scale)

    # RIFLEX modification - adjust last frequency component if L_test and k are provided
    if temporal and k > 0 and L_test:
        omega[k-1] = 0.9 * 2 * torch.pi / L_test

    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)

class EmbedND_RifleX(nn.Module):
    def __init__(self, dim, theta, axes_dim, num_frames, k):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim
        self.num_frames = num_frames
        self.k = k

    def forward(self, ids):
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope_riflex(ids[..., i], self.axes_dim[i], self.theta, self.num_frames, self.k, temporal=True if i == 0 else False) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)

def poly1d(coefficients, x):
    result = torch.zeros_like(x)
    for i, coeff in enumerate(coefficients):
        result += coeff * (x ** (len(coefficients) - 1 - i))
    return result.abs()

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def rope_params(max_seq_len, dim, theta=10000, L_test=25, k=0):
    assert dim % 2 == 0
    exponents = torch.arange(0, dim, 2, dtype=torch.float64).div(dim)
    inv_theta_pow = 1.0 / torch.pow(theta, exponents)
    
    if k > 0:
        print(f"RifleX: Using {k}th freq")
        inv_theta_pow[k-1] = 0.9 * 2 * torch.pi / L_test
        
    freqs = torch.outer(torch.arange(max_seq_len), inv_theta_pow)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

from comfy.model_management import get_torch_device, get_autocast_device
@torch.autocast(device_type=get_autocast_device(get_torch_device()), enabled=False)
@torch.compiler.disable()
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(x.dtype)


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, num_chunks=1):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        use_chunked = num_chunks > 1
        if use_chunked:
            return self.forward_chunked(x, num_chunks)
        else:
            return self._norm(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps).to(x.dtype)

    def forward_chunked(self, x, num_chunks=4):
        output = torch.empty_like(x)
        
        chunk_sizes = [x.shape[1] // num_chunks + (1 if i < x.shape[1] % num_chunks else 0) 
                    for i in range(num_chunks)]
        
        start_idx = 0
        for size in chunk_sizes:
            end_idx = start_idx + size
            
            chunk = x[:, start_idx:end_idx, :]
            
            norm_factor = torch.rsqrt(chunk.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            output[:, start_idx:end_idx, :] = chunk * norm_factor.to(chunk.dtype) * self.weight

            start_idx = end_idx
            
        return output


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 num_heads,
                 qk_norm=True,
                 eps=1e-6,
                 attention_mode='sdpa'):
        assert out_features % num_heads == 0
        super().__init__()
        self.dim = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.qk_norm = qk_norm
        self.eps = eps
        self.attention_mode = attention_mode

        #radial attention
        self.mask_map = None
        self.decay_factor = 0.2

        # layers
        self.q = nn.Linear(in_features, out_features)
        self.k = nn.Linear(in_features, out_features)
        self.v = nn.Linear(in_features, out_features)
        self.o = nn.Linear(in_features, out_features)
        self.norm_q = WanRMSNorm(out_features, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(out_features, eps=eps) if qk_norm else nn.Identity()
    
    def qkv_fn(self, x):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    def forward(self, q, k, v, seq_lens, block_mask=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """

        if self.attention_mode == 'flex_attention':
            padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
            padded_roped_query = torch.cat(
                [q, torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                             device=q.device, dtype=v.dtype)], dim=1
                )

            padded_roped_key = torch.cat(
                [k, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                        device=k.device, dtype=v.dtype)], dim=1
                )

            padded_v = torch.cat(
                [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                device=v.device, dtype=v.dtype)],  dim=1
                )

            x = flex_attention(
                query=padded_roped_query.transpose(2, 1),
                key=padded_roped_key.transpose(2, 1),
                value=padded_v.transpose(2, 1),
                block_mask=block_mask
            )[:, :, :-padded_length].transpose(2, 1)
        else:                
            x = attention(
                q, k, v,
                k_lens=seq_lens,
                attention_mode=self.attention_mode
                )

        # output
        x = x.flatten(2)
        x = self.o(x)

        return x
    
    def forward_radial(self, q, k, v, dense_step=False):
        if dense_step:
            x = RadialSpargeSageAttnDense(q, k, v, self.mask_map)
        else:
            x = RadialSpargeSageAttn(q, k, v, self.mask_map, decay_factor=self.decay_factor)

        x = self.o(x.flatten(2))

        return x
    
    def forward_multitalk(self, q, k, v, seq_lens, grid_sizes, ref_target_masks):
        x = attention(
            q, k, v,
            k_lens=seq_lens,
            attention_mode=self.attention_mode
            )

        # output
        x = x.flatten(2)
        x = self.o(x)

        x_ref_attn_map = get_attn_map_with_target(q.type_as(x), k.type_as(x), grid_sizes[0], ref_target_masks=ref_target_masks)

        return x, x_ref_attn_map
    
    def forward_split(self, q, k, v, seq_lens, grid_sizes, freqs, seq_chunks=1,current_step=0, video_attention_split_steps = []):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """

        # Split by frames if multiple prompts are provided
        if seq_chunks > 1 and current_step in video_attention_split_steps:
            outputs = []
            # Extract frame, height, width from grid_sizes - force to CPU scalars
            frames = grid_sizes[0][0].item()
            height = grid_sizes[0][1].item()
            width = grid_sizes[0][2].item()
            tokens_per_frame = height * width
            
            actual_chunks = min(seq_chunks, frames)
            if isinstance(actual_chunks, torch.Tensor):
                actual_chunks = actual_chunks.item()
            
            frame_chunks = []  # Pre-calculate all chunk boundaries
            start_frame = 0
            base_frames_per_chunk = frames // actual_chunks
            extra_frames = frames % actual_chunks
            
            # Pre-calculate all chunks
            for i in range(actual_chunks):
                chunk_size = base_frames_per_chunk + (1 if i < extra_frames else 0)
                end_frame = start_frame + chunk_size
                frame_chunks.append((start_frame, end_frame))
                start_frame = end_frame
            
            # Process each chunk using the pre-calculated boundaries
            for start_frame, end_frame in frame_chunks:
                # Convert to token indices
                start_idx = int(start_frame * tokens_per_frame)
                end_idx = int(end_frame * tokens_per_frame)
                
                chunk_q = q[:, start_idx:end_idx, :, :]
                chunk_k = k[:, start_idx:end_idx, :, :]
                chunk_v = v[:, start_idx:end_idx, :, :]
                
                chunk_out = attention(
                    q=chunk_q,
                    k=chunk_k,
                    v=chunk_v,
                    k_lens=seq_lens,
                    attention_mode=self.attention_mode)
                
                outputs.append(chunk_out)
            
            # Concatenate outputs along the sequence dimension
            x = torch.cat(outputs, dim=1)
        else:
            # Original attention computation
            x = attention(
                q=q,
                k=k,
                v=v,
                k_lens=seq_lens,
                attention_mode=self.attention_mode)

        # output
        x = x.flatten(2)
        x = self.o(x)

        return x
    
    def normalized_attention_guidance(self, b, n, d, q, context, nag_context=None, nag_params={}):
        # NAG text attention
        context_positive = context
        context_negative = nag_context
        nag_scale = nag_params['nag_scale']
        nag_alpha = nag_params['nag_alpha']
        nag_tau = nag_params['nag_tau']

        k_positive = self.norm_k(self.k(context_positive)).view(b, -1, n, d)
        v_positive = self.v(context_positive).view(b, -1, n, d)
        k_negative = self.norm_k(self.k(context_negative)).view(b, -1, n, d)
        v_negative = self.v(context_negative).view(b, -1, n, d)

        x_positive = attention(q, k_positive, v_positive, k_lens=None, attention_mode=self.attention_mode)
        x_positive = x_positive.flatten(2)

        x_negative = attention(q, k_negative, v_negative, k_lens=None, attention_mode=self.attention_mode)
        x_negative = x_negative.flatten(2)

        nag_guidance = x_positive * nag_scale - x_negative * (nag_scale - 1)
        
        norm_positive = torch.norm(x_positive, p=1, dim=-1, keepdim=True)
        norm_guidance = torch.norm(nag_guidance, p=1, dim=-1, keepdim=True)
        
        scale = norm_guidance / norm_positive
        scale = torch.nan_to_num(scale, nan=10.0)
        
        mask = scale > nag_tau
        adjustment = (norm_positive * nag_tau) / (norm_guidance + 1e-7)
        nag_guidance = torch.where(mask, nag_guidance * adjustment, nag_guidance)
        del mask, adjustment
        
        return nag_guidance * nag_alpha + x_positive * (1 - nag_alpha)

#region T2V crossattn
class WanT2VCrossAttention(WanSelfAttention):

    def __init__(self, in_features, out_features, num_heads, qk_norm=True, eps=1e-6, attention_mode='sdpa'):
        super().__init__(in_features, out_features, num_heads, qk_norm, eps)
        self.attention_mode = attention_mode

    def forward(self, x, context, context_lens, clip_embed=None, audio_proj=None, audio_context_lens=None, audio_scale=1.0, 
                num_latent_frames=21, nag_params={}, nag_context=None, is_uncond=False, rope_func="comfy"):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        # compute query
        q = self.norm_q(self.q(x),num_chunks=2 if rope_func == "comfy_chunked" else 1).view(b, -1, n, d)

        if nag_context is not None and not is_uncond:
            x_text = self.normalized_attention_guidance(b, n, d, q, context, nag_context, nag_params)
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
            x_text = attention(q, k, v, k_lens=None, attention_mode=self.attention_mode)
            x_text = x_text.flatten(2)

        x = x_text

        # FantasyTalking audio attention
        if audio_proj is not None:
            if len(audio_proj.shape) == 4:
                audio_q = q.view(b * num_latent_frames, -1, n, d)
                ip_key = self.k_proj(audio_proj).view(b * num_latent_frames, -1, n, d)
                ip_value = self.v_proj(audio_proj).view(b * num_latent_frames, -1, n, d)
                audio_x = attention(
                    audio_q, ip_key, ip_value, k_lens=audio_context_lens, attention_mode=self.attention_mode
                )
                audio_x = audio_x.view(b, q.size(1), n, d).flatten(2)
            elif len(audio_proj.shape) == 3:
                ip_key = self.k_proj(audio_proj).view(b, -1, n, d)
                ip_value = self.v_proj(audio_proj).view(b, -1, n, d)
                audio_x = attention(q, ip_key, ip_value, k_lens=audio_context_lens, attention_mode=self.attention_mode).flatten(2)
            
            x = x + audio_x * audio_scale

        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self, in_features, out_features, num_heads, qk_norm=True, eps=1e-6, attention_mode='sdpa'):
        super().__init__(in_features, out_features, num_heads, qk_norm, eps)
        self.k_img = nn.Linear(in_features, out_features)
        self.v_img = nn.Linear(in_features, out_features)
        self.norm_k_img = WanRMSNorm(out_features, eps=eps) if qk_norm else nn.Identity()
        self.attention_mode = attention_mode

    def forward(self, x, context, context_lens, clip_embed, audio_proj=None, audio_context_lens=None, 
                audio_scale=1.0, num_latent_frames=21, nag_params={}, nag_context=None, is_uncond=False, rope_func="comfy"):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        # compute query
        q = self.norm_q(self.q(x),num_chunks=2 if rope_func == "comfy_chunked" else 1).view(b, -1, n, d)

        if nag_context is not None and not is_uncond:
            x_text = self.normalized_attention_guidance(b, n, d, q, context, nag_context, nag_params)
        else:
            # text attention
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
            x_text = attention(q, k, v, k_lens=context_lens, attention_mode=self.attention_mode).flatten(2)

        #img attention
        if clip_embed is not None:
            k_img = self.norm_k_img(self.k_img(clip_embed)).view(b, -1, n, d)
            v_img = self.v_img(clip_embed).view(b, -1, n, d)
            img_x = attention(q, k_img, v_img, k_lens=None, attention_mode=self.attention_mode).flatten(2)
            x = x_text + img_x
        else:
            x = x_text

        # FantasyTalking audio attention
        if audio_proj is not None:
            if len(audio_proj.shape) == 4:
                audio_q = q.view(b * num_latent_frames, -1, n, d)
                ip_key = self.k_proj(audio_proj).view(b * num_latent_frames, -1, n, d)
                ip_value = self.v_proj(audio_proj).view(b * num_latent_frames, -1, n, d)
                audio_x = attention(
                    audio_q, ip_key, ip_value, k_lens=audio_context_lens, attention_mode=self.attention_mode
                )
                audio_x = audio_x.view(b, q.size(1), n, d).flatten(2)
            elif len(audio_proj.shape) == 3:
                ip_key = self.k_proj(audio_proj).view(b, -1, n, d)
                ip_value = self.v_proj(audio_proj).view(b, -1, n, d)
                audio_x = attention(q, ip_key, ip_value, k_lens=audio_context_lens, attention_mode=self.attention_mode).flatten(2)
            
            x = x + audio_x * audio_scale

        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 in_features,
                 out_features,
                 ffn_dim,
                 ffn2_dim,
                 num_heads,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 attention_mode='sdpa',
                 rope_func="comfy",
                 ):
        super().__init__()
        self.dim = out_features
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.attention_mode = attention_mode
        self.rope_func = rope_func
        #radial attn
        self.dense_timesteps = 10
        self.dense_block = False
        self.dense_attention_mode = "sageattn"

        # layers
        self.norm1 = WanLayerNorm(out_features, eps)
        self.self_attn = WanSelfAttention(in_features, out_features, num_heads, qk_norm,
                                          eps, self.attention_mode)
        if cross_attn_type != "no_cross_attn":
            self.norm3 = WanLayerNorm(
                out_features, eps,
                elementwise_affine=True) if cross_attn_norm else nn.Identity()
            self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](in_features,
                                                                          out_features,
                                                                          num_heads,
                                                                          qk_norm,
                                                                          eps,#attention_mode=attention_mode sageattn doesn't seem faster here
                                                                          )
        self.norm2 = WanLayerNorm(out_features, eps)
        self.ffn = nn.Sequential(
            nn.Linear(in_features, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn2_dim, out_features))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, out_features) / in_features**0.5)

    @torch.compiler.disable()
    def get_mod(self, e):
        if e.dim() == 3:
            modulation = self.modulation  # 1, 6, dim
            e = (modulation.to(e.device) + e).chunk(6, dim=1)
        elif e.dim() == 4:
            modulation = self.modulation.unsqueeze(2)  # 1, 6, 1, dim
            e = (modulation.to(e.device) + e).chunk(6, dim=1)
            e = [ei.squeeze(1) for ei in e]
        return e
    
    def modulate(self, x, shift_msa, scale_msa):
        return torch.addcmul(shift_msa, x, 1 + scale_msa)
    
    def ffn_chunked(self, x, shift_mlp, scale_mlp, num_chunks=4):
        modulated_input = torch.addcmul(shift_mlp, self.norm2(x), 1 + scale_mlp)
        
        result = torch.empty_like(x)
        seq_len = modulated_input.shape[1]
        
        chunk_sizes = [seq_len // num_chunks + (1 if i < seq_len % num_chunks else 0) 
                    for i in range(num_chunks)]
        
        start_idx = 0
        for size in chunk_sizes:
            end_idx = start_idx + size
            chunk = modulated_input[:, start_idx:end_idx, :]
            result[:, start_idx:end_idx, :] = self.ffn(chunk)
            start_idx = end_idx
        
        return result

    #region attention forward
    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        current_step,
        video_attention_split_steps=[],
        clip_embed=None,
        camera_embed=None,
        audio_proj=None,
        audio_context_lens=None,
        audio_scale=1.0,
        num_latent_frames=21,
        enhance_enabled=False,
        block_mask=None,
        nag_params={},
        nag_context=None,
        is_uncond=False,
        multitalk_audio_embedding=None,
        ref_target_masks=None,
        human_num=0
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        #e = (self.modulation.to(e.device) + e).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.get_mod(e)
        input_x = self.modulate(self.norm1(x), shift_msa, scale_msa)

        if camera_embed is not None:
            # encode ReCamMaster camera
            camera_embed = self.cam_encoder(camera_embed.to(x))
            camera_embed = camera_embed.repeat(1, 2, 1)
            camera_embed = camera_embed.unsqueeze(2).unsqueeze(3).repeat(1, 1, grid_sizes[0][1], grid_sizes[0][2], 1)
            camera_embed = rearrange(camera_embed, 'b f h w d -> b (f h w) d')
            input_x += camera_embed

        # self-attention
        x_ref_attn_map = None

        #query, key, value
        q, k, v = self.self_attn.qkv_fn(input_x)

        # FETA
        if enhance_enabled:
            feta_scores = get_feta_scores(q, k)

        #RoPE
        if self.rope_func == "comfy":
            q, k = apply_rope_comfy(q, k, freqs)
        elif self.rope_func == "comfy_chunked":
            q, k = apply_rope_comfy_chunked(q, k, freqs)
        else:
            q=rope_apply(q, grid_sizes, freqs)
            k=rope_apply(k, grid_sizes, freqs)

        #self-attention
        split_attn = context is not None and (context.shape[0] > 1 or (clip_embed is not None and clip_embed.shape[0] > 1)) and x.shape[0] == 1
        if split_attn:
            y = self.self_attn.forward_split(
            q, k, v, 
            seq_lens, grid_sizes, freqs, 
            seq_chunks=max(context.shape[0], clip_embed.shape[0] if clip_embed is not None else 0),
            current_step=current_step,
            video_attention_split_steps=video_attention_split_steps
            )
        elif ref_target_masks is not None:
            y, x_ref_attn_map = self.self_attn.forward_multitalk(q, k, v, seq_lens, grid_sizes, ref_target_masks)
        elif self.attention_mode == "radial_sage_attention":
            if self.dense_block or self.dense_timesteps is not None and current_step < self.dense_timesteps:
                if self.dense_attention_mode == "sparse_sage_attn":
                    y = self.self_attn.forward_radial(q, k, v, dense_step=True)
                else:
                    y = self.self_attn.forward(q, k, v, seq_lens, block_mask=block_mask)
            else:
                y = self.self_attn.forward_radial(q, k, v, dense_step=False)
        else:
            y = self.self_attn.forward(q, k, v, seq_lens, block_mask=block_mask)

        # FETA
        if enhance_enabled:
            y.mul_(feta_scores)

        #ReCamMaster
        if camera_embed is not None:
            y = self.projector(y)        

        x = x.addcmul(y, gate_msa)

        # cross-attention & ffn function
        
        if context is not None:
            if split_attn:
                if nag_context is not None:
                    raise NotImplementedError("nag_context is not supported in split_cross_attn_ffn")
                x = self.split_cross_attn_ffn(x, context, context_lens, shift_mlp, scale_mlp, gate_mlp, clip_embed=clip_embed, grid_sizes=grid_sizes)
            else:
                x = self.cross_attn_ffn(x, context, context_lens, shift_mlp, scale_mlp, gate_mlp, clip_embed=clip_embed, grid_sizes=grid_sizes, 
                                        audio_proj=audio_proj, audio_context_lens=audio_context_lens, audio_scale=audio_scale, 
                                        num_latent_frames=num_latent_frames, nag_params=nag_params, nag_context=nag_context, is_uncond=is_uncond, 
                                        multitalk_audio_embedding=multitalk_audio_embedding, x_ref_attn_map=x_ref_attn_map, human_num=human_num)
        else:
            if self.rope_func == "comfy_chunked":
                y = self.ffn_chunked(x, shift_mlp, scale_mlp)
            else:
                y = self.ffn(torch.addcmul(shift_mlp, self.norm2(x), 1 + scale_mlp))
            x = x.addcmul(y, gate_mlp)

        return x

    
    def cross_attn_ffn(self, x, context, context_lens, shift_mlp, scale_mlp, gate_mlp, clip_embed=None, grid_sizes=None, 
                       audio_proj=None, audio_context_lens=None, audio_scale=1.0, num_latent_frames=21, nag_params={}, 
                       nag_context=None, is_uncond=False, multitalk_audio_embedding=None, x_ref_attn_map=None, human_num=0):
            x = x + self.cross_attn(self.norm3(x), context, context_lens, clip_embed=clip_embed,
                                    audio_proj=audio_proj, audio_context_lens=audio_context_lens, audio_scale=audio_scale, 
                                    num_latent_frames=num_latent_frames, nag_params=nag_params, nag_context=nag_context, is_uncond=is_uncond, rope_func=self.rope_func)
            #multitalk
            if multitalk_audio_embedding is not None and not isinstance(self, VaceWanAttentionBlock):
                x_audio = self.audio_cross_attn(self.norm_x(x), encoder_hidden_states=multitalk_audio_embedding,
                                            shape=grid_sizes[0], x_ref_attn_map=x_ref_attn_map, human_num=human_num)
                x = x + x_audio * audio_scale

            if self.rope_func == "comfy_chunked":
                y = self.ffn_chunked(x, shift_mlp, scale_mlp)
            else:
                y = self.ffn(torch.addcmul(shift_mlp, self.norm2(x), 1 + scale_mlp))
            x = x.addcmul(y, gate_mlp)
            return x
    
    @torch.compiler.disable()
    def split_cross_attn_ffn(self, x, context, context_lens, shift_mlp, scale_mlp, gate_mlp, clip_embed=None, grid_sizes=None):
        # Get number of prompts
        num_prompts = context.shape[0]
        num_clip_embeds = 0 if clip_embed is None else clip_embed.shape[0]
        num_segments = max(num_prompts, num_clip_embeds)
        
        # Extract spatial dimensions
        frames, height, width = grid_sizes[0]  # Assuming batch size 1
        tokens_per_frame = height * width
        
        # Distribute frames across prompts
        frames_per_segment = max(1, frames // num_segments)
        
        # Process each prompt segment
        x_combined = torch.zeros_like(x)
        
        for i in range(num_segments):
            # Calculate frame boundaries for this segment
            start_frame = i * frames_per_segment
            end_frame = min((i+1) * frames_per_segment, frames) if i < num_segments-1 else frames
            
            # Convert frame indices to token indices
            start_idx = start_frame * tokens_per_frame
            end_idx = end_frame * tokens_per_frame
            segment_indices = torch.arange(start_idx, end_idx, device=x.device, dtype=torch.long)
            
            # Get prompt segment (cycle through available prompts if needed)
            prompt_idx = i % num_prompts
            segment_context = context[prompt_idx:prompt_idx+1]
            segment_context_lens = None
            if context_lens is not None:
                segment_context_lens = context_lens[prompt_idx:prompt_idx+1]
            
            # Handle clip_embed for this segment (cycle through available embeddings)
            segment_clip_embed = None
            if clip_embed is not None:
                clip_idx = i % num_clip_embeds
                segment_clip_embed = clip_embed[clip_idx:clip_idx+1]
            
            # Get tensor segment
            x_segment = x[:, segment_indices, :]
            
            # Process segment with its prompt and clip embedding
            processed_segment = self.cross_attn(self.norm3(x_segment), segment_context, segment_context_lens, clip_embed=segment_clip_embed)
            processed_segment = processed_segment.to(x.dtype)
            
            # Add to combined result
            x_combined[:, segment_indices, :] = processed_segment
        
        # Continue with FFN
        x = x + x_combined
        y = self.ffn_chunked(x, shift_mlp, scale_mlp)
        x = x.addcmul(y, gate_mlp)
        return x

class VaceWanAttentionBlock(WanAttentionBlock):
    def __init__(
            self,
            cross_attn_type,
            in_features,
            out_features,
            ffn_dim,
            ffn2_dim,
            num_heads,
            qk_norm=True,
            cross_attn_norm=False,
            eps=1e-6,
            block_id=0,
            attention_mode='sdpa',
            rope_func="comfy"
    ):
        super().__init__(cross_attn_type, in_features, out_features, ffn_dim, ffn2_dim, num_heads, qk_norm, cross_attn_norm, eps, attention_mode, rope_func)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(in_features, out_features)
        self.after_proj = nn.Linear(in_features, out_features)

    def forward(self, c, **kwargs):
        return super().forward(c, **kwargs)

class BaseWanAttentionBlock(WanAttentionBlock):
    def __init__(
        self,
        cross_attn_type,
        in_features,
        out_features,
        ffn_dim,
        ffn2_dim,
        num_heads,
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        block_id=None,
        attention_mode='sdpa',
        rope_func="comfy"
    ):
        super().__init__(cross_attn_type, in_features, out_features, ffn_dim, ffn2_dim, num_heads, qk_norm, cross_attn_norm, eps, attention_mode, rope_func)
        self.block_id = block_id

    def forward(self, x, vace_hints=None, vace_context_scale=[1.0], **kwargs):
        x = super().forward(x, **kwargs)
        if vace_hints is None:
            return x
        
        if self.block_id is not None:
            for i in range(len(vace_hints)):
                x.add_(vace_hints[i][self.block_id].to(x.device), alpha=vace_context_scale[i])
        return x

class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def get_mod(self, e):
        if e.dim() == 2:
            modulation = self.modulation.to(e.device)  # 1, 2, dim
            e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
        elif e.dim() == 3:
            modulation = self.modulation.to(e.device).unsqueeze(2)  # 1, 2, seq, dim
            e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
            e = [ei.squeeze(1) for ei in e]
        return e

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """

        e = self.get_mod(e)
        x = self.head(self.norm(x).mul_(1 + e[1]).add_(e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, fl_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))
        if fl_pos_emb:  # NOTE: we only use this for `fl2v`
            self.emb_pos = nn.Parameter(torch.zeros(1, 257 * 2, 1280))

    def forward(self, image_embeds):
        if hasattr(self, 'emb_pos'):
            image_embeds = image_embeds + self.emb_pos.to(image_embeds.device)
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 in_features=5120,
                 out_features=5120,
                 ffn_dim=8192,
                 ffn2_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 attention_mode='sdpa',
                 rope_func='comfy',
                 main_device=torch.device('cuda'),
                 offload_device=torch.device('cpu'),
                 teacache_coefficients=[],
                 magcache_ratios=[],
                 vace_layers=None,
                 vace_in_dim=None,
                 inject_sample_info=False,
                 add_ref_conv=False,
                 in_dim_ref_conv=16,
                 add_control_adapter=False,
                 in_dim_control_adapter=24,
                 ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.in_features = in_features
        self.out_features = out_features
        self.ffn_dim = ffn_dim
        self.ffn2_dim = ffn2_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.attention_mode = attention_mode
        self.rope_func = rope_func
        self.main_device = main_device
        self.offload_device = offload_device

        self.blocks_to_swap = -1
        self.offload_txt_emb = False
        self.offload_img_emb = False
        self.vace_blocks_to_swap = -1

        self.cache_device = offload_device

        #init TeaCache variables
        self.enable_teacache = False
        self.rel_l1_thresh = 0.15
        self.teacache_start_step= 0
        self.teacache_end_step = -1
        self.teacache_state = TeaCacheState(cache_device=self.cache_device)
        self.teacache_coefficients = teacache_coefficients
        self.teacache_use_coefficients = False
        self.teacache_mode = 'e'

        #init MagCache variables
        self.enable_magcache = False
        self.magcache_state = MagCacheState(cache_device=self.cache_device)
        self.magcache_thresh = 0.24
        self.magcache_K = 4
        self.magcache_start_step = 0
        self.magcache_end_step = -1
        self.magcache_ratios = magcache_ratios

        #init EasyCache variables
        self.enable_easycache = False
        self.easycache_thresh = 0.1
        self.easycache_start_step = 0
        self.easycache_end_step = -1
        self.easycache_state = EasyCacheState(cache_device=self.cache_device)

        self.slg_blocks = None
        self.slg_start_percent = 0.0
        self.slg_end_percent = 1.0

        self.use_non_blocking = True

        self.video_attention_split_steps = []

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        
        self.original_patch_embedding = self.patch_embedding
        self.expanded_patch_embedding = self.patch_embedding

        if model_type != 'no_cross_attn':
            self.text_embedding = nn.Sequential(
                nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
                nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        if vace_layers is not None:
            self.vace_layers = [i for i in range(0, self.num_layers, 2)] if vace_layers is None else vace_layers
            self.vace_in_dim = self.in_dim if vace_in_dim is None else vace_in_dim

            self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

            # vace blocks
            self.vace_blocks = nn.ModuleList([
                VaceWanAttentionBlock('t2v_cross_attn', self.in_features, self.out_features, self.ffn_dim, self.ffn2_dim,self.num_heads, self.qk_norm,
                                        self.cross_attn_norm, self.eps, block_id=i, attention_mode=self.attention_mode, rope_func=self.rope_func)
                for i in self.vace_layers
            ])

            # vace patch embeddings
            self.vace_patch_embedding = nn.Conv3d(
                self.vace_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size
            )
            self.blocks = nn.ModuleList([
            BaseWanAttentionBlock('t2v_cross_attn', self.in_features, self.out_features, ffn_dim, self.ffn2_dim, num_heads,
                              qk_norm, cross_attn_norm, eps,
                              attention_mode=self.attention_mode, rope_func=self.rope_func,
                              block_id=self.vace_layers_mapping[i] if i in self.vace_layers else None)
            for i in range(num_layers)
            ])
        else:
            # blocks
            if model_type == 't2v':
                cross_attn_type = 't2v_cross_attn'
            elif model_type == 'i2v' or model_type == 'fl2v':
                cross_attn_type = 'i2v_cross_attn'
            else:
                cross_attn_type = 'no_cross_attn'

            self.blocks = nn.ModuleList([
                WanAttentionBlock(cross_attn_type, self.in_features, self.out_features, ffn_dim, ffn2_dim, num_heads,
                                qk_norm, cross_attn_norm, eps,
                                attention_mode=self.attention_mode, rope_func=self.rope_func)
                for _ in range(num_layers)
            ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)
        

        d = self.dim // self.num_heads
        self.rope_embedder = EmbedND_RifleX(
            d, 
            10000.0, 
            [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)],
            num_frames=None,
            k=None,
            )
        self.cached_freqs = self.cached_shape = self.cached_cond = None

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        
        if model_type == 'i2v' or model_type == 'fl2v':
            self.img_emb = MLPProj(1280, dim, fl_pos_emb=model_type == 'fl2v')

        #skyreels v2
        if inject_sample_info:
            self.fps_embedding = nn.Embedding(2, dim)
            self.fps_projection = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim * 6))
        #fun 1.1
        if add_ref_conv:
            self.ref_conv = nn.Conv2d(in_dim_ref_conv, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.ref_conv = None

        if add_control_adapter:
            from .wan_camera_adapter import SimpleAdapter
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

        self.block_mask=None

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ):
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        print("num_frames", num_frames)
        print("frame_seqlen", frame_seqlen)
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        
        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        return block_mask

    def block_swap(self, blocks_to_swap, offload_txt_emb=False, offload_img_emb=False, vace_blocks_to_swap=None):
        log.info(f"Swapping {blocks_to_swap + 1} transformer blocks")
        self.blocks_to_swap = blocks_to_swap
        
        self.offload_img_emb = offload_img_emb
        self.offload_txt_emb = offload_txt_emb

        total_offload_memory = 0
        total_main_memory = 0
       
        for b, block in tqdm(enumerate(self.blocks), total=len(self.blocks), desc="Initializing block swap"):
            block_memory = get_module_memory_mb(block)
            
            if b > self.blocks_to_swap:
                block.to(self.main_device)
                total_main_memory += block_memory
            else:
                block.to(self.offload_device, non_blocking=self.use_non_blocking)
                total_offload_memory += block_memory

        if blocks_to_swap != -1 and vace_blocks_to_swap == 0:
            vace_blocks_to_swap = 1

        if vace_blocks_to_swap > 0 and self.vace_layers is not None:
            self.vace_blocks_to_swap = vace_blocks_to_swap

            for b, block in tqdm(enumerate(self.vace_blocks), total=len(self.vace_blocks), desc="Initializing vace block swap"):
                block_memory = get_module_memory_mb(block)
                
                if b > self.vace_blocks_to_swap:
                    block.to(self.main_device)
                    total_main_memory += block_memory
                else:
                    block.to(self.offload_device, non_blocking=self.use_non_blocking)
                    total_offload_memory += block_memory

        mm.soft_empty_cache()
        gc.collect()

        log.info("----------------------")
        log.info(f"Block swap memory summary:")
        log.info(f"Transformer blocks on {self.offload_device}: {total_offload_memory:.2f}MB")
        log.info(f"Transformer blocks on {self.main_device}: {total_main_memory:.2f}MB")
        log.info(f"Total memory used by transformer blocks: {(total_offload_memory + total_main_memory):.2f}MB")
        log.info(f"Non-blocking memory transfer: {self.use_non_blocking}")
        log.info("----------------------")

    def forward_vace(
        self,
        x,
        vace_context,
        seq_len,
        kwargs
    ):
        # embeddings
        c = [self.vace_patch_embedding(u.unsqueeze(0).float()).to(x.dtype) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in c
        ])

        if x.shape[1] > c.shape[1]:
            c = torch.cat([c.new_zeros(x.shape[0], x.shape[1] - c.shape[1], c.shape[2]), c], dim=1)
        if c.shape[1] > x.shape[1]:
            c = c[:, :x.shape[1]]
        
        hints = []
        current_c = c
        
        for b, block in enumerate(self.vace_blocks):
            if b <= self.vace_blocks_to_swap and self.vace_blocks_to_swap >= 0:
                block.to(self.main_device)
                
            if b == 0:
                c_processed = block.before_proj(current_c) + x
            else:
                c_processed = current_c
                
            c_processed = block.forward(c_processed, **kwargs)
            
            # Store skip connection
            c_skip = block.after_proj(c_processed)
            hints.append(c_skip.to(
                self.offload_device if self.vace_blocks_to_swap != -1 else self.main_device, 
                non_blocking=self.use_non_blocking
            ))
            
            current_c = c_processed
            
            if b <= self.vace_blocks_to_swap and self.vace_blocks_to_swap >= 0:
                block.to(self.offload_device, non_blocking=self.use_non_blocking)

        return hints

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        is_uncond=False,
        current_step_percentage=0.0,
        current_step=0,
        total_steps=50,
        clip_fea=None,
        y=None,
        device=torch.device('cuda'),
        freqs=None,
        enhance_enabled=False,
        pred_id=None,
        control_lora_enabled=False,
        vace_data=None,
        camera_embed=None,
        unianim_data=None,
        fps_embeds=None,
        fun_ref=None,
        fun_camera=None,
        audio_proj=None,
        audio_context_lens=None,
        audio_scale=1.0,
        pcd_data=None,
        controlnet=None,
        add_cond=None,
        attn_cond=None,
        nag_params={},
        nag_context=None,
        multitalk_audio=None,
        ref_target_masks=None
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """        
        # params
        device = self.patch_embedding.weight.device
        if freqs is not None and freqs.device != device:
           freqs = freqs.to(device)

        _, F, H, W = x[0].shape
 
        # Construct blockwise causal attn mask
        if self.attention_mode == 'flex_attention' and current_step == 0:
            self.block_mask = self._prepare_blockwise_causal_attn_mask(
                device, num_frames=F,
                frame_seqlen=H * W // (self.patch_size[1] * self.patch_size[2]),
                num_frame_per_block=3
            )
            
        if y is not None:
            if hasattr(self, "randomref_embedding_pose") and unianim_data is not None:
                if unianim_data['start_percent'] <= current_step_percentage <= unianim_data['end_percent']:
                    random_ref_emb = unianim_data["random_ref"]
                    if random_ref_emb is not None:
                        y[0] = y[0] + random_ref_emb * unianim_data["strength"]
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        
        #uni3c controlnet
        if pcd_data is not None:
            hidden_states = x[0].unsqueeze(0).clone().float()
            render_latent = torch.cat([hidden_states[:, :20], pcd_data["render_latent"]], dim=1)

        # embeddings
        if control_lora_enabled:
            self.expanded_patch_embedding.to(device)
            x = [
            self.expanded_patch_embedding(u.unsqueeze(0).to(torch.float32)).to(x[0].dtype)
            for u in x
            ]
        else:
            self.original_patch_embedding.to(self.main_device)
            x = [
            self.original_patch_embedding(u.unsqueeze(0).to(torch.float32)).to(x[0].dtype)
            for u in x
            ]

        if self.control_adapter is not None and fun_camera is not None:
            fun_camera = self.control_adapter(fun_camera)
            x = [u + v for u, v in zip(x, fun_camera)]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], device=device, dtype=torch.long) for u in x])

        x = [u.flatten(2).transpose(1, 2) for u in x]

        x_len = x[0].shape[1]

        if add_cond is not None:
            add_cond = self.add_conv_in(add_cond.to(self.add_conv_in.weight.dtype)).to(x[0].dtype)
            add_cond = add_cond.flatten(2).transpose(1, 2)
            x[0] = x[0] + self.add_proj(add_cond)
        if attn_cond is not None:
            F_cond, H_cond, W_cond = attn_cond.shape[2], attn_cond.shape[3], attn_cond.shape[4]
            grid_sizes = torch.stack([torch.tensor([u[0] + 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            attn_cond = self.attn_conv_in(attn_cond.to(self.attn_conv_in.weight.dtype)).to(x[0].dtype)
            attn_cond = attn_cond.flatten(2).transpose(1, 2)
            x[0] = torch.cat([x[0], attn_cond], dim=1)
            seq_len += attn_cond.size(1)
            for block in self.blocks:
                block.self_attn.mask_map = MaskMap(video_token_num=seq_len, num_frame=F+1)

        if self.ref_conv is not None and fun_ref is not None:
            fun_ref = self.ref_conv(fun_ref).flatten(2).transpose(1, 2)
            grid_sizes = torch.stack([torch.tensor([u[0] + 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            seq_len += fun_ref.size(1)
            F += 1
            x = [torch.concat([_fun_ref.unsqueeze(0), u], dim=1) for _fun_ref, u in zip(fun_ref, x)]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        if freqs is None: #comfy rope
            current_shape = (F, H, W)
            has_cond = attn_cond is not None
            if (self.cached_freqs is not None and 
                self.cached_shape == current_shape and 
                self.cached_cond == has_cond):
                freqs = self.cached_freqs
            else:
                f_len = ((F + (self.patch_size[0] // 2)) // self.patch_size[0])
                h_len = ((H + (self.patch_size[1] // 2)) // self.patch_size[1])
                w_len = ((W + (self.patch_size[2] // 2)) // self.patch_size[2])
                img_ids = torch.zeros((f_len, h_len, w_len, 3), device=x.device, dtype=x.dtype)
                img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(0, f_len - 1, steps=f_len, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
                img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).reshape(1, -1, 1)
                img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).reshape(1, 1, -1)

                if attn_cond is not None:   
                    cond_f_len = ((F_cond + (self.patch_size[0] // 2)) // self.patch_size[0])
                    cond_h_len = ((H_cond + (self.patch_size[1] // 2)) // self.patch_size[1])
                    cond_w_len = ((W_cond + (self.patch_size[2] // 2)) // self.patch_size[2])
                    cond_img_ids = torch.zeros((cond_f_len, cond_h_len, cond_w_len, 3), device=x.device, dtype=x.dtype)
                    
                    #shift
                    shift_f_size = 81 # Default value
                    shift_f = False
                    if shift_f:
                        cond_img_ids[:, :, :, 0] = cond_img_ids[:, :, :, 0] + torch.linspace(shift_f_size, shift_f_size + cond_f_len - 1,steps=cond_f_len, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
                    else:
                        cond_img_ids[:, :, :, 0] = cond_img_ids[:, :, :, 0] + torch.linspace(0, cond_f_len - 1, steps=cond_f_len, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
                    cond_img_ids[:, :, :, 1] = cond_img_ids[:, :, :, 1] + torch.linspace(h_len, h_len + cond_h_len - 1, steps=cond_h_len, device=x.device, dtype=x.dtype).reshape(1, -1, 1)
                    cond_img_ids[:, :, :, 2] = cond_img_ids[:, :, :, 2] + torch.linspace(w_len, w_len + cond_w_len - 1, steps=cond_w_len, device=x.device, dtype=x.dtype).reshape(1, 1, -1)
                
                    # Combine original and conditional position ids
                    img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=1)
                    cond_img_ids = repeat(cond_img_ids, "t h w c -> b (t h w) c", b=1)
                    combined_img_ids = torch.cat([img_ids, cond_img_ids], dim=1)
                    
                    # Generate RoPE frequencies for the combined positions
                    freqs = self.rope_embedder(combined_img_ids).movedim(1, 2)
                else:
                    img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=1)
                    freqs = self.rope_embedder(img_ids).movedim(1, 2)
                self.cached_freqs = freqs
                self.cached_shape = current_shape
                self.cached_cond = has_cond

        # time embeddings
        if t.dim() == 2:
            b, f = t.shape
            expanded_timesteps = True
        else:
            expanded_timesteps = False

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(x.dtype)
        )  # b, dim
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))  # b, 6, dim

        if fps_embeds is not None:
            fps_embeds = torch.tensor(fps_embeds, dtype=torch.long, device=device)

            fps_emb = self.fps_embedding(fps_embeds).to(e0.dtype)
            if expanded_timesteps:
                e0 = e0 + self.fps_projection(fps_emb).unflatten(1, (6, self.dim)).repeat(t.shape[1], 1, 1)
            else:
                e0 = e0 + self.fps_projection(fps_emb).unflatten(1, (6, self.dim))

        if expanded_timesteps:
            e = e.view(b, f, 1, 1, self.dim).expand(b, f, grid_sizes[0][1], grid_sizes[0][2], self.dim)
            e0 = e0.view(b, f, 1, 1, 6, self.dim).expand(b, f, grid_sizes[0][1], grid_sizes[0][2], 6, self.dim)
            
            e = e.flatten(1, 3)
            e0 = e0.flatten(1, 3)
            
            e0 = e0.transpose(1, 2)
            if not e0.is_contiguous():
                e0 = e0.contiguous()
            
            e = e.to(self.offload_device, non_blocking=self.use_non_blocking)

        #context (text embedding)
        context_lens = None
        if hasattr(self, "text_embedding") and context != []:
            if self.offload_txt_emb:
                self.text_embedding.to(self.main_device)
            context = self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]).to(x.dtype))
            # NAG
            if nag_context is not None:
                nag_context = self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in nag_context
                ]).to(x.dtype))
            
            if self.offload_txt_emb:
                self.text_embedding.to(self.offload_device, non_blocking=self.use_non_blocking)
        else:
            context = None

        clip_embed = None
        if clip_fea is not None and hasattr(self, "img_emb"):
            clip_fea = clip_fea.to(self.main_device)
            if self.offload_img_emb:
                self.img_emb.to(self.main_device)
            clip_embed = self.img_emb(clip_fea)  # bs x 257 x dim
            #context = torch.concat([context_clip, context], dim=1)
            if self.offload_img_emb:
                self.img_emb.to(self.offload_device, non_blocking=self.use_non_blocking)

        # MultiTalk
        if multitalk_audio is not None:
            self.audio_proj.to(self.main_device)
            audio_cond = multitalk_audio.to(device=x.device, dtype=x.dtype)
            first_frame_audio_emb_s = audio_cond[:, :1, ...] 
            latter_frame_audio_emb = audio_cond[:, 1:, ...] 
            latter_frame_audio_emb = rearrange(latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=4) 
            middle_index = self.audio_proj.seq_len // 2
            latter_first_frame_audio_emb = latter_frame_audio_emb[:, :, :1, :middle_index+1, ...] 
            latter_first_frame_audio_emb = rearrange(latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
            latter_last_frame_audio_emb = latter_frame_audio_emb[:, :, -1:, middle_index:, ...] 
            latter_last_frame_audio_emb = rearrange(latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
            latter_middle_frame_audio_emb = latter_frame_audio_emb[:, :, 1:-1, middle_index:middle_index+1, ...] 
            latter_middle_frame_audio_emb = rearrange(latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c") 
            latter_frame_audio_emb_s = torch.concat([latter_first_frame_audio_emb, latter_middle_frame_audio_emb, latter_last_frame_audio_emb], dim=2) 
            multitalk_audio_embedding = self.audio_proj(first_frame_audio_emb_s, latter_frame_audio_emb_s) 
            human_num = len(multitalk_audio_embedding)
            multitalk_audio_embedding = torch.concat(multitalk_audio_embedding.split(1), dim=2).to(x.dtype)
            self.audio_proj.to(self.offload_device)

        # convert ref_target_masks to token_ref_target_masks
        token_ref_target_masks = None
        if ref_target_masks is not None:
            ref_target_masks = ref_target_masks.unsqueeze(0).to(torch.float32) 
            token_ref_target_masks = nn.functional.interpolate(ref_target_masks, size=(H // 2, W // 2), mode='nearest') 
            token_ref_target_masks = token_ref_target_masks.squeeze(0)
            token_ref_target_masks = (token_ref_target_masks > 0)
            token_ref_target_masks = token_ref_target_masks.view(token_ref_target_masks.shape[0], -1) 
            token_ref_target_masks = token_ref_target_masks.to(x.dtype).to(device)

        should_calc = True
        #TeaCache
        if self.enable_teacache and self.teacache_start_step <= current_step <= self.teacache_end_step:
            accumulated_rel_l1_distance = torch.tensor(0.0, dtype=torch.float32, device=device)
            if pred_id is None:
                pred_id = self.teacache_state.new_prediction(cache_device=self.cache_device)
                should_calc = True                
            else:
                previous_modulated_input = self.teacache_state.get(pred_id)['previous_modulated_input']
                previous_modulated_input = previous_modulated_input.to(device)
                previous_residual = self.teacache_state.get(pred_id)['previous_residual']
                accumulated_rel_l1_distance = self.teacache_state.get(pred_id)['accumulated_rel_l1_distance']

                if self.teacache_use_coefficients:
                    rescale_func = np.poly1d(self.teacache_coefficients[self.teacache_mode])
                    temb = e if self.teacache_mode == 'e' else e0
                    accumulated_rel_l1_distance += rescale_func((
                        (temb.to(device) - previous_modulated_input).abs().mean() / previous_modulated_input.abs().mean()
                        ).cpu().item())
                    del temb
                else:
                    temb_relative_l1 = relative_l1_distance(previous_modulated_input, e0)
                    accumulated_rel_l1_distance = accumulated_rel_l1_distance.to(e0.device) + temb_relative_l1
                    del temb_relative_l1


                if accumulated_rel_l1_distance < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    accumulated_rel_l1_distance = torch.tensor(0.0, dtype=torch.float32, device=device)
                accumulated_rel_l1_distance = accumulated_rel_l1_distance.to(self.cache_device)

            previous_modulated_input = e.to(self.cache_device).clone() if (self.teacache_use_coefficients and self.teacache_mode == 'e') else e0.to(self.cache_device).clone()
           
            if not should_calc:
                x = x.to(previous_residual.dtype) + previous_residual.to(x.device)
                self.teacache_state.update(
                    pred_id,
                    accumulated_rel_l1_distance=accumulated_rel_l1_distance,
                )
                self.teacache_state.get(pred_id)['skipped_steps'].append(current_step)

        # MagCache
        if self.enable_magcache and self.magcache_start_step <= current_step <= self.magcache_end_step:
            if pred_id is None:
                pred_id = self.magcache_state.new_prediction(cache_device=self.cache_device)
                should_calc = True
            else:
                accumulated_ratio = self.magcache_state.get(pred_id)['accumulated_ratio']
                accumulated_err = self.magcache_state.get(pred_id)['accumulated_err']
                accumulated_steps = self.magcache_state.get(pred_id)['accumulated_steps']

                calibration_len = len(self.magcache_ratios) // 2
                cur_mag_ratio = self.magcache_ratios[int((current_step*(calibration_len/total_steps)))]

                accumulated_ratio *= cur_mag_ratio
                accumulated_err += np.abs(1-accumulated_ratio)
                accumulated_steps += 1

                self.magcache_state.update(
                    pred_id,
                    accumulated_ratio=accumulated_ratio,
                    accumulated_steps=accumulated_steps,
                    accumulated_err=accumulated_err
                )

                if accumulated_err<=self.magcache_thresh and accumulated_steps<=self.magcache_K:
                    should_calc = False
                    x += self.magcache_state.get(pred_id)['residual_cache'].to(x.device)
                    self.magcache_state.get(pred_id)['skipped_steps'].append(current_step)
                else:
                    should_calc = True
                    self.magcache_state.update(
                        pred_id,
                        accumulated_ratio=1.0,
                        accumulated_steps=0,
                        accumulated_err=0
                    )

        # EasyCache
        if self.enable_easycache and self.easycache_start_step <= current_step <= self.easycache_end_step:
            if pred_id is None:
                pred_id = self.easycache_state.new_prediction(cache_device=self.cache_device)
                should_calc = True
            else:
                state = self.easycache_state.get(pred_id)
                previous_raw_input = state.get('previous_raw_input')
                previous_raw_output = state.get('previous_raw_output')
                cache = state.get('cache')
                accumulated_error = state.get('accumulated_error')

                if previous_raw_input is not None and previous_raw_output is not None:
                    raw_input = x.clone()
                    # Calculate input change
                    raw_input_change = (raw_input - previous_raw_input.to(raw_input.device)).abs().mean()

                    accumulated_error += raw_input_change

                    # Predict output change
                    if accumulated_error < self.easycache_thresh:
                        should_calc = False
                        x = raw_input + cache.to(x.device)
                        self.easycache_state.get(pred_id)['skipped_steps'].append(current_step)
                    else:
                        should_calc = True
                        accumulated_error = 0.0
                else:
                    should_calc = True

        if should_calc:
            if self.enable_teacache or self.enable_magcache or self.enable_easycache:
                original_x = x.to(self.cache_device).clone()

            if hasattr(self, "dwpose_embedding") and unianim_data is not None:
                if unianim_data['start_percent'] <= current_step_percentage <= unianim_data['end_percent']:
                    dwpose_emb = unianim_data['dwpose']
                    x += dwpose_emb * unianim_data['strength']
            # arguments
            kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=freqs,
                context=context,
                context_lens=context_lens,
                clip_embed=clip_embed,
                current_step=current_step,
                video_attention_split_steps=self.video_attention_split_steps,
                camera_embed=camera_embed,
                audio_proj=audio_proj,
                audio_context_lens=audio_context_lens,
                num_latent_frames = F,
                enhance_enabled=enhance_enabled,
                audio_scale=audio_scale,
                block_mask=self.block_mask,
                nag_params=nag_params,
                nag_context=nag_context,
                is_uncond = is_uncond,
                multitalk_audio_embedding=multitalk_audio_embedding if multitalk_audio is not None else None,
                ref_target_masks=token_ref_target_masks if multitalk_audio is not None else None,
                human_num=human_num if multitalk_audio is not None else 0
                )
            
            if vace_data is not None:
                vace_hint_list = []
                vace_scale_list = []
                if isinstance(vace_data[0], dict):
                    for data in vace_data:
                        if (data["start"] <= current_step_percentage <= data["end"]) or \
                            (data["end"] > 0 and current_step == 0 and current_step_percentage >= data["start"]):

                            vace_hints = self.forward_vace(x, data["context"], data["seq_len"], kwargs)
                            vace_hint_list.append(vace_hints)
                            vace_scale_list.append(data["scale"][current_step])
                else:
                    vace_hints = self.forward_vace(x, vace_data, seq_len, kwargs)
                    vace_hint_list.append(vace_hints)
                    vace_scale_list.append(1.0)
                
                kwargs['vace_hints'] = vace_hint_list
                kwargs['vace_context_scale'] = vace_scale_list

            #uni3c controlnet
            pdc_controlnet_states = None
            if pcd_data is not None:
                if (pcd_data["start"] <= current_step_percentage <= pcd_data["end"]) or \
                            (pcd_data["end"] > 0 and current_step == 0 and current_step_percentage >= pcd_data["start"]):
                    self.controlnet.to(self.main_device)
                    with torch.autocast(device_type=mm.get_autocast_device(device), dtype=x.dtype, enabled=True):
                        pdc_controlnet_states = self.controlnet(
                            render_latent=render_latent.to(self.main_device, self.controlnet.dtype), 
                            render_mask=pcd_data["render_mask"], 
                            camera_embedding=pcd_data["camera_embedding"], 
                            temb=e.to(self.main_device),
                            device=self.offload_device)
                    self.controlnet.to(self.offload_device)

            for b, block in enumerate(self.blocks):
                #skip layer guidance
                if self.slg_blocks is not None:
                    if b in self.slg_blocks and is_uncond:
                        if self.slg_start_percent <= current_step_percentage <= self.slg_end_percent:
                            continue
                if b <= self.blocks_to_swap and self.blocks_to_swap >= 0:
                    block.to(self.main_device)
                x = block(x, **kwargs)

                #uni3c controlnet
                if pdc_controlnet_states is not None and b < len(pdc_controlnet_states):
                    x[:, :x_len] += pdc_controlnet_states[b].to(x) * pcd_data["controlnet_weight"]
                #controlnet
                if (controlnet is not None) and (b % controlnet["controlnet_stride"] == 0) and (b // controlnet["controlnet_stride"] < len(controlnet["controlnet_states"])):
                    x[:, :x_len] += controlnet["controlnet_states"][b // controlnet["controlnet_stride"]].to(x) * controlnet["controlnet_weight"]

                if b <= self.blocks_to_swap and self.blocks_to_swap >= 0:
                    block.to(self.offload_device, non_blocking=self.use_non_blocking)

            if self.enable_teacache and (self.teacache_start_step <= current_step <= self.teacache_end_step) and pred_id is not None:
                self.teacache_state.update(
                    pred_id,
                    previous_residual=(x.to(original_x.device) - original_x),
                    accumulated_rel_l1_distance=accumulated_rel_l1_distance,
                    previous_modulated_input=previous_modulated_input
                )
            elif self.enable_magcache and (self.magcache_start_step <= current_step <= self.magcache_end_step) and pred_id is not None:
                self.magcache_state.update(
                    pred_id,
                    residual_cache=(x.to(original_x.device) - original_x)
                )
            elif self.enable_easycache and (self.easycache_start_step <= current_step <= self.easycache_end_step) and pred_id is not None:
                self.easycache_state.update(
                    pred_id,
                    previous_raw_input=original_x,
                    previous_raw_output=x.clone(),
                    cache=x.to(original_x.device) - original_x,
                    accumulated_error=0.0
                )
                
        if self.ref_conv is not None and fun_ref is not None:
            full_ref_length = fun_ref.size(1)
            x = x[:, full_ref_length:]
            grid_sizes = torch.stack([torch.tensor([u[0] - 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        if attn_cond is not None:
            x = x[:, :x_len]
            grid_sizes = torch.stack([torch.tensor([u[0] - 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        x = self.head(x, e.to(x.device))
        x = self.unpatchify(x, grid_sizes) # type: ignore[arg-type]
        x = [u.float() for u in x]
        return (x, pred_id) if pred_id is not None else (x, None)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out
