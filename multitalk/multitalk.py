from diffusers import ModelMixin, ConfigMixin
from einops import rearrange, repeat
import torch
import torch.nn as nn
from functools import lru_cache
from ..wanvideo.modules.attention import attention

from comfy import model_management as mm

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

def add_noise(
    original_samples: torch.FloatTensor,
    noise: torch.FloatTensor,
    timesteps: torch.IntTensor,
) -> torch.FloatTensor:
    """
    compatible with diffusers add_noise()
    """
    timesteps = timesteps.float() / 1000
    timesteps = timesteps.view(timesteps.shape + (1,) * (len(noise.shape)-1))

    return (1 - timesteps) * original_samples + timesteps * noise

def normalize_and_scale(column, source_range, target_range, epsilon=1e-8):

    source_min, source_max = source_range
    new_min, new_max = target_range
 
    normalized = (column - source_min) / (source_max - source_min + epsilon)
    scaled = normalized * (new_max - new_min) + new_min
    return scaled

def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")

def calculate_x_ref_attn_map(visual_q, ref_k, ref_target_masks, mode='mean', attn_bias=None):
    
    ref_k = ref_k.to(visual_q.dtype).to(visual_q.device)
    scale = 1.0 / visual_q.shape[-1] ** 0.5
    visual_q = visual_q * scale
    visual_q = visual_q.transpose(1, 2)
    ref_k = ref_k.transpose(1, 2)
    attn = visual_q @ ref_k.transpose(-2, -1)

    if attn_bias is not None:
        attn = attn + attn_bias

    x_ref_attn_map_source = attn.softmax(-1) # B, H, x_seqlens, ref_seqlens


    x_ref_attn_maps = []
    ref_target_masks = ref_target_masks.to(visual_q.dtype)
    x_ref_attn_map_source = x_ref_attn_map_source.to(visual_q.dtype)

    for class_idx, ref_target_mask in enumerate(ref_target_masks):
        mm.soft_empty_cache()
        ref_target_mask = ref_target_mask[None, None, None, ...]
        x_ref_attnmap = x_ref_attn_map_source * ref_target_mask
        x_ref_attnmap = x_ref_attnmap.sum(-1) / ref_target_mask.sum() # B, H, x_seqlens, ref_seqlens --> B, H, x_seqlens
        x_ref_attnmap = x_ref_attnmap.permute(0, 2, 1) # B, x_seqlens, H
       
        if mode == 'mean':
            x_ref_attnmap = x_ref_attnmap.mean(-1) # B, x_seqlens
        elif mode == 'max':
            x_ref_attnmap = x_ref_attnmap.max(-1) # B, x_seqlens
        
        x_ref_attn_maps.append(x_ref_attnmap)
    
    del attn
    del x_ref_attn_map_source
    mm.soft_empty_cache()

    return torch.concat(x_ref_attn_maps, dim=0)

def get_attn_map_with_target(visual_q, ref_k, shape, ref_target_masks=None, split_num=2, enable_sp=False):
    """Args:
        query (torch.tensor): B M H K
        key (torch.tensor): B M H K
        shape (tuple): (N_t, N_h, N_w)
        ref_target_masks: [B, N_h * N_w]
    """

    N_t, N_h, N_w = shape
    
    x_seqlens = N_h * N_w
    ref_k     = ref_k[:, :x_seqlens]
    _, seq_lens, heads, _ = visual_q.shape
    class_num, _ = ref_target_masks.shape
    x_ref_attn_maps = torch.zeros(class_num, seq_lens).to(visual_q.device).to(visual_q.dtype)

    split_chunk = heads // split_num
    
    for i in range(split_num):
        x_ref_attn_maps_perhead = calculate_x_ref_attn_map(visual_q[:, :, i*split_chunk:(i+1)*split_chunk, :], ref_k[:, :, i*split_chunk:(i+1)*split_chunk, :], ref_target_masks)
        x_ref_attn_maps += x_ref_attn_maps_perhead
    
    return x_ref_attn_maps / split_num

class RotaryPositionalEmbedding1D(nn.Module):

    def __init__(self,
                 head_dim,
                 ):
        super().__init__()
        self.head_dim = head_dim
        self.base = 10000


    @lru_cache(maxsize=32)
    def precompute_freqs_cis_1d(self, pos_indices):

        freqs = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2)[: (self.head_dim // 2)].float() / self.head_dim))
        freqs = freqs.to(pos_indices.device)
        freqs = torch.einsum("..., f -> ... f", pos_indices.float(), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        return freqs

    def forward(self, x, pos_indices):
        """1D RoPE.

        Args:
            query (torch.tensor): [B, head, seq, head_dim]
            pos_indices (torch.tensor): [seq,]
        Returns:
            query with the same shape as input.
        """
        freqs_cis = self.precompute_freqs_cis_1d(pos_indices)

        x_ = x.float()

        freqs_cis = freqs_cis.float().to(x.device)
        cos, sin = freqs_cis.cos(), freqs_cis.sin()
        cos, sin = rearrange(cos, 'n d -> 1 1 n d'), rearrange(sin, 'n d -> 1 1 n d')
        x_ = (x_ * cos) + (rotate_half(x_) * sin)

        return x_.type_as(x)

class AudioProjModel(ModelMixin, ConfigMixin):
    def __init__(
        self,
        seq_len=5,
        seq_len_vf=12,
        blocks=12,  
        channels=768, 
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
        norm_output_audio=False,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels  
        self.input_dim_vf = seq_len_vf * blocks * channels
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj1_vf = nn.Linear(self.input_dim_vf, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.norm = nn.LayerNorm(output_dim) if norm_output_audio else nn.Identity()

    def forward(self, audio_embeds, audio_embeds_vf):
        video_length = audio_embeds.shape[1] + audio_embeds_vf.shape[1]
        B, _, _, S, C = audio_embeds.shape

        # process audio of first frame
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        # process audio of latter frame
        audio_embeds_vf = rearrange(audio_embeds_vf, "bz f w b c -> (bz f) w b c")
        batch_size_vf, window_size_vf, blocks_vf, channels_vf = audio_embeds_vf.shape
        audio_embeds_vf = audio_embeds_vf.view(batch_size_vf, window_size_vf * blocks_vf * channels_vf)

        # first projection
        audio_embeds = torch.relu(self.proj1(audio_embeds)) 
        audio_embeds_vf = torch.relu(self.proj1_vf(audio_embeds_vf)) 
        audio_embeds = rearrange(audio_embeds, "(bz f) c -> bz f c", bz=B)
        audio_embeds_vf = rearrange(audio_embeds_vf, "(bz f) c -> bz f c", bz=B)
        audio_embeds_c = torch.concat([audio_embeds, audio_embeds_vf], dim=1) 
        batch_size_c, N_t, C_a = audio_embeds_c.shape
        audio_embeds_c = audio_embeds_c.view(batch_size_c*N_t, C_a)

        # second projection
        audio_embeds_c = torch.relu(self.proj2(audio_embeds_c))

        context_tokens = self.proj3(audio_embeds_c).reshape(batch_size_c*N_t, self.context_tokens, self.output_dim)

        # normalization and reshape
        context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(context_tokens, "(bz f) m c -> bz f m c", f=video_length)

        return context_tokens

#@torch.compiler.disable()
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
        attention_mode: str = 'sdpa',
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

        self.attention_mode = attention_mode

    def forward(self, x: torch.Tensor, encoder_hidden_states: torch.Tensor, shape=None, enable_sp=False, kv_seq=None) -> torch.Tensor:
       
        N_t, N_h, N_w = shape
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

        x = attention(
            q.transpose(1, 2), 
            encoder_k.transpose(1, 2), 
            encoder_v.transpose(1, 2),
            attention_mode=self.attention_mode
            )
        #x = torch.nn.functional.scaled_dot_product_attention(
        #    q, encoder_k, encoder_v, attn_mask=None, is_causal=False, dropout_p=0.0)

        # linear transform
        x_output_shape = (B, N, C)
        #x = x.transpose(1, 2) 
        x = x.reshape(x_output_shape) 
        x = self.proj(x)
        x = self.proj_drop(x)

        x = rearrange(x, "(B N_t) S C -> B (N_t S) C", N_t=N_t)

        return x
    
class SingleStreamMultiAttention(SingleStreamAttention):
    """Multi-speaker rotary-position cross-attention.

    This implementation generalises the original 2-speaker logic to an arbitrary
    number of voices.  Each speaker is allocated a contiguous *class_interval*
    segment inside a shared *class_range* rotary bucket.  The centre of each
    bucket is applied to that speaker's KV tokens while queries are modulated
    per-token according to which speaker dominates the pixel.
    """

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
        attention_mode: str = 'sdpa',
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
            attention_mode=attention_mode,
        )

        # Rotary-embedding layout parameters
        self.class_interval = class_interval
        self.class_range = class_range
        self.max_humans = self.class_range // self.class_interval

        # Constant bucket used for background tokens
        self.rope_bak = int(self.class_range // 2)

        self.rope_1d = RotaryPositionalEmbedding1D(self.head_dim)

        self.attention_mode = attention_mode

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        shape=None,
        x_ref_attn_map=None,
        human_num=None,
    ) -> torch.Tensor:
        encoder_hidden_states = encoder_hidden_states.squeeze(0)

        # Single-speaker fall-through
        if human_num is None or human_num <= 1:
            return super().forward(x, encoder_hidden_states, shape)

        N_t, _, _ = shape
        x = rearrange(x, "B (N_t S) C -> (B N_t) S C", N_t=N_t)

        # Query projection
        B, N, C = x.shape
        q = self.q_linear(x)
        q = q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        if self.qk_norm:
            q = self.q_norm(q)

        if human_num == 2:
            # Use `class_range` logic for exactly 2 speakers
            rope_h1 = (0, self.class_interval)
            rope_h2 = (self.class_range - self.class_interval, self.class_range)
            rope_bak = int(self.class_range // 2)

            # Normalize and scale attention maps for each speaker
            max_values = x_ref_attn_map.max(1).values[:, None, None]
            min_values = x_ref_attn_map.min(1).values[:, None, None]
            max_min_values = torch.cat([max_values, min_values], dim=2)

            human1_max_value, human1_min_value = max_min_values[0, :, 0].max(), max_min_values[0, :, 1].min()
            human2_max_value, human2_min_value = max_min_values[1, :, 0].max(), max_min_values[1, :, 1].min()

            human1 = normalize_and_scale(x_ref_attn_map[0], (human1_min_value, human1_max_value), rope_h1)
            human2 = normalize_and_scale(x_ref_attn_map[1], (human2_min_value, human2_max_value), rope_h2)
            back = torch.full((x_ref_attn_map.size(1),), rope_bak, dtype=human1.dtype, device=human1.device)

            # Token-wise speaker dominance
            max_indices = x_ref_attn_map.argmax(dim=0)
            normalized_map = torch.stack([human1, human2, back], dim=1)
            normalized_pos = normalized_map[torch.arange(x_ref_attn_map.size(1)), max_indices]
        else:
            # General case for more than 2 speakers
            rope_ranges = [
                (i * self.class_interval, (i + 1) * self.class_interval)
                for i in range(human_num)
            ]

            # Normalize each speaker's attention map into its own bucket
            human_norm_list = []
            for idx in range(human_num):
                attn_map = x_ref_attn_map[idx]
                att_min, att_max = attn_map.min(), attn_map.max()
                human_norm = normalize_and_scale(
                    attn_map, (att_min, att_max), rope_ranges[idx]
                )
                human_norm_list.append(human_norm)

            # Background constant bucket
            back = torch.full(
                (x_ref_attn_map.size(1),),
                self.rope_bak,
                dtype=x_ref_attn_map.dtype,
                device=x_ref_attn_map.device,
            )

            # Token-wise speaker dominance
            max_indices = x_ref_attn_map.argmax(dim=0)
            normalized_map = torch.stack(human_norm_list + [back], dim=1)
            normalized_pos = normalized_map[torch.arange(x_ref_attn_map.size(1)), max_indices]

        # Apply rotary to Q
        q = rearrange(q, "(B N_t) H S C -> B H (N_t S) C", N_t=N_t)
        q = self.rope_1d(q, normalized_pos)
        q = rearrange(q, "B H (N_t S) C -> (B N_t) H S C", N_t=N_t)

        # Keys / Values
        _, N_a, _ = encoder_hidden_states.shape
        encoder_kv = self.kv_linear(encoder_hidden_states)
        encoder_kv = encoder_kv.view(B, N_a, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        encoder_k, encoder_v = encoder_kv.unbind(0)
        if self.qk_norm:
            encoder_k = self.add_k_norm(encoder_k)

        # Rotary for keys – assign centre of each speaker bucket to its context tokens
        if human_num == 2:
            per_frame = torch.zeros(N_a, dtype=encoder_k.dtype, device=encoder_k.device)
            per_frame[: per_frame.size(0) // 2] = (rope_h1[0] + rope_h1[1]) / 2
            per_frame[per_frame.size(0) // 2 :] = (rope_h2[0] + rope_h2[1]) / 2
            encoder_pos = torch.cat([per_frame] * N_t, dim=0)
        else:
            tokens_per_human = N_a // human_num
            encoder_pos_list = []
            for i in range(human_num):
                start, end = rope_ranges[i]
                centre = (start + end) / 2
                encoder_pos_list.append(
                    torch.full(
                        (tokens_per_human,), centre, dtype=encoder_k.dtype, device=encoder_k.device
                    )
                )
            encoder_pos = torch.cat(encoder_pos_list * N_t, dim=0)

        encoder_k = rearrange(encoder_k, "(B N_t) H S C -> B H (N_t S) C", N_t=N_t)
        encoder_k = self.rope_1d(encoder_k, encoder_pos)
        encoder_k = rearrange(encoder_k, "B H (N_t S) C -> (B N_t) H S C", N_t=N_t)

        # Final attention
        q = rearrange(q, "B H M K -> B M H K")
        encoder_k = rearrange(encoder_k, "B H M K -> B M H K")
        encoder_v = rearrange(encoder_v, "B H M K -> B M H K")
        x = attention(
            q, encoder_k, encoder_v, attention_mode=self.attention_mode
        )

        # Linear projection
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Restore original layout
        x = rearrange(x, "(B N_t) S C -> B (N_t S) C", N_t=N_t)

        return x