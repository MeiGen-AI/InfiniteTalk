# source https://github.com/TheDenk/wan2.1-dilated-controlnet/blob/main/wan_controlnet.py
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_wan import (
    WanTimeTextImageEmbedding, 
    WanRotaryPosEmbed, 
    WanTransformerBlock
)


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class WanControlnet(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    r"""
    A Controlnet Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        vae_channels (`int`, defaults to `16`):
            The number of channels in the vae input.
        in_channels (`int`, defaults to `16`):
            The number of channels in the controlnet input.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
        downscale_coef (`int`, *optional*, defaults to `8`):
            Coeficient for downscale controlnet input video.
        out_proj_dim (`int`, *optional*, defaults to `128 * 12`):
            Output projection dimention for last linear layers.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    
    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 3,
        vae_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 20,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        downscale_coef: int = 8,
        out_proj_dim: int = 128 * 12,
    ) -> None:
        super().__init__()

        start_channels = in_channels * (downscale_coef ** 2)
        input_channels = [start_channels, start_channels // 2, start_channels // 4]

        self.control_encoder = nn.ModuleList([
            ## Spatial compression with time awareness
            nn.Sequential(
                nn.Conv3d(
                    in_channels, 
                    input_channels[0], 
                    kernel_size=(3, downscale_coef  + 1, downscale_coef + 1),
                    stride=(1, downscale_coef, downscale_coef), 
                    padding=(1, downscale_coef // 2, downscale_coef // 2)
                ),
                nn.GELU(approximate="tanh"),
                nn.GroupNorm(2, input_channels[0]),
            ),
            ## Temporal compression with spatial awareness
            nn.Sequential(
                nn.Conv3d(input_channels[0], input_channels[1], kernel_size=3, stride=(2, 1, 1), padding=1),
                nn.GELU(approximate="tanh"),
                nn.GroupNorm(2, input_channels[1]),
            ),
            ## Temporal compression with spatial awareness
            nn.Sequential(
                nn.Conv3d(input_channels[1], input_channels[2], kernel_size=3, stride=(2, 1, 1), padding=1),
                nn.GELU(approximate="tanh"),
                nn.GroupNorm(2, input_channels[2]),
            )
        ])
        
        inner_dim = num_attention_heads * attention_head_dim
        
        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(vae_channels + input_channels[2], inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
        )
        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 4 Controlnet modules
        self.controlnet_blocks = nn.ModuleList([])

        for _ in range(len(self.blocks)):
            controlnet_block = nn.Linear(inner_dim, out_proj_dim)
            controlnet_block = zero_module(controlnet_block)
            self.controlnet_blocks.append(controlnet_block)
            
        self.gradient_checkpointing = False
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        controlnet_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        rotary_emb = self.rope(hidden_states)

        # 0. Controlnet encoder
        for control_encoder_block in self.control_encoder:
            controlnet_states = control_encoder_block(controlnet_states)
            
        hidden_states = torch.cat([hidden_states, controlnet_states], dim=1)

        ## 1. Patch embedding and stack
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)
        
        # 2. Transformer blocks
        controlnet_hidden_states = ()
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block, controlnet_block in zip(self.blocks, self.controlnet_blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
                controlnet_hidden_states += (controlnet_block(hidden_states),)
        else:
            for block, controlnet_block in zip(self.blocks, self.controlnet_blocks):
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
                controlnet_hidden_states += (controlnet_block(hidden_states),)


        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (controlnet_hidden_states,)

        return Transformer2DModelOutput(sample=controlnet_hidden_states)
    

if __name__ == "__main__":
    parameters = {
        "added_kv_proj_dim": None,
        "attention_head_dim": 128,
        "cross_attn_norm": True,
        "eps": 1e-06,
        "ffn_dim": 8960,
        "freq_dim": 256,
        "image_dim": None,
        "in_channels": 3,
        "num_attention_heads": 12,
        "num_layers": 2,
        "patch_size": [1, 2, 2],
        "qk_norm": "rms_norm_across_heads",
        "rope_max_seq_len": 1024,
        "text_dim": 4096,
        "downscale_coef": 8,
        "out_proj_dim": 12 * 128,
    }
    controlnet = WanControlnet(**parameters)

    hidden_states = torch.rand(1, 16, 21, 60, 90)
    timestep = torch.randint(low=0, high=1000, size=(1,), dtype=torch.long)
    encoder_hidden_states = torch.rand(1, 512, 4096)
    controlnet_states = torch.rand(1, 3, 81, 480, 720)

    controlnet_hidden_states = controlnet(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        controlnet_states=controlnet_states,
        return_dict=False
    )
    print("Output states count", len(controlnet_hidden_states[0]))
    for out_hidden_states in controlnet_hidden_states[0]:
        print(out_hidden_states.shape)
    