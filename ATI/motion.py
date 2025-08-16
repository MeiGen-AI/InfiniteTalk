# Copyright (c) 2024-2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

def process_tracks(tracks_np: np.ndarray, frame_size: Tuple[int, int], quant_multi: int = 8, **kwargs):
    # tracks: shape [t, h, w, 3] => samples align with 24 fps, model trained with 16 fps.
    # frame_size: tuple (W, H)

    tracks = torch.from_numpy(tracks_np).float()
    
    if tracks.shape[1] == 121:
        tracks = torch.permute(tracks, (1, 0, 2, 3))
    
    tracks, visibles = tracks[..., :2], tracks[..., 2:3]
    short_edge = min(*frame_size)

    tracks = tracks - torch.tensor([*frame_size]).type_as(tracks) / 2
    tracks = tracks / short_edge * 2

    visibles = visibles * 2 - 1

    trange = torch.linspace(-1, 1, tracks.shape[0]).view(-1, 1, 1, 1).expand(*visibles.shape)
    
    out_ = torch.cat([trange, tracks, visibles], dim=-1).view(121, -1, 4)
    out_0 = out_[:1]
    out_l = out_[1:] # 121 => 120 | 1
    out_l = torch.repeat_interleave(out_l, 2, dim=0)[1::3]  # 120 => 240 => 80
    return torch.cat([out_0, out_l], dim=0)
