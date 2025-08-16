import numpy as np
from typing import Callable, Optional, List
import torch

def ordered_halving(val):
    bin_str = f"{val:064b}"
    bin_flip = bin_str[::-1]
    as_int = int(bin_flip, 2)

    return as_int / (1 << 64)

def does_window_roll_over(window: list[int], num_frames: int) -> tuple[bool, int]:
    prev_val = -1
    for i, val in enumerate(window):
        val = val % num_frames
        if val < prev_val:
            return True, i
        prev_val = val
    return False, -1

def shift_window_to_start(window: list[int], num_frames: int):
    start_val = window[0]
    for i in range(len(window)):
        # 1) subtract each element by start_val to move vals relative to the start of all frames
        # 2) add num_frames and take modulus to get adjusted vals
        window[i] = ((window[i] - start_val) + num_frames) % num_frames

def shift_window_to_end(window: list[int], num_frames: int):
    # 1) shift window to start
    shift_window_to_start(window, num_frames)
    end_val = window[-1]
    end_delta = num_frames - end_val - 1
    for i in range(len(window)):
        # 2) add end_delta to each val to slide windows to end
        window[i] = window[i] + end_delta

def get_missing_indexes(windows: list[list[int]], num_frames: int) -> list[int]:
    all_indexes = list(range(num_frames))
    for w in windows:
        for val in w:
            try:
                all_indexes.remove(val)
            except ValueError:
                pass
    return all_indexes

def uniform_looped(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    if num_frames <= context_size:
        yield list(range(num_frames))
        return

    context_stride = min(context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1)

    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(step)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            yield [e % num_frames for e in range(j, j + context_size * context_step, context_step)]

#from AnimateDiff-Evolved by Kosinkadink (https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved)
def uniform_standard(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    windows = []
    if num_frames <= context_size:
        windows.append(list(range(num_frames)))
        return windows

    context_stride = min(context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1)

    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(step)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            windows.append([e % num_frames for e in range(j, j + context_size * context_step, context_step)])

    # now that windows are created, shift any windows that loop, and delete duplicate windows
    delete_idxs = []
    win_i = 0
    while win_i < len(windows):
        # if window is rolls over itself, need to shift it
        is_roll, roll_idx = does_window_roll_over(windows[win_i], num_frames)
        if is_roll:
            roll_val = windows[win_i][roll_idx]  # roll_val might not be 0 for windows of higher strides
            shift_window_to_end(windows[win_i], num_frames=num_frames)
            # check if next window (cyclical) is missing roll_val
            if roll_val not in windows[(win_i+1) % len(windows)]:
                # need to insert new window here - just insert window starting at roll_val
                windows.insert(win_i+1, list(range(roll_val, roll_val + context_size)))
        # delete window if it's not unique
        for pre_i in range(0, win_i):
            if windows[win_i] == windows[pre_i]:
                delete_idxs.append(win_i)
                break
        win_i += 1

    # reverse delete_idxs so that they will be deleted in an order that doesn't break idx correlation
    delete_idxs.reverse()
    for i in delete_idxs:
        windows.pop(i)
    return windows

def static_standard(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    windows = []
    if num_frames <= context_size:
        windows.append(list(range(num_frames)))
        return windows
    # always return the same set of windows
    delta = context_size - context_overlap
    for start_idx in range(0, num_frames, delta):
        # if past the end of frames, move start_idx back to allow same context_length
        ending = start_idx + context_size
        if ending >= num_frames:
            final_delta = ending - num_frames
            final_start_idx = start_idx - final_delta
            windows.append(list(range(final_start_idx, final_start_idx + context_size)))
            break
        windows.append(list(range(start_idx, start_idx + context_size)))
    return windows

def get_context_scheduler(name: str) -> Callable:
    if name == "uniform_looped":
        return uniform_looped
    elif name == "uniform_standard":
        return uniform_standard
    elif name == "static_standard":
        return static_standard
    else:
        raise ValueError(f"Unknown context_overlap policy {name}")


def get_total_steps(
    scheduler,
    timesteps: List[int],
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    return sum(
        len(
            list(
                scheduler(
                    i,
                    num_steps,
                    num_frames,
                    context_size,
                    context_stride,
                    context_overlap,
                )
            )
        )
        for i in range(len(timesteps))
    )

def create_window_mask(noise_pred_context, c, latent_video_length, context_overlap, looped=False, window_type="linear"):
    window_mask = torch.ones_like(noise_pred_context)
    
    if window_type == "pyramid":
        # Create pyramid weights that peak in the middle
        length = noise_pred_context.shape[1]
        if length % 2 == 0:
            max_weight = length // 2
            weight_sequence = list(range(1, max_weight + 1, 1)) + list(range(max_weight, 0, -1))
        else:
            max_weight = (length + 1) // 2
            weight_sequence = list(range(1, max_weight, 1)) + [max_weight] + list(range(max_weight - 1, 0, -1))
        
        # Normalize weights to range from 0 to 1
        max_val = max(weight_sequence)
        weight_sequence = [w / max_val for w in weight_sequence]
        
        # Apply the weights to create the mask
        weights_tensor = torch.tensor(weight_sequence, device=noise_pred_context.device)
        weights_tensor = weights_tensor.view(1, -1, 1, 1)
        window_mask = weights_tensor.expand_as(window_mask).clone()
        
        # Adjust for position in sequence if needed
        if not looped:
            if min(c) == 0:  # First chunk
                left_ramp = torch.linspace(0, 1, context_overlap, device=noise_pred_context.device).view(1, -1, 1, 1)
                # Clone to avoid in-place memory conflict
                left_section = window_mask[:, :context_overlap].clone()
                window_mask[:, :context_overlap] = torch.maximum(left_section, left_ramp)
                
            if max(c) == latent_video_length - 1:  # Last chunk
                right_ramp = torch.linspace(1, 0, context_overlap, device=noise_pred_context.device).view(1, -1, 1, 1)
                # Clone to avoid in-place memory conflict
                right_section = window_mask[:, -context_overlap:].clone()
                window_mask[:, -context_overlap:] = torch.maximum(right_section, right_ramp)
    else:  # Original "linear" window masking
        # Apply left-side blending for all except first chunk (or always in loop mode)
        if min(c) > 0 or (looped and max(c) == latent_video_length - 1):
            ramp_up = torch.linspace(0, 1, context_overlap, device=noise_pred_context.device)
            ramp_up = ramp_up.view(1, -1, 1, 1)
            window_mask[:, :context_overlap] = ramp_up
            
        # Apply right-side blending for all except last chunk (or always in loop mode)
        if max(c) < latent_video_length - 1 or (looped and min(c) == 0):
            ramp_down = torch.linspace(1, 0, context_overlap, device=noise_pred_context.device)
            ramp_down = ramp_down.view(1, -1, 1, 1)
            window_mask[:, -context_overlap:] = ramp_down
            
    return window_mask

class WindowTracker:
    def __init__(self, verbose=False):
        self.window_map = {}  # Maps frame sequence to persistent ID
        self.next_id = 0
        self.cache_states = {}  # Maps persistent ID to teacache state
        self.verbose = verbose
    
    def get_window_id(self, frames):
        key = tuple(sorted(frames))  # Order-independent frame sequence
        if key not in self.window_map:
            self.window_map[key] = self.next_id
            if self.verbose:
                log.info(f"New window pattern {key} -> ID {self.next_id}")
            self.next_id += 1
        return self.window_map[key]
    
    def get_teacache(self, window_id, base_state):
        if window_id not in self.cache_states:
            if self.verbose:
                log.info(f"Initializing persistent teacache for window {window_id}")
            self.cache_states[window_id] = base_state.copy()
        return self.cache_states[window_id]
