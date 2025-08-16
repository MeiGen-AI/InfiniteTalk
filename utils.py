import importlib.metadata
import torch
import logging
from tqdm import tqdm
import types
from comfy.utils import ProgressBar, copy_to_param, set_attr_param
from comfy.model_patcher import get_key_weight, string_to_seed
from comfy.lora import calculate_weight
from comfy.model_management import cast_to_device
from comfy.float import stochastic_rounding
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

from accelerate.utils import set_module_tensor_to_device
def check_diffusers_version():
    try:
        version = importlib.metadata.version('diffusers')
        required_version = '0.31.0'
        if version < required_version:
            raise AssertionError(f"diffusers version {version} is installed, but version {required_version} or higher is required.")
    except importlib.metadata.PackageNotFoundError:
        raise AssertionError("diffusers is not installed.")

def print_memory(device):
    memory = torch.cuda.memory_allocated(device) / 1024**3
    max_memory = torch.cuda.max_memory_allocated(device) / 1024**3
    max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    log.info(f"Allocated memory: {memory=:.3f} GB")
    log.info(f"Max allocated memory: {max_memory=:.3f} GB")
    log.info(f"Max reserved memory: {max_reserved=:.3f} GB")
    #memory_summary = torch.cuda.memory_summary(device=device, abbreviated=False)
    #log.info(f"Memory Summary:\n{memory_summary}")

def get_module_memory_mb(module):
    memory = 0
    for param in module.parameters():
        if param.data is not None:
            memory += param.nelement() * param.element_size()
    return memory / (1024 * 1024)  # Convert to MB

def get_tensor_memory(tensor):
    memory_bytes = tensor.element_size() * tensor.nelement()
    return f"{memory_bytes / (1024 * 1024):.2f} MB"

def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
    if key not in self.patches:
        return

    weight, set_func, convert_func = get_key_weight(self.model, key)
    inplace_update = self.weight_inplace_update or inplace_update

    if device_to is not None:
        temp_weight = cast_to_device(weight, device_to, torch.float32, copy=True)
    else:
        temp_weight = weight.to(torch.float32, copy=True)
    if convert_func is not None:
        temp_weight = convert_func(temp_weight, inplace=True)

    out_weight = calculate_weight(self.patches[key], temp_weight, key)
    if set_func is None:
        out_weight = stochastic_rounding(out_weight, weight.dtype, seed=string_to_seed(key))
        if inplace_update:
            copy_to_param(self.model, key, out_weight)
        else:
            set_attr_param(self.model, key, out_weight)
    else:
        set_func(out_weight, inplace_update=inplace_update, seed=string_to_seed(key))

def apply_lora(model, device_to, transformer_load_device, params_to_keep=None, dtype=None, base_dtype=None, state_dict=None, low_mem_load=False):
        model.patch_weight_to_device = types.MethodType(patch_weight_to_device, model)
        to_load = []
        for n, m in model.model.named_modules():
            params = []
            skip = False
            for name, param in m.named_parameters(recurse=False):
                params.append(name)
            for name, param in m.named_parameters(recurse=True):
                if name not in params:
                    skip = True # skip random weights in non leaf modules
                    break
            if not skip and (hasattr(m, "comfy_cast_weights") or len(params) > 0):
                to_load.append((n, m, params))

        to_load.sort(reverse=True)
        pbar = ProgressBar(len(to_load))
        for x in tqdm(to_load, desc="Loading model and applying LoRA weights:", leave=True):
            name = x[0]
            m = x[1]
            params = x[2]
            if hasattr(m, "comfy_patched_weights"):
                if m.comfy_patched_weights == True:
                    continue
            for param in params:
                name = name.replace("._orig_mod.", ".") # torch compiled modules have this prefix
                if low_mem_load:
                    dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype
                    if "patch_embedding" in name:
                        dtype_to_use = torch.float32
                    if name.startswith("diffusion_model."):
                        name_no_prefix = name[len("diffusion_model."):]
                    key = "{}.{}".format(name_no_prefix, param)
                    try:
                        set_module_tensor_to_device(model.model.diffusion_model, key, device=transformer_load_device, dtype=dtype_to_use, value=state_dict[key])
                    except:
                        continue
                if low_mem_load:
                    model.patch_weight_to_device("{}.{}".format(name, param), device_to=device_to, inplace_update=True)
                else:
                    model.patch_weight_to_device("{}.{}".format(name, param), device_to=device_to)
                    model.backup["{}.{}".format(name, param)] = None
                    set_module_tensor_to_device(m, param, device=transformer_load_device)
                if low_mem_load:
                    try:
                        set_module_tensor_to_device(model.model.diffusion_model, key, device=transformer_load_device, dtype=dtype_to_use, value=model.model.diffusion_model.state_dict()[key])
                    except:
                        continue
            m.comfy_patched_weights = True
            pbar.update(1)
      
        model.current_weight_patches_uuid = model.patches_uuid
        if low_mem_load:
            for name, param in model.model.diffusion_model.named_parameters():
                if param.device != transformer_load_device:
                    dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype
                    if "patch_embedding" in name:
                        dtype_to_use = torch.float32
                    try:
                        set_module_tensor_to_device(model.model.diffusion_model, name, device=transformer_load_device, dtype=dtype_to_use, value=state_dict[name])
                    except:
                        continue
        return model


# from https://github.com/cubiq/ComfyUI_IPAdapter_plus/blob/9d076a3df0d2763cef5510ec5ab807f6632c39f5/utils.py#L181
def split_tiles(embeds, num_split):
    _, H, W, _ = embeds.shape
    out = []
    for x in embeds:
        x = x.unsqueeze(0)
        h, w = H // num_split, W // num_split
        x_split = torch.cat([x[:, i*h:(i+1)*h, j*w:(j+1)*w, :] for i in range(num_split) for j in range(num_split)], dim=0)    
        out.append(x_split)
    
    x_split = torch.stack(out, dim=0)
    
    return x_split

def merge_hiddenstates(x, tiles):
    chunk_size = tiles*tiles
    x = x.split(chunk_size)

    out = []
    for embeds in x:
        num_tiles = embeds.shape[0]
        tile_size = int((embeds.shape[1]-1) ** 0.5)
        grid_size = int(num_tiles ** 0.5)

        # Extract class tokens
        class_tokens = embeds[:, 0, :]  # Save class tokens: [num_tiles, embeds[-1]]
        avg_class_token = class_tokens.mean(dim=0, keepdim=True).unsqueeze(0)  # Average token, shape: [1, 1, embeds[-1]]

        patch_embeds = embeds[:, 1:, :]  # Shape: [num_tiles, tile_size^2, embeds[-1]]
        reshaped = patch_embeds.reshape(grid_size, grid_size, tile_size, tile_size, embeds.shape[-1])

        merged = torch.cat([torch.cat([reshaped[i, j] for j in range(grid_size)], dim=1) 
                            for i in range(grid_size)], dim=0)
        
        merged = merged.unsqueeze(0)  # Shape: [1, grid_size*tile_size, grid_size*tile_size, embeds[-1]]
        
        # Pool to original size
        pooled = torch.nn.functional.adaptive_avg_pool2d(merged.permute(0, 3, 1, 2), (tile_size, tile_size)).permute(0, 2, 3, 1)
        flattened = pooled.reshape(1, tile_size*tile_size, embeds.shape[-1])
        
        # Add back the class token
        with_class = torch.cat([avg_class_token, flattened], dim=1)  # Shape: original shape
        out.append(with_class)
    
    out = torch.cat(out, dim=0)

    return out

from comfy.clip_vision import clip_preprocess, ClipVisionModel

def clip_encode_image_tiled(clip_vision, image, tiles=1, ratio=1.0):
    embeds = encode_image_(clip_vision, image)
    tiles = min(tiles, 16)

    if tiles > 1:
        # split in tiles
        image_split = split_tiles(image, tiles)

        # get the embeds for each tile
        embeds_split = {}
        for i in image_split:
            encoded = encode_image_(clip_vision, i)
            if not hasattr(embeds_split, "last_hidden_state"):
                embeds_split["last_hidden_state"] = encoded
            else:
                embeds_split["last_hidden_state"] = torch.cat(embeds_split["last_hidden_state"], encoded, dim=0)

        embeds_split['last_hidden_state'] = merge_hiddenstates(embeds_split['last_hidden_state'], tiles)

        if embeds.shape[0] > 1: # if we have more than one image we need to average the embeddings for consistency
            embeds = embeds * ratio + embeds_split['last_hidden_state']*(1-ratio)
        else: # otherwise we can concatenate them, they can be averaged later
            embeds = torch.cat([embeds * ratio, embeds_split['last_hidden_state']])

    return embeds

def encode_image_(clip_vision, image):
    if isinstance(clip_vision, ClipVisionModel):
        out = clip_vision.encode_image(image).last_hidden_state
    else:
        pixel_values = clip_preprocess(image, size=224, crop=True).float()
        out = clip_vision.visual(pixel_values)

    return out

# Code based on https://github.com/WikiChao/FreSca (MIT License)
import torch
import torch.fft as fft

def fourier_filter(x, scale_low=1.0, scale_high=1.5, freq_cutoff=20):
    """
    Apply frequency-dependent scaling to an image tensor using Fourier transforms.

    Parameters:
        x:           Input tensor of shape (B, C, H, W)
        scale_low:   Scaling factor for low-frequency components (default: 1.0)
        scale_high:  Scaling factor for high-frequency components (default: 1.5)
        freq_cutoff: Number of frequency indices around center to consider as low-frequency (default: 20)

    Returns:
        x_filtered: Filtered version of x in spatial domain with frequency-specific scaling applied.
    """
    # Preserve input dtype and device
    dtype, device = x.dtype, x.device

    # Convert to float32 for FFT computations
    x = x.to(torch.float32)

    # 1) Apply FFT and shift low frequencies to center
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    # 2) Create a mask to scale frequencies differently
    C, B, H, W = x_freq.shape
    crow, ccol = H // 2, W // 2

    # Initialize mask with high-frequency scaling factor
    mask = torch.ones((C, B, H, W), device=device) * scale_high

    # Apply low-frequency scaling factor to center region
    mask[
        ...,
        crow - freq_cutoff : crow + freq_cutoff,
        ccol - freq_cutoff : ccol + freq_cutoff,
    ] = scale_low

    # 3) Apply frequency-specific scaling
    x_freq = x_freq * mask

    # 4) Convert back to spatial domain
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    # 5) Restore original dtype
    x_filtered = x_filtered.to(dtype)

    return x_filtered

def is_image_black(image, threshold=1e-3):
    if image.min() < 0:
        image = (image + 1) / 2
    return torch.all(image < threshold).item()

def add_noise_to_reference_video(image, ratio=None):
    sigma = torch.ones((image.shape[0],)).to(image.device, image.dtype) * ratio 
    image_noise = torch.randn_like(image) * sigma[:, None, None, None]
    image_noise = torch.where(image==-1, torch.zeros_like(image), image_noise)
    image = image + image_noise
    return image

def optimized_scale(positive_flat, negative_flat):

    # Calculate dot production
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8

    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    st_star = dot_product / squared_norm
    
    return st_star

def find_closest_valid_dim(fixed_dim, var_dim, block_size):
    for delta in range(1, 17):
        for sign in [-1, 1]:
            candidate = var_dim + sign * delta
            if candidate > 0 and ((fixed_dim * candidate) // 4) % block_size == 0:
                return candidate
    return var_dim