#based on ComfyUI's and MinusZoneAI's fp8_linear optimization

import torch
import torch.nn as nn

def fp8_linear_forward(cls, original_dtype, input):
    weight_dtype = cls.weight.dtype
    if weight_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        if len(input.shape) == 3:
            #target_dtype = torch.float8_e5m2 if weight_dtype == torch.float8_e4m3fn else torch.float8_e4m3fn
            inn = input.reshape(-1, input.shape[2]).to(weight_dtype)
            w = cls.weight.t()

            scale = torch.ones((1), device=input.device, dtype=torch.float32)
            bias = cls.bias.to(original_dtype) if cls.bias is not None else None

            if bias is not None:
                o = torch._scaled_mm(inn, w, out_dtype=original_dtype, bias=bias, scale_a=scale, scale_b=scale)
            else:
                o = torch._scaled_mm(inn, w, out_dtype=original_dtype, scale_a=scale, scale_b=scale)

            if isinstance(o, tuple):
                o = o[0]

            return o.reshape((-1, input.shape[1], cls.weight.shape[0]))
        else:
            return cls.original_forward(input.to(original_dtype))
    else:
        return cls.original_forward(input)
    
def fp8_scaled_linear_forward(cls, original_dtype, input):
    weight = cls.weight.to(original_dtype)
    scale_weight = cls.scale_weight.to(input.device)
    bias = cls.bias.to(original_dtype) if cls.bias is not None else None

    if weight.numel() < input.numel():
        weight = weight * scale_weight
    else:
        input = input * scale_weight

    lora = getattr(cls, "lora", None)
    if lora is not None:
        for lora_diff, lora_strength in zip(lora[0], lora[1]):
            patch_diff = torch.mm(
                lora_diff[0].flatten(start_dim=1).to(weight.device),
                lora_diff[1].flatten(start_dim=1).to(weight.device)
            ).reshape(weight.shape)
            alpha = lora_diff[2] / lora_diff[1].shape[0] if lora_diff[2] is not None else 1.0
            scale = lora_strength * alpha
            weight = weight.add(patch_diff, alpha=scale).to(original_dtype)

    return torch.nn.functional.linear(input, weight, bias)

def linear_with_lora_forward(cls, original_dtype, input):
    weight = cls.weight.to(original_dtype)
    bias = cls.bias.to(original_dtype) if cls.bias is not None else None

    lora = getattr(cls, "lora", None)
    if lora is not None:
        for lora_diff, lora_strength in zip(lora[0], lora[1]):
            patch_diff = torch.mm(
                lora_diff[0].flatten(start_dim=1).to(weight.device),
                lora_diff[1].flatten(start_dim=1).to(weight.device)
            ).reshape(weight.shape)
            alpha = lora_diff[2] / lora_diff[1].shape[0] if lora_diff[2] is not None else 1.0
            scale = lora_strength * alpha
            weight = weight.add(patch_diff, alpha=scale).to(original_dtype)

    return torch.nn.functional.linear(input, weight, bias)
 

def convert_fp8_linear(module, original_dtype, params_to_keep={}):
    setattr(module, "fp8_matmul_enabled", True)
   
    for name, submodule in module.named_modules():
        if not any(keyword in name for keyword in params_to_keep):
            if isinstance(submodule, nn.Linear):
                original_forward = submodule.forward
                setattr(submodule, "original_forward", original_forward)
                setattr(submodule, "forward", lambda input, m=submodule: fp8_linear_forward(m, original_dtype, input))

def convert_fp8_scaled_linear(module, sd, original_dtype, params_to_keep={}, patches=None):
    setattr(module, "fp8_scaled_enabled", True)
   
    for name, submodule in module.named_modules():
        if not any(keyword in name for keyword in params_to_keep):
            scale_key = f"{name}.scale_weight"
            has_scale = scale_key in sd
            weight = getattr(submodule, 'weight', None)
            has_fp8_weight = weight is not None and weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
            if has_scale:
                setattr(submodule, "scale_weight", sd[scale_key])
            
            if patches is not None:
                patch_key = f"diffusion_model.{name}.weight"
                patch = patches.get(patch_key, [])
                #print("Patches for", patch_key, ":", patch)
                if len(patch) != 0:
                    lora_diffs = []
                    for p in patch:
                        lora_obj = p[1]
                        if hasattr(lora_obj, "weights"):
                            lora_diffs.append(lora_obj.weights)
                        elif isinstance(lora_obj, tuple) and lora_obj[0] == "diff":
                            lora_diffs.append(lora_obj[1])
                        else:
                            continue
                        
                    lora_strengths = [p[0] for p in patch]
                    lora = (lora_diffs, lora_strengths)
                    setattr(submodule, "lora", lora)

            if isinstance(submodule, nn.Linear) and (has_scale and has_fp8_weight):
                original_forward = submodule.forward
                setattr(submodule, "original_forward", original_forward)
                setattr(submodule, "forward", lambda input, m=submodule: fp8_scaled_linear_forward(m, original_dtype, input))

def convert_linear_with_lora(module, original_dtype, patches=None):
    for name, submodule in module.named_modules():
        if isinstance(submodule, nn.Linear):
            patch_key = f"diffusion_model.{name}.weight"
            patch = patches.get(patch_key, [])
            if len(patch) != 0:
                lora_diffs = []
                for p in patch:
                    lora_obj = p[1]
                    if hasattr(lora_obj, "weights"):
                        lora_diffs.append(lora_obj.weights)
                    elif isinstance(lora_obj, tuple) and lora_obj[0] == "diff":
                        lora_diffs.append(lora_obj[1])
                    else:
                        continue
                    
                lora_strengths = [p[0] for p in patch]
                lora = (lora_diffs, lora_strengths)
                setattr(submodule, "lora", lora)
                # original_forward = submodule.forward
                # setattr(submodule, "original_forward", original_forward)
                setattr(submodule, "forward", lambda input, m=submodule: linear_with_lora_forward(m, original_dtype, input))
