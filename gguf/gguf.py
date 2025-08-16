import torch
import torch.nn as nn
from diffusers.quantizers.gguf.utils import GGUFParameter, dequantize_gguf_tensor
from diffusers.utils import is_accelerate_available
from contextlib import nullcontext

if is_accelerate_available():
    import accelerate
    from accelerate import init_empty_weights

@torch.compiler.disable()
def dequantize_without_compile(tensor):
    return dequantize_gguf_tensor(tensor)

#based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/quantizers/gguf/utils.py
def _replace_with_gguf_linear(model, compute_dtype, state_dict, prefix="", modules_to_not_convert=[], patches=None):
    def _should_convert_to_gguf(state_dict, prefix):
        weight_key = prefix + "weight"
        return weight_key in state_dict and isinstance(state_dict[weight_key], GGUFParameter)

    has_children = list(model.children())
    if not has_children:
        return

    for name, module in model.named_children():
        module_prefix = prefix + name + "."
        _replace_with_gguf_linear(module, compute_dtype, state_dict, module_prefix, modules_to_not_convert, patches)

        if (
            isinstance(module, nn.Linear)
            and _should_convert_to_gguf(state_dict, module_prefix)
            and name not in modules_to_not_convert
        ):
            key = "diffusion_model." + module_prefix + "weight"
            patch = patches.get(key, [])
            
            lora_diffs = lora_strengths = None
            if len(patch) != 0:
                lora_diffs = [p[1].weights for p in patch]
                lora_strengths = [p[0] for p in patch]
            
            #print("lora_diff", lora_diff)

            #print(state_dict[module_prefix + "weight"].shape)
            in_features = state_dict[module_prefix + "weight"].shape[1]
            out_features = state_dict[module_prefix + "weight"].shape[0]

            ctx = init_empty_weights if is_accelerate_available() else nullcontext
            with ctx():
                model._modules[name] = GGUFLinear(
                    in_features,
                    out_features,
                    module.bias is not None,
                    compute_dtype=compute_dtype,
                    lora_diffs=lora_diffs,
                    lora_strengths = lora_strengths
                )
            model._modules[name].source_cls = type(module)
            # Force requires_grad to False to avoid unexpected errors
            model._modules[name].requires_grad_(False)

    return model

class GGUFLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        compute_dtype=None,
        device=None,
        lora_diffs=None,
        lora_strengths=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device)
        self.compute_dtype = compute_dtype
        self.lora_diffs = lora_diffs
        self.lora_strengths = lora_strengths

    def forward(self, inputs):
        weight = dequantize_without_compile(self.weight)
        weight = weight.to(self.compute_dtype)
        bias = self.bias.to(self.compute_dtype) if self.bias is not None else None

        if self.lora_diffs is not None:
            # Apply all LoRA patches
            for lora_diff, lora_strength in zip(self.lora_diffs, self.lora_strengths):
                # Calculate the diff for this patch
                patch_diff = torch.mm(
                    lora_diff[0].flatten(start_dim=1).to(weight.device), 
                    lora_diff[1].flatten(start_dim=1).to(weight.device)
                ).reshape(weight.shape)

                if lora_diff[2] is not None:
                    alpha = lora_diff[2] / lora_diff[1].shape[0]
                else:
                    alpha = 1.0
                
                # Apply the patch with its strength
                scale = lora_strength * alpha
                weight.add_(patch_diff, alpha=scale).to(self.compute_dtype)

        output = torch.nn.functional.linear(inputs, weight, bias)
        return output