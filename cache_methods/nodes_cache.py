from comfy import model_management as mm

class WanVideoTeaCache:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "rel_l1_thresh": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.001,
                                            "tooltip": "Higher values will make TeaCache more aggressive, faster, but may cause artifacts. Good value range for 1.3B: 0.05 - 0.08, for other models 0.15-0.30"}),
                "start_step": ("INT", {"default": 1, "min": 0, "max": 9999, "step": 1, "tooltip": "Start percentage of the steps to apply TeaCache"}),
                "end_step": ("INT", {"default": -1, "min": -1, "max": 9999, "step": 1, "tooltip": "End steps to apply TeaCache"}),
                "cache_device": (["main_device", "offload_device"], {"default": "offload_device", "tooltip": "Device to cache to"}),
                "use_coefficients": ("BOOLEAN", {"default": True, "tooltip": "Use calculated coefficients for more accuracy. When enabled therel_l1_thresh should be about 10 times higher than without"}),
            },
            "optional": {
                "mode": (["e", "e0"], {"default": "e", "tooltip": "Choice between using e (time embeds, default) or e0 (modulated time embeds)"}),
            },
        }
    RETURN_TYPES = ("CACHEARGS",)
    RETURN_NAMES = ("cache_args",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = """
Patch WanVideo model to use TeaCache. Speeds up inference by caching the output and  
applying it instead of doing the step.  Best results are achieved by choosing the  
appropriate coefficients for the model. Early steps should never be skipped, with too  
aggressive values this can happen and the motion suffers. Starting later can help with that too.   
When NOT using coefficients, the threshold value should be  
about 10 times smaller than the value used with coefficients.  

Official recommended values https://github.com/ali-vilab/TeaCache/tree/main/TeaCache4Wan2.1:


<pre style='font-family:monospace'>
+-------------------+--------+---------+--------+
|       Model       |  Low   | Medium  |  High  |
+-------------------+--------+---------+--------+
| Wan2.1 t2v 1.3B  |  0.05  |  0.07   |  0.08  |
| Wan2.1 t2v 14B   |  0.14  |  0.15   |  0.20  |
| Wan2.1 i2v 480P  |  0.13  |  0.19   |  0.26  |
| Wan2.1 i2v 720P  |  0.18  |  0.20   |  0.30  |
+-------------------+--------+---------+--------+
</pre> 
"""

    def process(self, rel_l1_thresh, start_step, end_step, cache_device, use_coefficients, mode="e"):
        if cache_device == "main_device":
            cache_device = mm.get_torch_device()
        else:
            cache_device = mm.unet_offload_device()
        cache_args = {
            "cache_type": "TeaCache",
            "rel_l1_thresh": rel_l1_thresh,
            "start_step": start_step,
            "end_step": end_step,
            "cache_device": cache_device,
            "use_coefficients": use_coefficients,
            "mode": mode,
        }
        return (cache_args,)
    
class WanVideoMagCache:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "magcache_thresh": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.3, "step": 0.001, "tooltip": "How strongly to cache the output of diffusion model. This value must be non-negative."}),
                "magcache_K": ("INT", {"default": 4, "min": 0, "max": 6, "step": 1, "tooltip": "The maxium skip steps of MagCache."}),
                "start_step": ("INT", {"default": 1, "min": 0, "max": 9999, "step": 1, "tooltip": "Step to start applying MagCache"}),
                "end_step": ("INT", {"default": -1, "min": -1, "max": 9999, "step": 1, "tooltip": "Step to end applying MagCache"}),
                "cache_device": (["main_device", "offload_device"], {"default": "offload_device", "tooltip": "Device to cache to"}),
            },
        }
    RETURN_TYPES = ("CACHEARGS",)
    RETURN_NAMES = ("cache_args",)
    FUNCTION = "setargs"
    CATEGORY = "WanVideoWrapper"
    EXPERIMENTAL = True
    DESCRIPTION = "MagCache for WanVideoWrapper, source https://github.com/Zehong-Ma/MagCache"

    def setargs(self, magcache_thresh, magcache_K, start_step, end_step, cache_device):
        if cache_device == "main_device":
            cache_device = mm.get_torch_device()
        else:
            cache_device = mm.unet_offload_device()

        cache_args = {
            "cache_type": "MagCache",
            "magcache_thresh": magcache_thresh,
            "magcache_K": magcache_K,
            "start_step": start_step,
            "end_step": end_step,
            "cache_device": cache_device,
        }
        return (cache_args,)
    
class WanVideoEasyCache:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "easycache_thresh": ("FLOAT", {"default": 0.015, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "How strongly to cache the output of diffusion model. This value must be non-negative."}),
                "start_step": ("INT", {"default": 10, "min": 1, "max": 9999, "step": 1, "tooltip": "Step to start applying EasyCache"}),
                "end_step": ("INT", {"default": -1, "min": -1, "max": 9999, "step": 1, "tooltip": "Step to end applying EasyCache"}),
                "cache_device": (["main_device", "offload_device"], {"default": "offload_device", "tooltip": "Device to cache to"}),
            },
        }
    RETURN_TYPES = ("CACHEARGS",)
    RETURN_NAMES = ("cache_args",)
    FUNCTION = "setargs"
    CATEGORY = "WanVideoWrapper"
    EXPERIMENTAL = True
    DESCRIPTION = "EasyCache for WanVideoWrapper, source https://github.com/H-EmbodVis/EasyCache"

    def setargs(self, easycache_thresh, start_step, end_step, cache_device):
        if cache_device == "main_device":
            cache_device = mm.get_torch_device()
        else:
            cache_device = mm.unet_offload_device()

        cache_args = {
            "cache_type": "EasyCache",
            "easycache_thresh": easycache_thresh,
            "start_step": start_step,
            "end_step": end_step,
            "cache_device": cache_device,
        }
        return (cache_args,)

    
NODE_CLASS_MAPPINGS = {
    "WanVideoTeaCache": WanVideoTeaCache,
    "WanVideoMagCache": WanVideoMagCache,
    "WanVideoEasyCache": WanVideoEasyCache,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoTeaCache": "WaWanVideo TeaCache",
    "WanVideoMagCache": "WanVideo MagCache",
    "WanVideoEasyCache": "WanVideo EasyCache"
    }