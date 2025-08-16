from ..utils import log
import torch

def set_transformer_cache_method(transformer, timesteps, cache_args=None):      
    transformer.cache_device = cache_args["cache_device"]
    if cache_args["cache_type"] == "TeaCache":
        log.info(f"TeaCache: Using cache device: {transformer.cache_device}")
        transformer.teacache_state.clear_all()
        transformer.enable_teacache = True
        transformer.rel_l1_thresh = cache_args["rel_l1_thresh"]
        transformer.teacache_start_step = cache_args["start_step"]
        transformer.teacache_end_step = len(timesteps)-1 if cache_args["end_step"] == -1 else cache_args["end_step"]
        transformer.teacache_use_coefficients = cache_args["use_coefficients"]
        transformer.teacache_mode = cache_args["mode"]
    elif cache_args["cache_type"] == "MagCache":
        log.info(f"MagCache: Using cache device: {transformer.cache_device}")
        transformer.magcache_state.clear_all()
        transformer.enable_magcache = True
        transformer.magcache_start_step = cache_args["start_step"]
        transformer.magcache_end_step = len(timesteps)-1 if cache_args["end_step"] == -1 else cache_args["end_step"]
        transformer.magcache_thresh = cache_args["magcache_thresh"]
        transformer.magcache_K = cache_args["magcache_K"]
    elif cache_args["cache_type"] == "EasyCache":
        log.info(f"EasyCache: Using cache device: {transformer.cache_device}")
        transformer.easycache_state.clear_all()
        transformer.enable_easycache = True
        transformer.easycache_start_step = cache_args["start_step"]
        transformer.easycache_end_step = len(timesteps)-1 if cache_args["end_step"] == -1 else cache_args["end_step"]
        transformer.easycache_thresh = cache_args["easycache_thresh"]
    return transformer

class TeaCacheState:
    def __init__(self, cache_device='cpu'):
        self.cache_device = cache_device
        self.states = {}
        self._next_pred_id = 0
    
    def new_prediction(self, cache_device='cpu'):
        """Create new prediction state and return its ID"""
        self.cache_device = cache_device
        pred_id = self._next_pred_id
        self._next_pred_id += 1
        self.states[pred_id] = {
            'previous_residual': None,
            'accumulated_rel_l1_distance': 0,
            'previous_modulated_input': None,
            'skipped_steps': [],
        }
        return pred_id
    
    def update(self, pred_id, **kwargs):
        """Update state for specific prediction"""
        if pred_id not in self.states:
            return None
        for key, value in kwargs.items():
            self.states[pred_id][key] = value
    
    def get(self, pred_id):
        return self.states.get(pred_id, {})
    
    def clear_all(self):
        self.states = {}
        self._next_pred_id = 0

class MagCacheState:
    def __init__(self, cache_device='cpu'):
        self.cache_device = cache_device
        self.states = {}
        self._next_pred_id = 0
    
    def new_prediction(self, cache_device='cpu'):
        """Create new prediction state and return its ID"""
        self.cache_device = cache_device
        pred_id = self._next_pred_id
        self._next_pred_id += 1
        self.states[pred_id] = {
            'residual_cache': None,
            'accumulated_ratio': 1.0,
            'accumulated_steps': 0,
            'accumulated_err': 0,
            'skipped_steps': [],
        }
        return pred_id
    
    def update(self, pred_id, **kwargs):
        """Update state for specific prediction"""
        if pred_id not in self.states:
            return None
        for key, value in kwargs.items():
            self.states[pred_id][key] = value
    
    def get(self, pred_id):
        return self.states.get(pred_id, {})
    
    def clear_all(self):
        self.states = {}
        self._next_pred_id = 0

class EasyCacheState:
    def __init__(self, cache_device='cpu'):
        self.cache_device = cache_device
        self.states = {}
        self._next_pred_id = 0

    def new_prediction(self, cache_device='cpu'):
        """Create a new prediction state and return its ID."""
        self.cache_device = cache_device
        pred_id = self._next_pred_id
        self._next_pred_id += 1
        self.states[pred_id] = {
            'previous_raw_input': None,
            'previous_raw_output': None,
            'cache': None,
            'accumulated_error': 0.0,
            'skipped_steps': [],
        }
        return pred_id

    def update(self, pred_id, **kwargs):
        """Update state for a specific prediction."""
        if pred_id not in self.states:
            return None
        for key, value in kwargs.items():
            self.states[pred_id][key] = value

    def get(self, pred_id):
        return self.states.get(pred_id, {})

    def clear_all(self):
        self.states = {}
        self._next_pred_id = 0

def relative_l1_distance(last_tensor, current_tensor):
    l1_distance = torch.abs(last_tensor.to(current_tensor.device) - current_tensor).mean()
    norm = torch.abs(last_tensor).mean()
    relative_l1_distance = l1_distance / norm
    return relative_l1_distance.to(torch.float32).to(current_tensor.device)

def cache_report(transformer, cache_args):
    cache_type = cache_args["cache_type"]
    states = (
        transformer.teacache_state.states if cache_type == "TeaCache" else
        transformer.magcache_state.states if cache_type == "MagCache" else
        transformer.easycache_state.states if cache_type == "EasyCache" else
        None
    )
    state_names = {
        0: "conditional",
        1: "unconditional"
    }
    for pred_id, state in states.items():
        name = state_names.get(pred_id, f"prediction_{pred_id}")
        if 'skipped_steps' in state:
            log.info(f"{cache_type} skipped: {len(state['skipped_steps'])} {name} steps: {state['skipped_steps']}")
    transformer.teacache_state.clear_all()
    transformer.magcache_state.clear_all()
    transformer.easycache_state.clear_all()
    del states