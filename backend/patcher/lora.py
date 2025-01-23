import torch

import packages_3rdparty.webui_lora_collection.lora as lora_utils_webui
import packages_3rdparty.comfyui_lora_collection.lora as lora_utils_comfyui

from backend import memory_management, utils
from backend import operations

# Priority order for LoRA collections
lora_collection_priority = [lora_utils_webui, lora_utils_comfyui]

extra_weight_calculators = {}

_function_cache = {}

def get_function(function_name: str):
    """
    Retrieve a function by name from the prioritized LoRA collections.
    The first collection that has the function will be used.
    """
    if function_name in _function_cache:
        return _function_cache[function_name]
    for lora_collection in lora_collection_priority:
        if hasattr(lora_collection, function_name):
            func = getattr(lora_collection, function_name)
            _function_cache[function_name] = func
            return func
    return None


def load_lora(lora, to_load):
    """
    Load LoRA data using the currently active LoRA collections.
    """
    func = get_function('load_lora')
    patch_dict, remaining_dict = func(lora, to_load)
    return patch_dict, remaining_dict


def inner_str(k, prefix="", suffix=""):
    """
    Extract a substring from `k` by removing a specified prefix and suffix.
    """
    start = len(prefix)
    end = len(k) - len(suffix) if suffix else len(k)
    return k[start:end]


def _resolve_model_keys(model, key_map, model_type):
    """
    Generic function to resolve model keys for different model types.
    """
    if key_map is None:
        key_map = {}
    
    func_name = f'model_lora_keys_{model_type}'
    resolver = get_function(func_name)
    if not resolver:
        raise LoraError(f"No key resolver found for model type: {model_type}")
    
    model_keys, key_maps = resolver(model, key_map)
    
    if model_type == 'clip':
        # Special handling for T5 XXL transformer layers
        for model_key in model_keys:
            if (model_key.endswith(".weight") and 
                model_key.startswith("t5xxl.transformer.")):
                for prefix in ('te1', 'te2', 'te3'):
                    formatted = inner_str(model_key, "t5xxl.transformer.", ".weight").replace(".", "_")
                    formatted = f"lora_{prefix}_{formatted}"
                    key_map[formatted] = model_key
    
    return key_maps

def model_lora_keys_clip(model, key_map=None):
    """Resolve model keys for CLIP (text encoder) LoRA layers."""
    return _resolve_model_keys(model, key_map, 'clip')

def model_lora_keys_unet(model, key_map=None):
    """Resolve model keys for UNet LoRA layers."""
    return _resolve_model_keys(model, key_map, 'unet')


@torch.inference_mode()
def weight_decompose(dora_scale, weight, lora_diff, alpha, strength, computation_dtype):
    """
    Efficiently apply LoRA decomposition and scaling in inference mode.
    """
    dora_scale = memory_management.cast_to_device(dora_scale, weight.device, computation_dtype)

    lora_diff.mul_(alpha)
    if lora_diff.dtype != weight.dtype:
        lora_diff = lora_diff.to(weight.dtype)

    weight.add_(lora_diff)

    weight_flat = weight.view(weight.size(0), -1)
    weight_norm = weight_flat.norm(dim=1, keepdim=True).reshape(weight.shape[0], 1, *([1]*(weight.ndim-2)))

    scale_factor = dora_scale.div_(weight_norm).to(weight.dtype)
    weight.mul_(scale_factor)

    if strength != 1.0:
        weight.mul_(strength)

    return weight


def _helper_apply_diff_patch(weight, w1, strength):
    """Apply a 'diff' type patch to a weight. Always returns a new tensor."""
    if strength == 0.0:
        return weight.clone()
        
    if w1.shape == weight.shape:
        result = weight.clone()
        result.add_(strength * w1.to(weight.dtype, copy=False))
    else:
        new_shape = [max(n, m) for n, m in zip(weight.shape, w1.shape)]
        result = torch.zeros(new_shape, dtype=weight.dtype, device=weight.device)
        w1 = w1.to(device=weight.device, dtype=weight.dtype, copy=False)
        slices = tuple(slice(0, s) for s in w1.shape)
        result.copy_(weight.expand(new_shape))
        result[slices].add_(strength * w1)
    
    return result


def _helper_apply_lora_patch(weight, v, strength, computation_dtype, key):
    """
    Helper: Apply a 'lora' type patch efficiently.
    """
    try:
        mat1 = memory_management.cast_to_device(v[0], weight.device, computation_dtype)
        mat2 = memory_management.cast_to_device(v[1], weight.device, computation_dtype)
        dora_scale = v[4]
        alpha = v[2] / mat2.size(0) if v[2] is not None else 1.0

        if v[3] is not None:
            mat3 = memory_management.cast_to_device(v[3], weight.device, computation_dtype)
            mat2_t = mat2.transpose(0, 1).flatten(1)
            mat3_t = mat3.transpose(0, 1).flatten(1)
            mat2 = torch.mm(mat2_t, mat3_t).reshape(
                mat2.size(1), mat2.size(0), mat3.size(2), mat3.size(3)
            ).transpose(0, 1)

        lora_diff = torch.mm(mat1.flatten(1), mat2.flatten(1)).reshape(weight.shape)
        if dora_scale is not None:
            weight = weight_decompose(dora_scale, weight, lora_diff, alpha, strength, computation_dtype)
        else:
            weight.add_((strength * alpha * lora_diff).to(weight.dtype, copy=False))
    except Exception as e:
        raise LoraError(f"Failed to apply LoRA patch for key {key}: {str(e)}") from e

    return weight


def _apply_single_patch(weight, patch_data, computation_dtype):
    """Handle application of a single patch."""
    strength, v, strength_model, offset, function = patch_data
    function = function or (lambda a: a)

    if offset is not None:
        weight = weight.narrow(offset[0], offset[1], offset[2])

    if strength_model != 1.0:
        weight = weight * strength_model

    if isinstance(v, list):
        base_weight = v[0].to(device=weight.device, dtype=weight.dtype, copy=False)
        weight = merge_lora_to_weight(v[1:], base_weight, "nested_lora", computation_dtype)
        v = (weight,)

    patch_type = 'diff' if len(v) == 1 else v[0]
    v = v[1] if len(v) == 2 else v

    try:
        if patch_type == "diff":
            weight = _helper_apply_diff_patch(weight, v[0], strength)
        elif patch_type == "lora":
            weight = _helper_apply_lora_patch(weight, v, strength, computation_dtype, "lora_patch")
            weight = function(weight)
    except Exception as e:
        raise LoraError(f"Failed to apply {patch_type} patch: {str(e)}") from e

    return weight


@torch.inference_mode()
def merge_lora_to_weight(patches, weight, key="online_lora", computation_dtype=torch.float32):
    """Merge LoRA patches into weight tensor."""
    weight_dtype_backup = weight.dtype if computation_dtype != weight.dtype else None
    
    if weight_dtype_backup:
        weight = weight.to(dtype=computation_dtype, copy=True)
    else:
        weight = weight.clone()

    try:
        for patch in patches:
            weight = _apply_single_patch(weight, patch, computation_dtype)
    except Exception as e:
        raise LoraError(f"Failed to merge LoRA patches for key {key}") from e

    if weight_dtype_backup:
        weight = weight.to(dtype=weight_dtype_backup, copy=False)

    return weight


def get_parameter_devices(model):
    """
    Get a mapping of model parameter names to their devices.
    """
    return {key: p.device for key, p in model.named_parameters()}


def set_parameter_devices(model, parameter_devices):
    """
    Move model parameters to specified devices in inference mode.
    """
    for key, device in parameter_devices.items():
        p = utils.get_attr(model, key)
        if p.device != device:
            p = utils.tensor2parameter(p.to(device=device, copy=False))
            p.requires_grad = False
            utils.set_attr_raw(model, key, p)
    return model


def calculate_patch_hash(patches):
    """Calculate a more robust hash for patch comparison."""
    import hashlib
    import json
    
    def serialize_tensor(t):
        if isinstance(t, torch.Tensor):
            return {'shape': list(t.shape), 'device': str(t.device), 'dtype': str(t.dtype)}
        return str(t)
    
    patch_data = {str(k): [serialize_tensor(p) for p in v] 
                 for k, v in patches.items()}
    return hashlib.sha256(json.dumps(patch_data, sort_keys=True).encode()).hexdigest()

class LoraError(Exception):
    """Custom exception for LoRA-related errors."""
    pass

class LoraLoader:
    """
    A LoRA loader class to apply LoRA patches to a model in an inference-only scenario.
    """

    def __init__(self, model):
        self.model = model
        self.backup = {}
        self.online_backup = []
        self.loaded_hash = None
        self._weight_backup = {}
        
        # Enable compression by default with conservative settings
        self.compression_enabled = True
        self.compression_config = {
            'min_weight_size': 2048,  # More conservative threshold
            'rank': None,  # Auto-determine rank
            'energy_threshold': 0.95,  # Higher energy retention (95%)
            'block_size': 1024  # Default block size for large matrices
        }

    def disable_compression(self):
        """Disable SVD compression with a warning."""
        print("Warning: Disabling weight compression may increase memory usage.")
        self.compression_enabled = False

    def enable_compression(self, **kwargs):
        """Enable SVD compression with optional configuration."""
        self.compression_enabled = True
        self.compression_config.update(kwargs)

    def _backup_weight(self, key, weight, device):
        """Safely backup a weight tensor."""
        if key not in self._weight_backup:
            self._weight_backup[key] = weight.to(device=device, copy=True)

    def _restore_weights(self):
        """Safely restore original weights."""
        for k, w in self._weight_backup.items():
            if not isinstance(w, torch.nn.Parameter):
                w = torch.nn.Parameter(w, requires_grad=False)
            try:
                utils.set_attr_raw(self.model, k, w)
            except Exception as e:
                raise LoraError(f"Failed to restore weight {k}: {str(e)}") from e

    @torch.inference_mode()
    def refresh(self, lora_patches, offload_device=None):
        """
        Refresh the model with the given LoRA patches in inference mode.
        """
        if offload_device is None:
            offload_device = torch.device('cpu')

        current_hash = calculate_patch_hash(lora_patches)
        if current_hash == self.loaded_hash:
            return

        try:
            hashes = str(list(lora_patches.keys()))
            if hashes == self.loaded_hash:
                return

            all_patches = {}
            for (_, _, _, online_mode), patches in lora_patches.items():
                for key, current_patches in patches.items():
                    all_patches[(key, online_mode)] = all_patches.get((key, online_mode), []) + current_patches

            memory_management.signal_empty_cache = True
            parameter_devices = get_parameter_devices(self.model)

            for m in set(self.online_backup):
                if hasattr(m, 'forge_online_loras'):
                    del m.forge_online_loras
            self.online_backup = []

            for k, w in self.backup.items():
                if not isinstance(w, torch.nn.Parameter):
                    w = torch.nn.Parameter(w, requires_grad=False)
                utils.set_attr_raw(self.model, k, w)
            self.backup = {}

            set_parameter_devices(self.model, parameter_devices=parameter_devices)

            from backend.operations_bnb import functional_dequantize_4bit
            from backend.operations_gguf import dequantize_tensor

            for (key, online_mode), current_patches in all_patches.items():
                try:
                    parent_layer, child_key, weight = utils.get_attr_with_parent(self.model, key)
                    assert isinstance(weight, torch.nn.Parameter)
                except Exception:
                    raise ValueError(f"Wrong LoRA Key: {key}") from None

                if online_mode:
                    if not hasattr(parent_layer, 'forge_online_loras'):
                        parent_layer.forge_online_loras = {}
                    parent_layer.forge_online_loras[child_key] = current_patches
                    self.online_backup.append(parent_layer)
                    continue

                if key not in self.backup:
                    self.backup[key] = weight.to(device=offload_device, copy=False)

                bnb_layer = None
                gguf_cls = getattr(weight, 'gguf_cls', None)
                gguf_parameter = None

                if hasattr(weight, 'bnb_quantized') and operations.bnb_avaliable:
                    bnb_layer = parent_layer
                    weight = functional_dequantize_4bit(weight)

                if gguf_cls is not None:
                    gguf_parameter = weight
                    weight = dequantize_tensor(weight)

                try:
                    weight = merge_lora_to_weight(current_patches, weight, key, computation_dtype=torch.float32)
                except RuntimeError:
                    print('Patching LoRA weights out of memory. Retrying by offloading models.')
                    set_parameter_devices(self.model, parameter_devices={k: offload_device for k in parameter_devices.keys()})
                    memory_management.soft_empty_cache()
                    weight = merge_lora_to_weight(current_patches, weight, key, computation_dtype=torch.float32)

                if bnb_layer is not None:
                    bnb_layer.reload_weight(weight)
                    continue

                if gguf_cls is not None:
                    gguf_cls.quantize_pytorch(weight, gguf_parameter)
                    continue

                utils.set_attr_raw(self.model, key, torch.nn.Parameter(weight, requires_grad=False))

            if self.compression_enabled:
                from backend.weight_compression import compress_model_weights
                try:
                    self.model = compress_model_weights(
                        self.model,
                        **self.compression_config
                    )
                except Exception as e:
                    print(f"Warning: Weight compression failed: {str(e)}")
        
            set_parameter_devices(self.model, parameter_devices=parameter_devices)
            self.loaded_hash = current_hash

        except Exception as e:
            # Attempt to restore original state on error
            self._restore_weights()
            raise LoraError("Failed to apply LoRA patches, model restored to original state") from e
        return
