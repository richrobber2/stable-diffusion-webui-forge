# Copyright Forge 2024

import time
import torch
import contextlib

from backend import stream, memory_management, utils
from backend.patcher.lora import merge_lora_to_weight

stash = {}


def get_weight_and_bias(layer, weight_args=None, bias_args=None, weight_fn=None, bias_fn=None):
    """
    Retrieve and optionally cast the weight and bias parameters from a given layer.
    Also applies any LoRA patches if present.

    Args:
        layer (torch.nn.Module): The layer to extract weight and bias from.
        weight_args (dict, optional): Arguments for casting the weight tensor (e.g. device, dtype).
        bias_args (dict, optional): Arguments for casting the bias tensor (e.g. device, dtype).
        weight_fn (callable, optional): Optional function to apply to weight before casting.
        bias_fn (callable, optional): Optional function to apply to bias before casting.

    Returns:
        (torch.Tensor, torch.Tensor): The processed weight and bias tensors.
    """
    patches = getattr(layer, 'forge_online_loras', None)
    weight_patches, bias_patches = None, None
    if patches is not None:
        weight_patches = patches.get('weight', None)
        bias_patches = patches.get('bias', None)

    weight = layer.weight if layer.weight is not None else None
    if weight is not None:
        # Optional pre-processing function on weight
        if weight_fn is not None:
            weight = weight_fn(weight)
        # Casting weight
        if weight_args is not None:
            weight = weight.to(**weight_args)
        # Merge LoRA weight patches if any
        if weight_patches is not None:
            weight = merge_lora_to_weight(weight, weight_patches)

    bias = layer.bias if layer.bias is not None else None
    if bias is not None:
        # Optional pre-processing function on bias
        if bias_fn is not None:
            bias = bias_fn(bias)
        # Casting bias
        if bias_args is not None:
            bias = bias.to(**bias_args)
        # Merge LoRA bias patches if any
        if bias_patches is not None:
            bias = merge_lora_to_weight(bias, bias_patches)

    return weight, bias


def weights_manual_cast(layer, x, skip_weight_dtype=False, skip_bias_dtype=False, weight_fn=None, bias_fn=None):
    """
    Manually cast weights and bias parameters of a layer to match the input tensor.
    This ensures that operations on the layer parameters occur in an expected device/dtype environment.

    Args:
        layer (torch.nn.Module): The layer whose parameters we want to cast.
        x (torch.Tensor): The input tensor that dictates the target device/dtype.
        skip_weight_dtype (bool): If True, do not force weight to match x's dtype.
        skip_bias_dtype (bool): If True, do not force bias to match x's dtype.
        weight_fn (callable, optional): Optional function to apply to weight before casting.
        bias_fn (callable, optional): Optional function to apply to bias before casting.

    Returns:
        (torch.Tensor, torch.Tensor, torch.cuda.Event or None): Processed weight, bias, and optional CUDA event for synchronization.
    """
    weight, bias, signal = None, None, None
    non_blocking = getattr(x.device, 'type', None) != 'mps'
    target_dtype = x.dtype
    target_device = x.device

    if skip_weight_dtype:
        weight_args = dict(device=target_device, non_blocking=non_blocking)
    else:
        weight_args = dict(device=target_device, dtype=target_dtype, non_blocking=non_blocking)

    if skip_bias_dtype:
        bias_args = dict(device=target_device, non_blocking=non_blocking)
    else:
        bias_args = dict(device=target_device, dtype=target_dtype, non_blocking=non_blocking)

    # Attempt asynchronous casting if streaming is enabled
    if stream.should_use_stream():
        with stream.stream_context()(stream.mover_stream):
            weight, bias = get_weight_and_bias(layer, weight_args, bias_args, weight_fn=weight_fn, bias_fn=bias_fn)
            signal = torch.cuda.Event(enable_timing=True) if target_device.type == 'cuda' else None
    else:
        weight, bias = get_weight_and_bias(layer, weight_args, bias_args, weight_fn=weight_fn, bias_fn=bias_fn)
        signal = None

    return weight, bias, signal


@contextlib.contextmanager
def main_stream_worker(weight, bias, signal):
    """
    A context manager to coordinate operations on the main CUDA stream after asynchronous weight casting.
    It waits on a recorded event (signal) indicating that weights are ready, then executes operations,
    and finally cleans up any cached references after completion.

    Args:
        weight (torch.Tensor or None): The processed weight tensor.
        bias (torch.Tensor or None): The processed bias tensor.
        signal (torch.cuda.Event or None): CUDA event to wait on before continuing execution.
    """
    if signal is None or not stream.should_use_stream():
        # No special streaming context needed
        yield
        return

    with stream.stream_context()(stream.current_stream):
        # Wait until the event recorded after casting is complete
        stream.current_stream.wait_event(signal)
        yield
        finished_signal = stream.current_stream.record_event()
        # Store references to weight, bias, and completion event
        stash[id(finished_signal)] = (weight, bias, finished_signal)

    # Clean up any stale references from stash
    garbage = [k for k, (w, b, s) in stash.items() if s.query()]
    for k in garbage:
        del stash[k]
    return


def cleanup_cache():
    if stream.should_use_stream():
        stream.current_stream.synchronize()
        stream.mover_stream.synchronize()
    stash.clear()
    return


current_device = None
current_dtype = None
current_manual_cast_enabled = False
current_bnb_dtype = None


def _manual_cast_forward(module, x, forward_fn, *args, weight_fn=None, bias_fn=None, 
                         skip_weight_dtype=False, skip_bias_dtype=False, **kwargs):
    if module.parameters_manual_cast:
        weight, bias, signal = weights_manual_cast(module, x, skip_weight_dtype=skip_weight_dtype, 
                                                   skip_bias_dtype=skip_bias_dtype, weight_fn=weight_fn, bias_fn=bias_fn)
        with main_stream_worker(weight, bias, signal):
            return forward_fn(x, weight, bias, *args, **kwargs)
    else:
        weight, bias = get_weight_and_bias(module, weight_fn=weight_fn, bias_fn=bias_fn)
        return forward_fn(x, weight, bias, *args, **kwargs)

def _conv_base_forward(self, x, conv_func):
    return _manual_cast_forward(self, x, lambda inp, w, b: conv_func(inp, w, b))

def _conv_transpose_forward(self, x, conv_func, output_size=None, num_spatial_dims=2):
    output_padding = self._output_padding(
        x, output_size, self.stride, self.padding, 
        self.kernel_size, num_spatial_dims, self.dilation
    )
    return _manual_cast_forward(self, x, lambda inp, w, b: conv_func(inp, w, b, self.stride, self.padding, 
                                                                     output_padding, self.groups, self.dilation))

class BaseForgeConv:
    """Mixin class for common conv layer functionality"""
    def __init__(self, *args, **kwargs):
        kwargs['device'] = current_device
        kwargs['dtype'] = current_dtype
        super().__init__(*args, **kwargs)
        self.parameters_manual_cast = current_manual_cast_enabled
    
    def reset_parameters(self):
        return None

class ForgeOperations:
    """
    Default Forge Operations with optimized implementations
    """
    class Linear(torch.nn.Module):
        def __init__(self, in_features, out_features, *args, **kwargs):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            # Use empty_like for better memory allocation
            self.dummy = torch.nn.Parameter(torch.empty(1, device=current_device, dtype=current_dtype))
            self.weight = None
            self.bias = None
            self.parameters_manual_cast = current_manual_cast_enabled

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            # Load parameters from state_dict into a dummy parameter first for device/dtype alignment
            if hasattr(self, 'dummy'):
                if f'{prefix}weight' in state_dict:
                    self.weight = torch.nn.Parameter(state_dict[f'{prefix}weight'].to(self.dummy))
                if f'{prefix}bias' in state_dict:
                    self.bias = torch.nn.Parameter(state_dict[f'{prefix}bias'].to(self.dummy))
                del self.dummy
            else:
                super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        def forward(self, x):
            return _manual_cast_forward(self, x, torch.nn.functional.linear)

    class Conv2d(BaseForgeConv, torch.nn.Conv2d):
        def forward(self, x):
            return _conv_base_forward(self, x, self._conv_forward)

    class Conv3d(BaseForgeConv, torch.nn.Conv3d):
        def forward(self, x):
            return _conv_base_forward(self, x, self._conv_forward)

    class Conv1d(BaseForgeConv, torch.nn.Conv1d):
        def forward(self, x):
            return _conv_base_forward(self, x, self._conv_forward)

    class ConvTranspose2d(BaseForgeConv, torch.nn.ConvTranspose2d):
        def forward(self, x, output_size=None):
            return _conv_transpose_forward(self, x, torch.nn.functional.conv_transpose2d, output_size, 2)

    class ConvTranspose1d(BaseForgeConv, torch.nn.ConvTranspose1d):
        def forward(self, x, output_size=None):
            return _conv_transpose_forward(self, x, torch.nn.functional.conv_transpose1d, output_size, 1)

    class ConvTranspose3d(BaseForgeConv, torch.nn.ConvTranspose3d):
        def forward(self, x, output_size=None):
            return _conv_transpose_forward(self, x, torch.nn.functional.conv_transpose3d, output_size, 3)

    class GroupNorm(torch.nn.GroupNorm):
        def __init__(self, *args, **kwargs):
            kwargs['device'] = current_device
            kwargs['dtype'] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x):
            return _manual_cast_forward(self, x, 
                lambda inp, w, b: torch.nn.functional.group_norm(inp, self.num_groups, w, b, self.eps))

    class LayerNorm(torch.nn.LayerNorm):
        def __init__(self, *args, **kwargs):
            kwargs['device'] = current_device
            kwargs['dtype'] = current_dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x):
            return _manual_cast_forward(self, x, 
                lambda inp, w, b: torch.nn.functional.layer_norm(inp, self.normalized_shape, w, b, self.eps))

    class Embedding(torch.nn.Embedding):
        def __init__(self, *args, **kwargs):
            kwargs['device'] = current_device
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = current_manual_cast_enabled
            self.bias = None

        def reset_parameters(self):
            # Setting bias to None explicitly, if needed.
            self.bias = None
            return None

        def forward(self, x):
            return _manual_cast_forward(
                self, x, 
                lambda inp, w, b: torch.nn.functional.embedding(inp, w, self.padding_idx, 
                                                                self.max_norm, self.norm_type, 
                                                                self.scale_grad_by_freq, self.sparse),
                skip_weight_dtype=True, skip_bias_dtype=True
            )


# Attempting to import BNB functionalities if available
try:
    from backend.operations_bnb import ForgeLoader4Bit, ForgeParams4bit, functional_linear_4bits, functional_dequantize_4bit

    class ForgeOperationsBNB4bits(ForgeOperations):
        class Linear(ForgeLoader4Bit):
            def __init__(self, *args, **kwargs):
                super().__init__(device=current_device, dtype=current_dtype, quant_type=current_bnb_dtype)
                self.parameters_manual_cast = current_manual_cast_enabled

            def forward(self, x):
                if self.bias is not None and self.bias.dtype != x.dtype:
                    # Ensure bias is cast to the same dtype as x to avoid slow operations
                    self.bias = utils.tensor2parameter(self.bias.to(x.dtype))

                # If LoRA patches exist, handle them separately
                if hasattr(self, 'forge_online_loras'):
                    weight, bias, signal = weights_manual_cast(self, x, weight_fn=functional_dequantize_4bit, 
                                                               bias_fn=None, skip_bias_dtype=True)
                    with main_stream_worker(weight, bias, signal):
                        return torch.nn.functional.linear(x, weight, bias)

                if not self.parameters_manual_cast:
                    # If manual casting not required, directly use BNB quantized functions
                    return functional_linear_4bits(x, self.weight, self.bias)
                elif not self.weight.bnb_quantized:
                    # If weight is not yet BNB quantized, do so and revert after operation
                    return self._quantize_and_forward(x)
                else:
                    # If weight is already BNB quantized, just cast if necessary
                    weight, bias, signal = weights_manual_cast(self, x, skip_weight_dtype=True, skip_bias_dtype=True)
                    with main_stream_worker(weight, bias, signal):
                        return functional_linear_4bits(x, weight, bias)

            # Renamed from _extracted_from_forward_15
            def _quantize_and_forward(self, x):
                assert x.device.type == 'cuda', 'BNB must use CUDA as computation device!'
                layer_original_device = self.weight.device
                self.weight = self.weight._quantize(x.device)
                bias = self.bias.to(x.device) if self.bias is not None else None
                out = functional_linear_4bits(x, self.weight, bias)
                self.weight = self.weight.to(layer_original_device)
                return out

    bnb_avaliable = True
except ImportError:
    bnb_avaliable = False


from backend.operations_gguf import dequantize_tensor

class ForgeOperationsGGUF(ForgeOperations):
    class Linear(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.empty(1, device=current_device, dtype=current_dtype))
            self.weight = None
            self.bias = None
            self.parameters_manual_cast = current_manual_cast_enabled

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            # Load potentially GGUF quantized weights and bias and store them
            if hasattr(self, 'dummy'):
                computation_dtype = self.dummy.dtype
                if computation_dtype not in [torch.float16, torch.bfloat16]:
                    computation_dtype = torch.float16
                if f'{prefix}weight' in state_dict:
                    self.weight = state_dict[f'{prefix}weight'].to(device=self.dummy.device)
                    self.weight.computation_dtype = computation_dtype
                if f'{prefix}bias' in state_dict:
                    self.bias = state_dict[f'{prefix}bias'].to(device=self.dummy.device)
                    self.bias.computation_dtype = computation_dtype
                del self.dummy
            else:
                if f'{prefix}weight' in state_dict:
                    self.weight = state_dict[f'{prefix}weight']
                if f'{prefix}bias' in state_dict:
                    self.bias = state_dict[f'{prefix}bias']

        def _apply(self, fn, recurse=True):
            for k, p in self.named_parameters(recurse=False, remove_duplicate=True):
                setattr(self, k, utils.tensor2parameter(fn(p)))
            return self

        def forward(self, x):
            if self.bias is not None and self.bias.dtype != x.dtype:
                self.bias = utils.tensor2parameter(dequantize_tensor(self.bias).to(x.dtype))

            if self.weight is not None and self.weight.dtype != x.dtype and getattr(self.weight, 'gguf_cls', None) is None:
                self.weight = utils.tensor2parameter(self.weight.to(x.dtype))

            weight, bias, signal = weights_manual_cast(self, x, weight_fn=dequantize_tensor, bias_fn=None, skip_bias_dtype=True)
            with main_stream_worker(weight, bias, signal):
                return torch.nn.functional.linear(x, weight, bias)


@contextlib.contextmanager
def using_forge_operations(operations=None, device=None, dtype=None, manual_cast_enabled=False, bnb_dtype=None):
    """Optimized context manager implementation"""
    global current_device, current_dtype, current_manual_cast_enabled, current_bnb_dtype

    # Cache previous values for restoration
    prev_values = (current_device, current_dtype, current_manual_cast_enabled, current_bnb_dtype)
    current_device, current_dtype, current_manual_cast_enabled, current_bnb_dtype = device, dtype, manual_cast_enabled, bnb_dtype

    if operations is None:
        operations = (ForgeOperationsGGUF if bnb_dtype == 'gguf' else
                     ForgeOperationsBNB4bits if bnb_avaliable and bnb_dtype in ['nf4', 'fp4'] else
                     ForgeOperations)

    # Use tuple for faster lookup
    op_names = ('Linear', 'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 
                'ConvTranspose2d', 'ConvTranspose3d', 'GroupNorm', 'LayerNorm', 'Embedding')
    
    # Store original classes
    backups = {name: getattr(torch.nn, name) for name in op_names}

    try:
        # Set new classes
        for name in op_names:
            setattr(torch.nn, name, getattr(operations, name))
        yield
    finally:
        # Restore original classes and global values
        for name, cls in backups.items():
            setattr(torch.nn, name, cls)
        current_device, current_dtype, current_manual_cast_enabled, current_bnb_dtype = prev_values


def shift_manual_cast(model, enabled):
    """
    Shift the manual casting behavior of all modules in a model.

    Args:
        model (torch.nn.Module): The model whose modules should have `parameters_manual_cast` toggled.
        enabled (bool): If True, enable manual casting; if False, disable it.
    """
    for m in model.modules():
        if hasattr(m, 'parameters_manual_cast'):
            m.parameters_manual_cast = enabled
    return


@contextlib.contextmanager
def automatic_memory_management():
    """
    A context manager to automatically free up memory before model initialization or loading.
    It patches `torch.nn.Module.__init__` and `torch.nn.Module.to` to track modules and later moves them to CPU.
    Finally, it empties the cache.
    """
    memory_management.free_memory(
        memory_required=3 * 1024 * 1024 * 1024,
        device=memory_management.get_torch_device()
    )

    module_list = []

    original_init = torch.nn.Module.__init__
    original_to = torch.nn.Module.to

    def patched_init(self, *args, **kwargs):
        module_list.append(self)
        return original_init(self, *args, **kwargs)

    def patched_to(self, *args, **kwargs):
        module_list.append(self)
        return original_to(self, *args, **kwargs)

    try:
        torch.nn.Module.__init__ = patched_init
        torch.nn.Module.to = patched_to
        yield
    finally:
        torch.nn.Module.__init__ = original_init
        torch.nn.Module.to = original_to

    # After exiting the context, move all tracked modules to CPU and clear cache
    start = time.perf_counter()
    module_list = set(module_list)

    for module in module_list:
        module.cpu()

    memory_management.soft_empty_cache()
    end = time.perf_counter()

    print(f'Automatic Memory Management: {len(module_list)} Modules in {(end - start):.2f} seconds.')
    return


class DynamicSwapInstaller:
    """
    Utility class for dynamically changing the behavior of modules at runtime by modifying their __class__ attributes.
    """

    @staticmethod
    def _install_module(module: torch.nn.Module, target_device: torch.device):
        original_class = module.__class__
        module.__dict__['forge_backup_original_class'] = original_class

        def hacked_get_attr(self, name: str):
            # Dynamically fetch parameters/buffers and move them to the target device
            if '_parameters' in self.__dict__:
                _parameters = self.__dict__['_parameters']
                if name in _parameters:
                    p = _parameters[name]
                    if p is None:
                        return None
                    if p.__class__ == torch.nn.Parameter:
                        return torch.nn.Parameter(p.to(target_device), requires_grad=p.requires_grad)
                    else:
                        return p.to(target_device)
            if '_buffers' in self.__dict__:
                _buffers = self.__dict__['_buffers']
                if name in _buffers:
                    return _buffers[name].to(target_device)
            return super(original_class, self).__getattr__(name)

        module.__class__ = type(
            f'DynamicSwap_{original_class.__name__}',
            (original_class,),
            {
                '__getattr__': hacked_get_attr,
            },
        )
        return

    @staticmethod
    def _uninstall_module(module: torch.nn.Module):
        if 'forge_backup_original_class' in module.__dict__:
            module.__class__ = module.__dict__.pop('forge_backup_original_class')
        return

    @staticmethod
    def install_model(model: torch.nn.Module, target_device: torch.device):
        """
        Install dynamic swapping behavior into all modules of a model, enabling on-the-fly device switching.
        """
        for m in model.modules():
            DynamicSwapInstaller._install_module(m, target_device)
        return

    @staticmethod
    def uninstall_model(model: torch.nn.Module):
        """
        Uninstall previously added dynamic swapping behavior from all modules of a model.
        """
        for m in model.modules():
            DynamicSwapInstaller._uninstall_module(m)
        return
