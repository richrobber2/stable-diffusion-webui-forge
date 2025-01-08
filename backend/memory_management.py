# Cherry-picked from ComfyUI with improvements and adaptive strategies

import sys
import time
import psutil
import torch
import platform
from enum import Enum

from backend import stream, utils
from backend.args import args

cpu = torch.device('cpu')

GIGABYTE = 1024 * 1024 * 1024
benchmarking_mode = False

def check_reliable_info() -> None:
    """Check if CUDA device reports reliable memory information."""
    global benchmarking_mode
    try:
        props = torch.cuda.get_device_properties("cuda")
        if props.total_memory <= 0:
            print(f"Warning: CUDA device '{props.name}' reports invalid total memory: {props.total_memory}.")
            print("Falling back to benchmarking mode for memory management.")
            benchmarking_mode = True
            return
        
        # Additional sanity check for very low reported memory
        if props.total_memory < 1024:  # Less than 1MB reported
            print(f"Warning: CUDA device '{props.name}' reports suspiciously low memory: {props.total_memory / 1024:.2f} MB")
            print("Falling back to benchmarking mode for memory management.")
            benchmarking_mode = True
            return
            
        benchmarking_mode = False
    except Exception as e:
        print(f"Warning: Could not query CUDA device properties: {str(e)}")
        print("Falling back to benchmarking mode for memory management.")
        benchmarking_mode = True


class VRAMState(Enum):
    """Enumeration for GPU VRAM state categories."""
    DISABLED = 0
    NO_VRAM = 1
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5


class CPUState(Enum):
    """Enumeration for CPU state or computation device mode."""
    GPU = 0
    CPU = 1
    MPS = 2


# Initialize global state
vram_state = VRAMState.NORMAL_VRAM
set_vram_to = VRAMState.NORMAL_VRAM
cpu_state = CPUState.GPU
total_vram = 0
lowvram_available = True
xpu_available = False

if args.pytorch_deterministic:
    print("Using deterministic algorithms for PyTorch")
    torch.use_deterministic_algorithms(True, warn_only=True)

directml_enabled = False
if args.directml is not None:
    import torch_directml
    directml_enabled = True
    device_index = args.directml
    if device_index < 0:
        directml_device = torch_directml.device()
    else:
        directml_device = torch_directml.device(device_index)
    print(f"Using DirectML with device: {torch_directml.device_name(device_index)}")

try:
    import intel_extension_for_pytorch as ipex
    if torch.xpu.is_available():
        xpu_available = True
except ImportError:
    pass

try:
    if torch.backends.mps.is_available():
        cpu_state = CPUState.MPS
        import torch.mps
except:
    pass

if args.always_cpu:
    cpu_state = CPUState.CPU


def is_intel_xpu():
    return bool(cpu_state == CPUState.GPU and xpu_available)


def get_torch_device():
    global directml_enabled, cpu_state
    if directml_enabled:
        return directml_device
    if cpu_state == CPUState.MPS:
        return torch.device("mps")
    if cpu_state == CPUState.CPU:
        return torch.device("cpu")
    if is_intel_xpu():
        return torch.device("xpu", torch.xpu.current_device())
    return torch.device(torch.cuda.current_device())


def get_mem_stats(dev):
    global benchmarking_mode
    check_reliable_info()
    if benchmarking_mode:
        print("benchmarking mode active - exact info unavailable.")
        return {'mem_total': GIGABYTE, 'mem_reserved': 0, 'mem_active': 0}
    if dev is None:
        dev = get_torch_device()
    # Handle CPU/MPS
    if hasattr(dev, 'type') and dev.type in ['cpu', 'mps']:
        mem_total = psutil.virtual_memory().total
        return {'mem_total': mem_total, 'mem_reserved': 0, 'mem_active': 0}
    # Handle directml_enabled
    if directml_enabled:
        return {'mem_total': GIGABYTE, 'mem_reserved': 0, 'mem_active': 0}
    # Handle Intel XPU
    if is_intel_xpu():
        stats = torch.xpu.memory_stats(dev)
        return {
            'mem_total': torch.xpu.get_device_properties(dev).total_memory,
            'mem_reserved': stats['reserved_bytes.all.current'],
            'mem_active': stats['active_bytes.all.current']
        }
    # Handle CUDA
    stats = torch.cuda.memory_stats(dev)
    return {
        'mem_total': torch.cuda.mem_get_info(dev)[1],
        'mem_reserved': stats['reserved_bytes.all.current'],
        'mem_active': stats['active_bytes.all.current']
    }


def get_total_memory(dev=None, torch_total_too=False):
    """Get total available memory across devices.
    
    Args:
        dev: Target device. If None, uses first available CUDA device or CPU
        torch_total_too: Whether to return torch-managed memory too
    
    Returns:
        Total memory available (and optionally torch-managed memory)
    """
    if dev is None:
        if directml_enabled:
            dev = directml_device
        elif is_intel_xpu():
            dev = torch.device("xpu:0")
        elif torch.cuda.is_available():
            dev = torch.device("cuda:0")
        else:
            dev = torch.device("cpu")

    # Handle different device types
    if hasattr(dev, 'type'):
        if dev.type == "cuda":
            # Multi-GPU aggregation for CUDA
            total_memory = 0
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                d = torch.device(f"cuda:{i}")
                stats = get_mem_stats(d)
                mem = stats['mem_reserved'] if torch_total_too else stats['mem_total']
                total_memory += mem
            return total_memory
        elif dev.type == "xpu":
            # Single XPU device handling
            stats = get_mem_stats(dev)
            return stats['mem_reserved'] if torch_total_too else stats['mem_total']
        elif dev.type in ["cpu", "mps"]:
            # CPU/MPS memory handling
            stats = get_mem_stats(dev)
            return stats['mem_total']
    
    # Fallback for other devices (including directml)
    stats = get_mem_stats(dev)
    return stats['mem_reserved'] if torch_total_too else stats['mem_total']


total_vram = get_total_memory(get_torch_device()) / (1024 * 1024)
total_ram = psutil.virtual_memory().total / (1024 * 1024)
print(f"Total VRAM {total_vram:.0f} MB, total RAM {total_ram:.0f} MB")

try:
    print(f"PyTorch version: {torch.version.__version__}")
except:
    pass

try:
    OOM_EXCEPTION = torch.cuda.OutOfMemoryError
except:
    OOM_EXCEPTION = Exception

if directml_enabled:
    OOM_EXCEPTION = Exception

XFORMERS_VERSION = ""
XFORMERS_ENABLED_VAE = True
if args.disable_xformers:
    XFORMERS_IS_AVAILABLE = False
else:
    try:
        import xformers
        import xformers.ops
        XFORMERS_IS_AVAILABLE = True
        try:
            XFORMERS_IS_AVAILABLE = xformers._has_cpp_library
        except:
            pass
        try:
            XFORMERS_VERSION = xformers.version.__version__
            print(f"xformers version: {XFORMERS_VERSION}")
            if XFORMERS_VERSION.startswith("0.0.18"):
                print("\nWARNING: xformers 0.0.18 can produce black images for high-res. Consider another version.\n")
                XFORMERS_ENABLED_VAE = False
        except:
            pass
    except:
        XFORMERS_IS_AVAILABLE = False


def is_nvidia():
    global cpu_state
    return bool(cpu_state == CPUState.GPU and torch.version.cuda)


ENABLE_PYTORCH_ATTENTION = False
if args.attention_pytorch:
    ENABLE_PYTORCH_ATTENTION = True
    XFORMERS_IS_AVAILABLE = False

VAE_DTYPES = [torch.float32]

try:
    if is_nvidia():
        torch_version = torch.version.__version__
        if int(torch_version[0]) >= 2:
            if (not ENABLE_PYTORCH_ATTENTION
                    and args.attention_split == False
                    and args.attention_quad == False):
                ENABLE_PYTORCH_ATTENTION = True
            if (torch.cuda.is_bf16_supported() and
                    torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 8):
                VAE_DTYPES = [torch.bfloat16] + VAE_DTYPES

    if is_intel_xpu() and (args.attention_split == False and args.attention_quad == False):
        ENABLE_PYTORCH_ATTENTION = True
except:
    pass

if is_intel_xpu():
    VAE_DTYPES = [torch.bfloat16] + VAE_DTYPES

if args.vae_in_cpu:
    VAE_DTYPES = [torch.float32]

VAE_ALWAYS_TILED = False

if ENABLE_PYTORCH_ATTENTION:
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

# Set VRAM state based on arguments
if args.always_low_vram:
    set_vram_to = VRAMState.LOW_VRAM
    lowvram_available = True
elif args.always_no_vram:
    set_vram_to = VRAMState.NO_VRAM
elif args.always_high_vram or args.always_gpu:
    vram_state = VRAMState.HIGH_VRAM

FORCE_FP32 = False
FORCE_FP16 = False
if args.all_in_fp32:
    print("Forcing FP32 parameters.")
    FORCE_FP32 = True

if args.all_in_fp16:
    print("Forcing FP16 parameters.")
    FORCE_FP16 = True

if lowvram_available and set_vram_to in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
    vram_state = set_vram_to

if cpu_state != CPUState.GPU:
    vram_state = VRAMState.DISABLED

if cpu_state == CPUState.MPS:
    vram_state = VRAMState.SHARED

print(f"Set vram state to: {vram_state.name}")

ALWAYS_VRAM_OFFLOAD = args.always_offload_from_vram
if ALWAYS_VRAM_OFFLOAD:
    print("Always offloading VRAM is enabled.")

PIN_SHARED_MEMORY = args.pin_shared_memory
if PIN_SHARED_MEMORY:
    print("Pinned shared memory enabled.")


def get_torch_device_name(device):
    if hasattr(device, 'type'):
        if device.type != "cuda":
            return f"{device.type}"
        try:
            allocator_backend = torch.cuda.get_allocator_backend()
        except:
            allocator_backend = ""
        return f"{device} {torch.cuda.get_device_name(device)} : {allocator_backend}"
    elif is_intel_xpu():
        return f"{device} {torch.xpu.get_device_name(device)}"
    else:
        return f"CUDA {device}: {torch.cuda.get_device_name(device)}"


try:
    torch_device_name = get_torch_device_name(get_torch_device())
    print(f"Device: {torch_device_name}")
except:
    torch_device_name = ''
    print("Could not determine default device.")

if 'rtx' in torch_device_name.lower() and not args.cuda_malloc:
    print('Hint: Your RTX device might benefit from using --cuda-malloc.')

current_loaded_models = []


def state_dict_size(sd, exclude_device=None):
    return sum(sd[k].element_size() * sd[k].nelement() for k in sd)


def state_dict_parameters(sd):
    return sum(v.numel() for k, v in sd.items())


def state_dict_dtype(state_dict):
    for k, v in state_dict.items():
        state_dict[k] = v.to(state_dict[k].dtype)

    dtype_counts = {}
    for tensor in state_dict.values():
        dtype_counts[tensor.dtype] = dtype_counts.get(tensor.dtype, 0) + 1

    major_dtype = max(dtype_counts, key=dtype_counts.get)
    return major_dtype


def bake_gguf_model(model):
    if getattr(model, 'gguf_baked', False):
        return model

    for p in model.parameters():
        p.requires_grad = False

    global signal_empty_cache
    signal_empty_cache = True

    model.gguf_baked = True
    return model


def module_size(module, exclude_device=None, include_device=None, return_split=False):
    module_mem = 0
    weight_mem = 0
    weight_patterns = ['weight']

    for k, p in module.named_parameters():
        t = p.data

        if exclude_device is not None and t.device == exclude_device:
            continue
        if include_device is not None and t.device != include_device:
            continue

        element_size = t.element_size()
        if getattr(p, 'quant_type', None) in ['fp4', 'nf4']:
            element_size = 0.55 if element_size > 1 else 1.1

        module_mem += t.nelement() * element_size
        if k in weight_patterns:
            weight_mem += t.nelement() * element_size

    if return_split:
        return module_mem, weight_mem, module_mem - weight_mem
    return module_mem


def module_move(module, device, recursive=True, excluded_patterns=None):
    if excluded_patterns is None:
        excluded_patterns = []
    if recursive:
        module.to(device, non_blocking=True)
    else:
        for k, p in module.named_parameters(recurse=False, remove_duplicate=True):
            if all(pattern not in k for pattern in excluded_patterns):
                p.data = p.data.to(device, non_blocking=True)
    return module


def build_module_profile(model, model_gpu_memory_when_using_cpu_swap):
    all_modules = []
    legacy_modules = []

    for m in model.modules():
        if hasattr(m, "parameters_manual_cast"):
            m.total_mem, m.weight_mem, m.extra_mem = module_size(m, return_split=True)
            all_modules.append(m)
        elif hasattr(m, "weight"):
            m.total_mem, m.weight_mem, m.extra_mem = module_size(m, return_split=True)
            legacy_modules.append(m)

    gpu_modules = []
    gpu_modules_only_extras = []
    mem_counter = 0

    for m in legacy_modules.copy():
        gpu_modules.append(m)
        legacy_modules.remove(m)
        mem_counter += m.total_mem

    for m in sorted(all_modules, key=lambda x: x.extra_mem).copy():
        if mem_counter + m.extra_mem < model_gpu_memory_when_using_cpu_swap:
            gpu_modules_only_extras.append(m)
            all_modules.remove(m)
            mem_counter += m.extra_mem

    cpu_modules = all_modules

    for m in sorted(gpu_modules_only_extras, key=lambda x: x.weight_mem).copy():
        if mem_counter + m.weight_mem < model_gpu_memory_when_using_cpu_swap:
            gpu_modules.append(m)
            gpu_modules_only_extras.remove(m)
            mem_counter += m.weight_mem

    return gpu_modules, gpu_modules_only_extras, cpu_modules


class LoadedModel:
    def __init__(self, model):
        self.model = model
        self.model_accelerated = False
        self.device = model.load_device
        self.inclusive_memory = 0
        self.exclusive_memory = 0
        self.load_attempts = 0  # Track load attempts for adaptive strategies

    def compute_inclusive_exclusive_memory(self):
        self.inclusive_memory = module_size(self.model.model, include_device=self.device)
        self.exclusive_memory = module_size(self.model.model, exclude_device=self.device)
        return

    def model_load(self, model_gpu_memory_when_using_cpu_swap=-1):
        self.load_attempts += 1
        do_not_need_cpu_swap = model_gpu_memory_when_using_cpu_swap < 0

        patch_model_to = self.device if do_not_need_cpu_swap else None
        self.model.model_patches_to(self.device)
        self.model.model_patches_to(self.model.model_dtype())

        try:
            self.real_model = self.model.forge_patch_model(patch_model_to)
            self.model.current_device = self.model.load_device
        except OOM_EXCEPTION as e:
            # ADAPTIVE STRATEGY:
            # If OOM, try reducing dtype or switching VRAM state for next attempt
            print(f"OOM encountered during model load: {e}")
            self._adaptive_oom_recovery()
            raise e
        except Exception as e:
            self.model.forge_unpatch_model(self.model.offload_device)
            self.model_unload()
            raise e

        if do_not_need_cpu_swap:
            print('Model fully loaded into GPU.')
        else:
            self._extracted_from_model_load_19(model_gpu_memory_when_using_cpu_swap)

        bake_gguf_model(self.real_model)
        self.model.refresh_loras()

        if is_intel_xpu() and not args.disable_ipex_hijack:
            self.real_model = torch.xpu.optimize(
                self.real_model.eval(), inplace=True,
                auto_kernel_selection=True, graph_mode=True
            )

        return self.real_model

    def _extracted_from_model_load_19(self, model_gpu_memory_when_using_cpu_swap):
        gpu_modules, gpu_modules_only_extras, cpu_modules = build_module_profile(self.real_model, model_gpu_memory_when_using_cpu_swap)
        pin_memory = PIN_SHARED_MEMORY and is_device_cpu(self.model.offload_device)

        mem_counter = 0
        swap_counter = 0

        for m in gpu_modules:
            m.to(self.device)
            mem_counter += m.total_mem

        for m in cpu_modules:
            m.prev_parameters_manual_cast = m.parameters_manual_cast
            m.parameters_manual_cast = True
            m.to(self.model.offload_device)
            if pin_memory:
                m._apply(lambda x: x.pin_memory())
            swap_counter += m.total_mem

        # ADAPTIVE STRATEGY:
        # If too many modules forced to CPU, consider lowering vram_state next time
        if len(cpu_modules) > len(gpu_modules):
            self._adaptive_lower_vram_state()

        for m in gpu_modules_only_extras:
            m.prev_parameters_manual_cast = m.parameters_manual_cast
            m.parameters_manual_cast = True
            module_move(m, device=self.device, recursive=False, excluded_patterns=['weight'])
            if hasattr(m, 'weight') and m.weight is not None:
                m.weight = utils.tensor2parameter(
                    m.weight.to(self.model.offload_device).pin_memory() if pin_memory else m.weight.to(self.model.offload_device)
                )
            mem_counter += m.extra_mem
            swap_counter += m.weight_mem

        swap_flag = 'Shared' if PIN_SHARED_MEMORY else 'CPU'
        method_flag = 'asynchronous' if stream.should_use_stream() else 'blocked'
        print(f"{swap_flag} Swap Loaded ({method_flag} method): {swap_counter / (1024 * 1024):.2f} MB, GPU Loaded: {mem_counter / (1024 * 1024):.2f} MB")

        self.model_accelerated = True
        global signal_empty_cache
        signal_empty_cache = True

    def _adaptive_oom_recovery(self):
        """
        Attempt to recover from OOM by adjusting dtype or VRAM state.
        For example:
        - If we tried fp16 and failed, try bf16 or fp32.
        - If VRAM state is NORMAL_VRAM, switch to LOW_VRAM or NO_VRAM next time.
        """
        global vram_state
        if vram_state == VRAMState.NORMAL_VRAM:
            print("OOM detected, switching VRAM state to LOW_VRAM for next attempt.")
            vram_state = VRAMState.LOW_VRAM
        elif vram_state == VRAMState.LOW_VRAM:
            print("Still OOM at LOW_VRAM, switching to NO_VRAM.")
            vram_state = VRAMState.NO_VRAM
        # Additional logic could try changing model dtype if accessible

    def _adaptive_lower_vram_state(self):
        """
        If too many modules ended up on CPU, we can lower VRAM usage state next time
        to reduce failed attempts.
        """
        global vram_state
        if vram_state == VRAMState.NORMAL_VRAM:
            print("High CPU offload detected, lowering VRAM state to LOW_VRAM for future loads.")
            vram_state = VRAMState.LOW_VRAM

    def model_unload(self, avoid_model_moving=False):
        if self.model_accelerated:
            for m in self.real_model.modules():
                if hasattr(m, "prev_parameters_manual_cast"):
                    m.parameters_manual_cast = m.prev_parameters_manual_cast
                    del m.prev_parameters_manual_cast
            self.model_accelerated = False

        if avoid_model_moving:
            self.model.forge_unpatch_model()
        else:
            self.model.forge_unpatch_model(self.model.offload_device)
            self.model.model_patches_to(self.model.offload_device)

    def __eq__(self, other):
        return self.model is other.model


current_inference_memory = GIGABYTE


def minimum_inference_memory():
    return current_inference_memory


def unload_model_clones(model):
    to_unload = []
    for i in range(len(current_loaded_models)):
        if model.is_clone(current_loaded_models[i].model):
            to_unload = [i] + to_unload

    for i in to_unload:
        current_loaded_models.pop(i).model_unload(avoid_model_moving=True)


def free_memory(memory_required, device, keep_loaded=None, free_all=False):
    global benchmarking_mode
    if keep_loaded is None:
        keep_loaded = []

    # Unload models with no references
    for i in range(len(current_loaded_models) - 1, -1, -1):
        if sys.getrefcount(current_loaded_models[i].model) <= 2:
            current_loaded_models.pop(i).model_unload(avoid_model_moving=True)

    if free_all:
        memory_required = 1e30
        print(f"[Unload] Freeing all memory on {device} with {len(keep_loaded)} models to keep...", end="")
    else:
        print(f"[Unload] Trying to free {memory_required / (1024 * 1024):.2f} MB for {device} with {len(keep_loaded)} models kept...", end="")

    offload_everything = ALWAYS_VRAM_OFFLOAD or vram_state == VRAMState.NO_VRAM
    unloaded_model = False
    for i in range(len(current_loaded_models) - 1, -1, -1):
        if not offload_everything:
            free_memory_now = get_free_memory(device)
            print(f" Current free memory: {free_memory_now / (1024 * 1024):.2f} MB ... ", end="")
            if free_memory_now > memory_required:
                break
        shift_model = current_loaded_models[i]
        if shift_model.device == device and shift_model not in keep_loaded:
            m = current_loaded_models.pop(i)
            print(f" Unloading model {m.model.model.__class__.__name__}...", end="")
            m.model_unload()
            del m
            unloaded_model = True

    if unloaded_model:
        soft_empty_cache()
    elif vram_state != VRAMState.HIGH_VRAM:
        mem_free_total, mem_free_torch = get_free_memory(device, torch_free_too=True)
        if mem_free_torch > mem_free_total * 0.25:
            soft_empty_cache()

    print('Done.')
    return


def compute_model_gpu_memory_when_using_cpu_swap(current_free_mem, inference_memory):
    maximum_memory_available = current_free_mem - inference_memory
    suggestion = max(maximum_memory_available / 1.3, maximum_memory_available - 1.25 * GIGABYTE)
    return int(max(0, suggestion))


def load_models_gpu(models, memory_required=0, hard_memory_preservation=0):
    global vram_state
    execution_start_time = time.perf_counter()
    memory_to_free = max(minimum_inference_memory(), memory_required) + hard_memory_preservation
    memory_for_inference = minimum_inference_memory() + hard_memory_preservation

    models_to_load = []
    models_already_loaded = []
    for x in models:
        loaded_model = LoadedModel(x)
        if loaded_model in current_loaded_models:
            index = current_loaded_models.index(loaded_model)
            current_loaded_models.insert(0, current_loaded_models.pop(index))
            models_already_loaded.append(loaded_model)
        else:
            models_to_load.append(loaded_model)

    if not models_to_load:
        devs = {m.device for m in models_already_loaded}
        for d in devs:
            if d != torch.device("cpu"):
                free_memory(memory_to_free, d, models_already_loaded)
        moving_time = time.perf_counter() - execution_start_time
        if moving_time > 0.1:
            print(f'Memory cleanup took {moving_time:.2f} seconds')
        return

    for loaded_model in models_to_load:
        unload_model_clones(loaded_model.model)

    total_memory_required = {}
    for loaded_model in models_to_load:
        loaded_model.compute_inclusive_exclusive_memory()
        total_memory_required[loaded_model.device] = total_memory_required.get(loaded_model.device, 0) + loaded_model.exclusive_memory + loaded_model.inclusive_memory * 0.25

    for device, value in total_memory_required.items():
        if device != torch.device("cpu"):
            free_memory(value * 1.3 + memory_to_free, device, models_already_loaded)

    for loaded_model in models_to_load:
        model = loaded_model.model
        torch_dev = model.load_device
        vram_set_state = VRAMState.DISABLED if is_device_cpu(torch_dev) else vram_state
        model_gpu_memory_when_using_cpu_swap = -1

        if lowvram_available and vram_set_state in [VRAMState.LOW_VRAM, VRAMState.NORMAL_VRAM]:
            model_require = loaded_model.exclusive_memory
            previously_loaded = loaded_model.inclusive_memory
            current_free_mem = get_free_memory(torch_dev)
            estimated_remaining_memory = current_free_mem - model_require - memory_for_inference

            print(f"[Memory Management] Target: {model.model.__class__.__name__}, Free GPU: {current_free_mem / (1024 * 1024):.2f} MB, "
                  f"Model Require: {model_require / (1024 * 1024):.2f} MB, Previously Loaded: {previously_loaded / (1024 * 1024):.2f} MB, "
                  f"Inference Require: {memory_for_inference / (1024 * 1024):.2f} MB, Remaining: {estimated_remaining_memory / (1024 * 1024):.2f} MB, ",
                  end="")

            if estimated_remaining_memory < 0:
                vram_set_state = VRAMState.LOW_VRAM
                model_gpu_memory_when_using_cpu_swap = compute_model_gpu_memory_when_using_cpu_swap(current_free_mem, memory_for_inference)
                if previously_loaded > 0:
                    model_gpu_memory_when_using_cpu_swap = previously_loaded

        if vram_set_state == VRAMState.NO_VRAM:
            model_gpu_memory_when_using_cpu_swap = 0

        # ADAPTIVE STRATEGY:
        # If many attempts fail, try a safer dtype (like bf16 or fp32) if accessible through the model interface
        # (This requires model interface to accept and adjust dtype.)
        # This is a placeholder comment; actual dtype adjustment logic would go where model loaders are invoked.

        loaded_model.model_load(model_gpu_memory_when_using_cpu_swap)
        current_loaded_models.insert(0, loaded_model)

    moving_time = time.perf_counter() - execution_start_time
    print(f'Loading model(s) took {moving_time:.2f} seconds')
    return


def load_model_gpu(model):
    return load_models_gpu([model])


def cleanup_models():
    to_delete = []
    for i in range(len(current_loaded_models)):
        if sys.getrefcount(current_loaded_models[i].model) <= 2:
            to_delete = [i] + to_delete

    for i in to_delete:
        x = current_loaded_models.pop(i)
        x.model_unload()
        del x


def dtype_size(dtype):
    if dtype in [torch.float16, torch.bfloat16]:
        return 2
    elif dtype == torch.float32:
        return 4
    else:
        try:
            return dtype.itemsize
        except:
            return 4


def unet_offload_device():
    if vram_state == VRAMState.HIGH_VRAM:
        return get_torch_device()
    else:
        return torch.device("cpu")


def unet_inital_load_device(parameters, dtype):
    torch_dev = get_torch_device()
    if vram_state == VRAMState.HIGH_VRAM:
        return torch_dev

    cpu_dev = torch.device("cpu")
    if ALWAYS_VRAM_OFFLOAD:
        return cpu_dev

    model_size = dtype_size(dtype) * parameters
    mem_dev = get_free_memory(torch_dev)
    mem_cpu = get_free_memory(cpu_dev)
    return torch_dev if mem_dev > mem_cpu and model_size < mem_dev else cpu_dev


def unet_dtype(device=None, model_params=0, supported_dtypes=None):
    if supported_dtypes is None:
        supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    if args.unet_in_bf16:
        return torch.bfloat16
    if args.unet_in_fp16:
        return torch.float16
    if args.unet_in_fp8_e4m3fn:
        return torch.float8_e4m3fn
    if args.unet_in_fp8_e5m2:
        return torch.float8_e5m2

    for candidate in supported_dtypes:
        if candidate == torch.float16 and should_use_fp16(device, model_params=model_params, prioritize_performance=True, manual_cast=True):
            return candidate
        if candidate == torch.bfloat16 and should_use_bf16(device, model_params=model_params, prioritize_performance=True, manual_cast=True):
            return candidate

    return torch.float32


def get_computation_dtype(inference_device, parameters=0, supported_dtypes=None):
    if supported_dtypes is None:
        supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    for candidate in supported_dtypes:
        if candidate == torch.float16 and should_use_fp16(inference_device, model_params=parameters, prioritize_performance=True, manual_cast=False):
            return candidate
        if candidate == torch.bfloat16 and should_use_bf16(inference_device, model_params=parameters, prioritize_performance=True, manual_cast=False):
            return candidate
    return torch.float32


def text_encoder_offload_device():
    return get_torch_device() if args.always_gpu else torch.device("cpu")


def text_encoder_device():
    if args.always_gpu:
        return get_torch_device()
    elif vram_state in [VRAMState.HIGH_VRAM, VRAMState.NORMAL_VRAM]:
        if should_use_fp16(prioritize_performance=False):
            return get_torch_device()
        else:
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def text_encoder_dtype(device=None):
    if args.clip_in_fp8_e4m3fn:
        return torch.float8_e4m3fn
    elif args.clip_in_fp8_e5m2:
        return torch.float8_e5m2
    elif args.clip_in_fp16:
        return torch.float16
    elif args.clip_in_fp32:
        return torch.float32

    return torch.float16 if not is_device_cpu(device) else torch.float16


def intermediate_device():
    return get_torch_device() if args.always_gpu else torch.device("cpu")


def vae_device():
    return torch.device("cpu") if args.vae_in_cpu else get_torch_device()


def vae_offload_device():
    return get_torch_device() if args.always_gpu else torch.device("cpu")


def vae_dtype(device=None, allowed_dtypes=None):
    global VAE_DTYPES
    if allowed_dtypes is None:
        allowed_dtypes = []
    if args.vae_in_fp16:
        return torch.float16
    elif args.vae_in_bf16:
        return torch.bfloat16
    elif args.vae_in_fp32:
        return torch.float32

    for d in allowed_dtypes:
        if d == torch.float16 and should_use_fp16(device, prioritize_performance=False):
            return d
        if d in VAE_DTYPES:
            return d
    return VAE_DTYPES[0]


print(f"VAE dtype preferences: {VAE_DTYPES} -> {vae_dtype()}")


def get_autocast_device(dev):
    return dev.type if hasattr(dev, 'type') else "cuda"


def supports_dtype(device, dtype):
    if dtype == torch.float32:
        return True
    if is_device_cpu(device):
        return False
    return True if dtype == torch.float16 else dtype == torch.bfloat16


def supports_cast(device, dtype):
    if dtype == torch.float32:
        return True
    if dtype == torch.float16:
        return True
    if directml_enabled:
        return False
    if dtype == torch.bfloat16:
        return True
    if is_device_mps(device):
        return False
    return True if dtype in [torch.float8_e4m3fn, torch.float8_e5m2] else False


def pick_weight_dtype(dtype, fallback_dtype, device=None):
    if dtype is None or (dtype is not None and dtype_size(dtype) > dtype_size(fallback_dtype)):
        dtype = fallback_dtype
    if not supports_cast(device, dtype):
        dtype = fallback_dtype
    return dtype


def device_supports_non_blocking(device):
    if is_device_mps(device):
        return False
    if is_intel_xpu():
        return False
    return False if args.pytorch_deterministic else not directml_enabled


def device_should_use_non_blocking(device):
    return False


def force_channels_last():
    return bool(args.force_channels_last)


def cast_to_device(tensor, device, dtype, copy=False):
    non_blocking = device_should_use_non_blocking(device)
    return tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)


def xformers_enabled():
    global directml_enabled, cpu_state
    if cpu_state != CPUState.GPU:
        return False
    if is_intel_xpu():
        return False
    return False if directml_enabled else XFORMERS_IS_AVAILABLE


def xformers_enabled_vae():
    enabled = xformers_enabled()
    return XFORMERS_ENABLED_VAE if enabled else False


def pytorch_attention_enabled():
    global ENABLE_PYTORCH_ATTENTION
    return ENABLE_PYTORCH_ATTENTION


def pytorch_attention_flash_attention():
    global ENABLE_PYTORCH_ATTENTION
    if ENABLE_PYTORCH_ATTENTION:
        if is_nvidia() or is_intel_xpu():
            return True
    return False


def force_upcast_attention_dtype():
    upcast = args.force_upcast_attention
    try:
        if platform.mac_ver()[0] in ['14.5']:
            upcast = True
    except:
        pass
    return torch.float32 if upcast else None


def get_free_memory(dev=None, torch_free_too=False):
    stats = get_mem_stats(dev)
    mem_total = stats['mem_total']
    mem_reserved = stats['mem_reserved']
    mem_active = stats['mem_active']

    if hasattr(dev, 'type') and dev.type in ['cpu', 'mps', 'xpu']:
        mem_free_total = psutil.virtual_memory().available if dev.type in ['cpu','mps'] else (mem_total - mem_reserved)
        mem_free_torch = mem_free_total if dev.type in ['cpu','mps'] else (mem_reserved - mem_active)
    elif directml_enabled:
        mem_free_total = GIGABYTE
        mem_free_torch = mem_free_total
    else:
        mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
        mem_free_torch = mem_reserved - mem_active
        mem_free_total = mem_free_cuda + mem_free_torch

    return (mem_free_total, mem_free_torch) if torch_free_too else mem_free_total


def cpu_mode():
    global cpu_state
    return cpu_state == CPUState.CPU


def mps_mode():
    global cpu_state
    return cpu_state == CPUState.MPS


def is_device_type(device, type):
    return bool(hasattr(device, 'type') and device.type == type)


def is_device_cpu(device):
    return is_device_type(device, 'cpu')


def is_device_mps(device):
    return is_device_type(device, 'mps')


def is_device_cuda(device):
    return is_device_type(device, 'cuda')


def should_use_fp16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    global directml_enabled

    if device is not None and is_device_cpu(device):
        return False
    if FORCE_FP16:
        return True
    if device is not None and is_device_mps(device):
        return True
    if FORCE_FP32:
        return False
    if directml_enabled:
        return False
    if mps_mode():
        return True
    if cpu_mode():
        return False
    if is_intel_xpu():
        return True
    if torch.version.hip:
        return True

    props = torch.cuda.get_device_properties("cuda")
    if props.major >= 8:
        return True
    if props.major < 6:
        return False

    nvidia_10_series = ["1080", "1070", "titan x", "p3000", "p3200", "p4000", "p4200",
                        "p5000", "p5200", "p6000", "1060", "1050", "p40", "p100", "p6", "p4"]
    for x in nvidia_10_series:
        if x in props.name.lower():
            if not manual_cast:
                return False
            free_model_memory = get_free_memory() * 0.9 - minimum_inference_memory()
            if (not prioritize_performance) or model_params * 4 > free_model_memory:
                return True

    nvidia_16_series = ["1660", "1650", "1630", "T500", "T550", "T600", "MX550", "MX450",
                        "CMP 30HX", "T2000", "T1000", "T1200"]
    return all(x not in props.name for x in nvidia_16_series)


def should_use_bf16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    if device is not None and is_device_cpu(device):
        return False
    if device is not None and is_device_mps(device):
        return True
    if FORCE_FP32:
        return False
    if directml_enabled:
        return False
    if mps_mode():
        return True
    if cpu_mode():
        return False
    if is_intel_xpu():
        return True

    if device is None:
        device = torch.device("cuda")

    props = torch.cuda.get_device_properties(device)
    if props.major >= 8:
        return True

    if torch.cuda.is_bf16_supported() and manual_cast:
        free_model_memory = get_free_memory() * 0.9 - minimum_inference_memory()
        if (not prioritize_performance) or model_params * 4 > free_model_memory:
            return True

    return False


def can_install_bnb():
    try:
        if not torch.cuda.is_available():
            return False
        cuda_version = tuple(int(x) for x in torch.version.cuda.split('.'))
        return cuda_version >= (11, 7)
    except:
        return False


signal_empty_cache = False


def soft_empty_cache(force=False):
    global cpu_state, signal_empty_cache
    if cpu_state == CPUState.MPS:
        torch.mps.empty_cache()
    elif is_intel_xpu():
        pass
    elif torch.cuda.is_available():
        if force or is_nvidia():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    signal_empty_cache = False
    return


def unload_all_models():
    free_memory(1e30, get_torch_device(), free_all=True)
    current_loaded_models.clear()
