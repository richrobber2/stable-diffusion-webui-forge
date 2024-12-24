from typing import Optional, Tuple, Dict, Any, Callable, Union, Protocol, TypeVar
from dataclasses import dataclass
from functools import lru_cache
import contextlib
import math
import torch
import einops
from torch import Tensor
import threading
import weakref
from abc import ABC, abstractmethod

from backend.args import args
from backend import memory_management
from backend.misc.sub_quadratic_attention import efficient_dot_product_attention

# Custom exceptions for better error handling
class AttentionError(Exception):
    """Base exception for attention-related errors"""
    pass

class MemoryError(AttentionError):
    """Raised when running into memory issues"""
    pass

class InvalidConfigurationError(AttentionError):
    """Raised when attention configuration is invalid"""
    pass

# Define attention strategy protocol
class AttentionStrategy(Protocol):
    def __call__(self, q: Tensor, k: Tensor, v: Tensor,
                 heads: int, mask: Optional[Tensor] = None,
                 attn_precision: Optional[torch.dtype] = None) -> Tensor:
        ...

# Memory-aware cache implementation
class MemoryAwareCache:
    def __init__(self, max_memory_fraction: float = 0.3):
        self.max_memory_fraction = max_memory_fraction
        self.cache = weakref.WeakValueDictionary()
        self.lock = threading.Lock()

    def get(self, key: Tuple) -> Optional[Tensor]:
        with self.lock:
            return self.cache.get(key)

    def set(self, key: Tuple, value: Tensor) -> None:
        with self.lock:
            self.cache[key] = value
            self._maybe_cleanup()

    def _maybe_cleanup(self) -> None:
        if not self._check_memory_usage():
            self.cache.clear()

    def _check_memory_usage(self) -> bool:
        mem_free_total, mem_free_torch = memory_management.get_free_memory(torch.device('cuda'), True)
        return mem_free_torch / mem_free_total > self.max_memory_fraction

# Performance monitoring
class AttentionMetrics:
    def __init__(self):
        self.call_durations = []
        self.cache_hits = 0
        self.total_calls = 0

    def record_call(self, duration: float, cache_hit: bool) -> None:
        self.call_durations.append(duration)
        if cache_hit:
            self.cache_hits += 1
        self.total_calls += 1

    def get_stats(self) -> Dict[str, float]:
        avg_duration = sum(self.call_durations) / len(self.call_durations) if self.call_durations else 0.0
        cache_hit_rate = self.cache_hits / self.total_calls if self.total_calls else 0.0
        return {
            'avg_duration': avg_duration,
            'cache_hit_rate': cache_hit_rate
        }

@dataclass
class AttentionConfig:
    dim_head: int
    attn_precision: torch.dtype
    heads: int
    device: torch.device
    dtype: torch.dtype
    use_cache: bool = True
    max_cache_memory_fraction: float = 0.3
    enable_metrics: bool = False

# Base attention implementation
class BaseAttention(ABC):
    def __init__(self, config: AttentionConfig):
        self.config = config
        self.cache = MemoryAwareCache(config.max_cache_memory_fraction) if config.use_cache else None
        self.metrics = AttentionMetrics() if config.enable_metrics else None

    @abstractmethod
    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                heads: int, mask: Optional[Tensor] = None) -> Tensor:
        pass

    def __call__(self, *args, **kwargs) -> Tensor:
        cache_key = self._get_cache_key(*args, **kwargs)
        if self.cache:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                if self.metrics:
                    self.metrics.record_call(0.0, True)
                return cached_result

        start_time = time.time()
        result = self.forward(*args, **kwargs)
        duration = time.time() - start_time

        if self.cache:
            self.cache.set(cache_key, result)
        if self.metrics:
            self.metrics.record_call(duration, False)

        return result

    def _get_cache_key(self, *args, **kwargs) -> Tuple:
        return args, tuple(sorted(kwargs.items()))

# Implement concrete attention strategies
class XFormersAttention(BaseAttention):
    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                heads: int, mask: Optional[Tensor] = None) -> Tensor:
        return attention_xformers(q, k, v, heads, mask, self.config.attn_precision)

class PyTorchAttention(BaseAttention):
    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                heads: int, mask: Optional[Tensor] = None) -> Tensor:
        return attention_pytorch(q, k, v, heads, mask, self.config.attn_precision)

# Factory for attention mechanisms
class AttentionFactory:
    @staticmethod
    def create(config: AttentionConfig) -> BaseAttention:
        if _BACKEND_CONFIG['xformers_enabled']:
            return XFormersAttention(config)
        elif _BACKEND_CONFIG['pytorch_attention_enabled']:
            return PyTorchAttention(config)
        else:
            raise InvalidConfigurationError("No valid attention backend available.")

# Cache management
def clear_attention_caches():
    """Clear all LRU caches used in attention calculations"""
    get_scale.cache_clear()
    get_mask_for_heads.cache_clear()
    create_causal_mask.cache_clear()

def resize_attention_caches(new_maxsize: int = 32):
    """Resize all attention-related LRU caches"""
    get_scale.cache_clear()
    get_mask_for_heads.cache_clear()
    create_causal_mask.cache_clear()
    
    get_scale._maxsize = new_maxsize
    get_mask_for_heads._maxsize = new_maxsize
    create_causal_mask._maxsize = new_maxsize

# Unified mask handling
def prepare_attention_mask(mask: Optional[torch.Tensor], 
                         target_shape: Tuple[int, ...],
                         heads: int,
                         device: torch.device,
                         dtype: torch.dtype) -> Optional[torch.Tensor]:
    """Unified mask preparation that handles all input sizes and shapes"""
    if mask is None:
        return None
        
    # Ensure mask is the correct dtype
    if mask.dtype != dtype:
        mask = mask.to(dtype)
    
    # Calculate required broadcasting shape
    batch_size = target_shape[0]
    seq_len = target_shape[1]
    
    # Reshape mask to (batch, 1, seq_len, seq_len) if needed
    if len(mask.shape) == 2:
        mask = mask.unsqueeze(0)
    if len(mask.shape) == 3:
        mask = mask.unsqueeze(1)
        
    # Broadcast mask for attention heads
    mask = mask.expand(batch_size, heads, seq_len, seq_len)
    
    # Reshape to (batch * heads, seq_len, seq_len)
    mask = mask.reshape(-1, seq_len, seq_len)
    
    return mask

# Dynamic chunk size calculation
def calculate_optimal_chunk_size(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> int:
    """Calculate optimal chunk size based on available memory and tensor sizes"""
    device = q.device
    dtype = q.dtype
    
    # Get available memory
    mem_free_total, mem_free_torch = memory_management.get_free_memory(device, True)
    
    # Calculate memory requirements per token
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    tokens_in_sequence = q.shape[1]
    head_size = q.shape[-1]
    
    # Memory for attention matrix (N x N)
    attn_memory = tokens_in_sequence * tokens_in_sequence * bytes_per_element
    
    # Memory for key/value cache
    kv_memory = 2 * tokens_in_sequence * head_size * bytes_per_element
    
    # Target using 80% of available memory
    target_memory = 0.8 * mem_free_total
    
    # Calculate chunk size
    chunk_size = max(128, min(
        512,  # Maximum chunk size
        int(math.sqrt(target_memory / (attn_memory + kv_memory)))
    ))
    
    # Round to nearest power of 2 for better performance
    chunk_size = 2 ** int(math.log2(chunk_size))
    
    return chunk_size

# Implement LRU caches for better memory management
@lru_cache(maxsize=128)
def get_scale(dim_head: int, attn_precision: torch.dtype, heads: int) -> float:
    """Cached computation of attention scale factor"""
    epsilon = 1e-8 if attn_precision == torch.float32 else 1e-5
    return 1.0 / math.sqrt(dim_head + epsilon)

@lru_cache(maxsize=128)
def get_mask_for_heads(mask_shape: Tuple[int, ...], heads: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Cached mask expansion for attention heads"""
    if len(mask_shape) == 2:
        mask = torch.ones(mask_shape, device=device, dtype=dtype)
        return mask.unsqueeze(1).repeat_interleave(heads, dim=0)
    else:
        mask = torch.ones(mask_shape, device=device, dtype=dtype)
        return mask.repeat_interleave(heads, dim=0)

def get_attn_precision(attn_precision: Optional[torch.dtype] = torch.float32) -> Optional[torch.dtype]:
    if args.disable_attention_upcast:
        return None
    return FORCE_UPCAST_ATTENTION_DTYPE or attn_precision

# Initialize backend configuration once
_BACKEND_CONFIG = {
    'xformers_enabled': memory_management.xformers_enabled(),
    'pytorch_attention_enabled': memory_management.pytorch_attention_enabled(),
    'force_upcast_attention': memory_management.force_upcast_attention_dtype()
}

# Check xformers version compatibility once
BROKEN_XFORMERS = False
if _BACKEND_CONFIG['xformers_enabled']:
    try:
        import xformers
        import xformers.ops
        BROKEN_XFORMERS = xformers.__version__.startswith("0.0.2") and not xformers.__version__.startswith("0.0.20")
    except ImportError:
        _BACKEND_CONFIG['xformers_enabled'] = False

# Add efficient tensor caching
@lru_cache(maxsize=256)
def create_causal_mask(size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Cache causal masks for better performance"""
    return torch.triu(torch.ones(size, size, device=device, dtype=dtype) * -float('inf'), diagonal=1)

# Optimize tensor reshaping
def efficient_reshape_qkv(tensor: Tensor, b: int, heads: int, dim_head: int, skip_reshape: bool = False) -> Tensor:
    """More efficient tensor reshaping using contiguous memory"""
    if skip_reshape:
        return tensor.view(b * heads, -1, dim_head).contiguous()
    return tensor.view(b, -1, heads, dim_head).transpose(1, 2).contiguous().view(b * heads, -1, dim_head)

def simple_rearrange_qkv(tensor: torch.Tensor, b: int, heads: int, dim_head: int, skip_reshape: bool = False) -> torch.Tensor:
    """Optimized tensor reshaping for attention"""
    if skip_reshape:
        return tensor.view(b * heads, -1, dim_head)
    return tensor.view(b, -1, heads, dim_head).transpose(1, 2).reshape(b * heads, -1, dim_head)

def create_spiral_bias(height: int, width: int, 
                      a: float = 0.0, 
                      b: float = 0.1,
                      device: torch.device = None,
                      dtype: torch.dtype = None) -> torch.Tensor:
    """Create a spiral attention bias matrix.
    
    Args:
        height: Image height
        width: Image width
        a: Spiral starting radius
        b: Spiral growth rate
        device: Target device
        dtype: Target dtype
    
    Returns:
        Tensor: (H*W, H*W) spiral bias matrix
    """
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing='ij'
    )
    
    # Normalize coordinates to [-1, 1]
    x_norm = (x_coords.float() / (width - 1)) * 2 - 1
    y_norm = (y_coords.float() / (height - 1)) * 2 - 1
    
    # Convert to polar coordinates
    r = torch.sqrt(x_norm**2 + y_norm**2)
    theta = torch.atan2(y_norm, x_norm)
    theta_pos = theta + math.pi
    
    # Create spiral pattern
    spiral_r = a + b * theta_pos
    spiral_dist = torch.abs(r - spiral_r)
    spiral_score = torch.exp(-spiral_dist * 5.0)
    
    # Create bias matrix
    flat_score = spiral_score.view(-1)
    spiral_bias = (flat_score.unsqueeze(0) + flat_score.unsqueeze(1)) / 2.0
    
    return spiral_bias.to(dtype=dtype)

@lru_cache(maxsize=32)
def get_spiral_bias(height: int, width: int, heads: int, 
                   device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Cached spiral bias computation"""
    bias = create_spiral_bias(height, width, device=device, dtype=dtype)
    return bias.unsqueeze(0).repeat(heads, 1, 1)

def attention_with_spiral_bias(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                             heads: int, mask: Optional[torch.Tensor] = None,
                             attn_precision: Optional[torch.dtype] = None,
                             use_spiral: bool = True) -> torch.Tensor:
    """Attention computation with optional spiral bias"""
    b, seq_len, _ = q.shape
    height = width = int(math.sqrt(seq_len))

    # Regular attention computation
    attn_precision = get_attn_precision(attn_precision)
    scale = get_scale(q.shape[-1], attn_precision, heads)

    # Reshape for attention
    q = simple_rearrange_qkv(q, b, heads, q.shape[-1] // heads)
    k = simple_rearrange_qkv(k, b, heads, k.shape[-1] // heads)
    v = simple_rearrange_qkv(v, b, heads, k.shape[-1] // heads)

    # Compute attention scores
    sim = q.bmm(k.transpose(-2, -1))
    sim *= scale

    # Add spiral bias if requested
    if use_spiral and height**2 == seq_len:  # Only apply to square inputs
        spiral_bias = get_spiral_bias(height, width, heads, q.device, q.dtype)
        sim += spiral_bias.log()  # Convert closeness to additive bias

    # Apply mask if provided
    if mask is not None:
        sim = sim + mask

    # Softmax and final computation
    sim = torch.softmax(sim, dim=-1)
    hidden_states = torch.bmm(sim, v)

    # Reshape output
    hidden_states = hidden_states.view(b, heads, seq_len, v.shape[-1])
    hidden_states = hidden_states.transpose(1, 2).reshape(b, seq_len, heads * v.shape[-1])

    return hidden_states

def attention_basic(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                   heads: int, mask: Optional[torch.Tensor] = None, 
                   attn_precision: torch.dtype = torch.float32,
                   skip_reshape: bool = False,
                   use_spiral: bool = False) -> torch.Tensor:
    """Optimized basic attention implementation"""
    if use_spiral:
        return attention_with_spiral_bias(q, k, v, heads, mask, attn_precision)
    with torch.inference_mode(), torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
        attn_precision = get_attn_precision(attn_precision)
        b, seq, total_dim = q.shape
        dim_head = q.shape[-1] if skip_reshape else total_dim // heads
        scale = get_scale(dim_head, attn_precision, heads)

        q = simple_rearrange_qkv(q, b, heads, dim_head, skip_reshape)
        k = simple_rearrange_qkv(k, b, heads, dim_head, skip_reshape)
        v = simple_rearrange_qkv(v, b, heads, dim_head, skip_reshape)

        try:
            if attn_precision == torch.float32:
                out = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=False
                )
                return out.view(b, heads, seq, dim_head).transpose(1, 2).reshape(b, seq, heads * dim_head)
        except:
            pass

        sim_dtype = torch.float32 if attn_precision == torch.float32 else q.dtype
        sim = q_.bmm(k_.transpose(-2, -1))
        sim *= scale

        if mask is not None:
            if mask.dtype == torch.bool:
                sim.masked_fill_(~mask, -torch.finfo(sim.dtype).max)
            else:
                sim = sim + mask

        sim = torch.softmax(sim, dim=-1)
        hidden_states = torch.bmm(sim, v)
        return hidden_states.view(b, heads, seq, dim_head).transpose(1, 2).reshape(b, seq, heads * dim_head)

def attention_sub_quad(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      heads: int, mask: Optional[torch.Tensor] = None,
                      attn_precision: Optional[torch.dtype] = None,
                      skip_reshape: bool = False) -> torch.Tensor:
    """Memory-efficient sub-quadratic attention"""
    attn_precision = get_attn_precision(attn_precision)
    b, seq, total_dim = q.shape
    dim_head = q.shape[-1] if skip_reshape else total_dim // heads
    scale = get_scale(dim_head, attn_precision, heads)

    q = simple_rearrange_qkv(q, b, heads, dim_head, skip_reshape)
    k = simple_rearrange_qkv(k, b, heads, dim_head, skip_reshape)
    v = simple_rearrange_qkv(v, b, heads, dim_head, skip_reshape)

    dtype = torch.float32 if attn_precision == torch.float32 else q.dtype

    hidden_states = efficient_dot_product_attention(
        q, k, v,
        query_chunk_size=512,
        kv_chunk_size=k.shape[1],
        kv_chunk_size_min=None,
        use_checkpoint=False,
        upcast_attention=(attn_precision == torch.float32 and q.dtype != torch.float32),
        mask=mask,
    ).to(dtype)

    return hidden_states.view(b, heads, seq, dim_head).transpose(1, 2).reshape(b, seq, heads * dim_head)

def attention_split(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   heads: int, mask: Optional[torch.Tensor] = None,
                   attn_precision: Optional[torch.dtype] = None,
                   skip_reshape: bool = False) -> torch.Tensor:
    return attention_sub_quad(q, k, v, heads, mask, attn_precision, skip_reshape)

def attention_xformers(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      heads: int, mask: Optional[torch.Tensor] = None,
                      attn_precision: Optional[torch.dtype] = None,
                      skip_reshape: bool = False) -> torch.Tensor:
    """Optimized xformers attention implementation"""
    if skip_reshape:
        b, seq, _, dim_head = q.shape
    else:
        b, seq, total_dim = q.shape
        dim_head = total_dim // heads

    if BROKEN_XFORMERS and b * heads > 65535:
        return attention_pytorch(q, k, v, heads, mask, skip_reshape=skip_reshape)

    q = simple_rearrange_qkv(q, b, heads, dim_head, skip_reshape)
    k = simple_rearrange_qkv(k, b, heads, dim_head, skip_reshape)
    v = simple_rearrange_qkv(v, b, heads, dim_head, skip_reshape)

    if mask is not None:
        mask = get_mask_for_heads(mask.shape, heads, mask.device, mask.dtype)

    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask)
    return out.view(b, heads, seq, dim_head).transpose(1, 2).reshape(b, seq, heads * dim_head)

def attention_pytorch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     heads: int, mask: Optional[torch.Tensor] = None,
                     attn_precision: Optional[torch.dtype] = None,
                     skip_reshape: bool = False) -> torch.Tensor:
    """PyTorch native attention implementation"""
    attn_precision = get_attn_precision(attn_precision)

    b, q_seq, q_totdim = q.shape
    b_k, k_seq, k_totdim = k.shape
    b_v, v_seq, v_totdim = v.shape

    if not (b == b_k == b_v):
        raise RuntimeError("Batch size mismatch in q, k, v.")

    if heads == 0:
        raise RuntimeError("Heads cannot be zero.")
    if q_totdim % heads != 0:
        raise RuntimeError(f"q_totdim {q_totdim} not divisible by heads {heads}.")

    dim_head = q_totdim // heads
    if k_totdim % heads != 0 or v_totdim % heads != 0:
        raise RuntimeError("Key/Value total dimensions not divisible by heads.")

    if (k_totdim // heads) != dim_head or (v_totdim // heads) != dim_head:
        raise RuntimeError("Dim_head from q doesn't match k or v.")

    if not skip_reshape:
        q_ = q.view(b, q_seq, heads, dim_head).transpose(1, 2).reshape(b * heads, q_seq, dim_head)
        k_ = k.view(b, k_seq, heads, dim_head).transpose(1, 2).reshape(b * heads, k_seq, dim_head)
        v_ = v.view(b, v_seq, heads, dim_head).transpose(1, 2).reshape(b * heads, v_seq, dim_head)
    else:
        q_, k_, v_ = q, k, v

    scale = get_scale(dim_head, attn_precision, heads)

    out = torch.nn.functional.scaled_dot_product_attention(
        q_, k_, v_, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=False
    )

    out = out.view(b, heads, q_seq, dim_head).transpose(1, 2).reshape(b, q_seq, heads * dim_head)
    return out

def slice_attention_single_head_spatial(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Memory-efficient spatial attention slicing"""
    with torch.inference_mode():
        dtype = torch.float32 if q.dtype == torch.float16 else q.dtype
        b, hw, c = q.shape
        scale = 1.0 / math.sqrt(q.shape[-1] + (1e-5 if dtype == torch.float32 else 1e-4))

        r1 = torch.empty((v.shape[0], v.shape[1], q.shape[1]), dtype=q.dtype, device=q.device)

        mem_free_total, mem_free_torch = memory_management.get_free_memory(q.device, True)
        tensor_size = q.numel() * k.shape[2] * q.element_size()

        steps = max(1, min(128, 2 ** math.ceil(math.log2(tensor_size * 3.0 / mem_free_total))))
        slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]

        for i in range(0, q.shape[1], slice_size):
            end = i + slice_size
            s1 = q[:, i:end].bmm(k)
            s1 *= scale
            s1 = torch.softmax(s1, dim=-1)
            r1[:, :, i:end].copy_(torch.bmm(v, s1.transpose(-2, -1)))
            del s1
        return r1

def normal_attention_single_head_spatial(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Standard spatial attention implementation"""
    b, c, h, w = q.shape
    q_ = q.view(b, c, h*w).transpose(1, 2)
    k_ = k.view(b, c, h*w)
    v_ = v.view(b, c, h*w)

    r1 = slice_attention_single_head_spatial(q_, k_, v_)
    return r1.view(b, c, h, w)

def xformers_attention_single_head_spatial(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Xformers-based spatial attention"""
    B, C, H, W = q.shape
    try:
        q_ = q.view(B, C, H*W).transpose(1, 2).contiguous()
        k_ = k.view(B, C, H*W).transpose(1, 2).contiguous()
        v_ = v.view(B, C, H*W).transpose(1, 2).contiguous()
        out = xformers.ops.memory_efficient_attention(q_, k_, v_, attn_bias=None)
        return out.transpose(1, 2).view(B, C, H, W)
    except RuntimeError:
        q_ = q.view(B, C, H*W).transpose(1,2)
        k_ = k.view(B, C, H*W)
        v_ = v.view(B, C, H*W)
        return slice_attention_single_head_spatial(q_, k_, v_).view(B, C, H, W)

def pytorch_attention_single_head_spatial(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """PyTorch native spatial attention"""
    B, C, H, W = q.shape
    try:
        q_ = q.view(B, 1, C, H*W).transpose(2, 3)
        k_ = k.view(B, 1, C, H*W).transpose(2, 3)
        v_ = v.view(B, 1, C, H*W).transpose(2, 3)
        out = torch.nn.functional.scaled_dot_product_attention(
            q_, k_, v_, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        return out.transpose(2, 3).view(B, C, H, W)
    except memory_management.OOM_EXCEPTION:
        q_ = q.view(B, C, H*W).transpose(1,2)
        k_ = k.view(B, C, H*W)
        v_ = v.view(B, C, H*W)
        return slice_attention_single_head_spatial(q_, k_, v_).view(B, C, H, W)

# Select attention implementations based on backend configuration
if _BACKEND_CONFIG['xformers_enabled']:
    attention_function = attention_xformers
    attention_function_single_head_spatial = xformers_attention_single_head_spatial
elif _BACKEND_CONFIG['pytorch_attention_enabled']:
    attention_function = attention_pytorch
    attention_function_single_head_spatial = pytorch_attention_single_head_spatial
elif args.attention_split:
    attention_function = attention_split
    attention_function_single_head_spatial = normal_attention_single_head_spatial
else:
    attention_function = attention_sub_quad
    attention_function_single_head_spatial = normal_attention_single_head_spatial

# Memory-efficient attention implementation
def attention_memory_efficient(q: Tensor, k: Tensor, v: Tensor,
                             heads: int, mask: Optional[Tensor] = None,
                             attn_precision: Optional[torch.dtype] = None) -> Tensor:
    """Memory-efficient attention using dynamic chunking"""
    chunk_size = calculate_optimal_chunk_size(q, k, v)
    
    try:
        # Try PyTorch's native attention first
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
            )
    except (RuntimeError, memory_management.OOM_EXCEPTION):
        # Fallback to chunked attention
        b, seq_len, _ = q.shape
        output = torch.zeros_like(q)
        
        for i in range(0, seq_len, chunk_size):
            chunk_q = q[:, i:i + chunk_size]
            chunk_mask = mask[:, i:i + chunk_size] if mask is not None else None
            
            # Process chunks
            chunk_output = process_attention_chunk(chunk_q, k, v, chunk_mask, chunk_size)
            output[:, i:i + chunk_size] = chunk_output
            
        return output

class AttentionProcessorForge:
    """Memory-optimized attention processor"""
    def __init__(self, config: Optional[AttentionConfig] = None):
        self.config = config or AttentionConfig(
            dim_head=64,
            attn_precision=torch.float32,
            heads=8,
            device=torch.device('cuda'),
            dtype=torch.float32
        )
        self.attention = AttentionFactory.create(self.config)
        self._attention_cache = {}
        self._last_clear_time = time.time()
        self._cache_lifetime = 300  # Clear cache every 5 minutes
        self.use_spiral = True  # Add spiral attention flag
        
    def _maybe_clear_cache(self):
        current_time = time.time()
        if current_time - self._last_clear_time > self._cache_lifetime:
            self._attention_cache.clear()
            clear_attention_caches()
            self._last_clear_time = current_time
    
    @torch.inference_mode()
    def __call__(self, attn: Any, hidden_states: Tensor,
                 encoder_hidden_states: Optional[Tensor] = None,
                 attention_mask: Optional[Tensor] = None,
                 temb: Optional[Tensor] = None,
                 *args: Any, **kwargs: Any) -> Tensor:
        
        # Cache key for similar operations
        cache_key = (hidden_states.shape, getattr(attention_mask, 'shape', None))
        
        if cache_key in self._attention_cache and not any([attn.spatial_norm, attn.group_norm, attention_mask]):
            return self._attention_cache[cache_key]
            
        result = self._process_attention(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        
        # Cache result for future use
        if len(self._attention_cache) > 100:  # Prevent unlimited growth
            self._attention_cache.clear()
        self._attention_cache[cache_key] = result
        
        return result
    
    def enable_spiral_attention(self, enabled: bool = True):
        """Enable or disable spiral attention pattern"""
        self.use_spiral = enabled
    
    def _process_attention(self, attn: Any, hidden_states: Tensor,
                          encoder_hidden_states: Optional[Tensor],
                          attention_mask: Optional[Tensor],
                          temb: Optional[Tensor]) -> Tensor:
        # Use memory efficient attention for large sequences
        if hidden_states.shape[1] > 1024:
            return self._process_large_attention(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        return self._process_simple_attention(attn, hidden_states, encoder_hidden_states, attention_mask, temb)

    def _process_simple_attention(self, attn: Any, hidden_states: torch.Tensor,
                                encoder_hidden_states: Optional[torch.Tensor],
                                attention_mask: Optional[torch.Tensor],
                                temb: Optional[torch.Tensor]) -> torch.Tensor:
        """Optimized path for simple attention without normalization or mask"""
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states if encoder_hidden_states is None else encoder_hidden_states)
        value = attn.to_v(hidden_states if encoder_hidden_states is None else encoder_hidden_states)

        # Use spiral attention if enabled
        if self.use_spiral:
            hidden_states = attention_with_spiral_bias(
                query, key, value,
                heads=attn.heads,
                mask=attention_mask,
                use_spiral=True
            )
        else:
            hidden_states = attention_function(
                query, key, value,
                heads=attn.heads,
                mask=attention_mask
            )

        hidden_states = attn.to_out[0](hidden_states)
        return hidden_states / attn.rescale_output_factor

    def _process_complex_attention(self, attn: Any, hidden_states: torch.Tensor,
                                 encoder_hidden_states: Optional[torch.Tensor],
                                 attention_mask: Optional[torch.Tensor],
                                 temb: Optional[torch.Tensor]) -> torch.Tensor:
        """Handle complex attention cases with normalization and masks"""
        residual = hidden_states

        if attn.spatial_norm:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        ndim = hidden_states.ndim
        if ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(b, h*w, c)

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, hidden_states.shape[1], hidden_states.shape[0])
            # Already handled repeating heads inside attention functions, so just pass as is.

        if attn.group_norm:
            hidden_states = attn.group_norm(hidden_states)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)
        value = attn.to_v(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)

        if attn.single_head_spatial:
            b, hw, c = hidden_states.shape
            h = attn.height
            w = attn.width
            q_ = query.view(b, hw, c).permute(0, 2, 1).view(b, c, h, w)
            k_ = key.view(b, hw, c).permute(0, 2, 1).view(b, c, h, w)
            v_ = value.view(b, hw, c).permute(0, 2, 1).view(b, c, h, w)

            hidden_states = attention_function_single_head_spatial(q_, k_, v_)
            hidden_states = hidden_states.view(b, c, h*w).permute(0, 2, 1)
        else:
            hidden_states = attention_function(query, key, value, heads=attn.heads, mask=attention_mask)

        hidden_states = attn.to_out[0](hidden_states)

        if ndim == 4:
            hidden_states = hidden_states.view(b, h, w, c).permute(0, 3, 1, 2)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor

    def _process_large_attention(self, attn: Any, hidden_states: Tensor,
                               encoder_hidden_states: Optional[Tensor],
                               attention_mask: Optional[Tensor],
                               temb: Optional[Tensor]) -> Tensor:
        """Handle large sequence attention using chunked computation"""
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)
        value = attn.to_v(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)
        
        hidden_states = attention_memory_efficient(
            query, key, value,
            heads=attn.heads,
            mask=attention_mask,
            chunk_size=512
        )
        
        hidden_states = attn.to_out[0](hidden_states)
        return hidden_states / attn.rescale_output_factor
