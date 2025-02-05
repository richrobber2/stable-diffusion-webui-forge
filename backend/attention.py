import time
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
from functools import lru_cache
from abc import ABC, abstractmethod
from enum import Enum, auto
import logging

from backend.args import args
from backend import memory_management
from backend.misc.sub_quadratic_attention import efficient_dot_product_attention
# Configure logging
logger = logging.getLogger(__name__)

# Add xformers import with proper error handling
try:
    import xformers
    import xformers.ops
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    print("xformers not available - using fallback attention implementation")

class AttentionBackend(Enum):
    """Available attention computation backends"""
    XFORMERS = auto()
    PYTORCH = auto()
    SPLIT = auto()
    SUB_QUADRATIC = auto()

@dataclass(frozen=True)
class BackendConfig:
    """Immutable attention backend configuration"""
    xformers_enabled: bool
    pytorch_attention_enabled: bool
    force_upcast_attention: Optional[torch.dtype]
    broken_xformers: bool = False
    
    @classmethod
    def create(cls) -> 'BackendConfig':
        """Create backend configuration from environment"""
        xformers_ok = memory_management.xformers_enabled() and XFORMERS_AVAILABLE
        pytorch_ok = memory_management.pytorch_attention_enabled()
        upcast = memory_management.force_upcast_attention_dtype()
        
        # Check xformers version
        broken = False
        if xformers_ok:
            try:
                broken = xformers.__version__.startswith("0.0.2") and not xformers.__version__.startswith("0.0.20")
            except:
                xformers_ok = False
                
        return cls(
            xformers_enabled=xformers_ok,
            pytorch_attention_enabled=pytorch_ok,
            force_upcast_attention=upcast,
            broken_xformers=broken
        )

class CacheManager:
    """Manages attention caches with memory awareness"""
    
    def __init__(self, max_memory_fraction: float = 0.3, cache_lifetime: int = 300):
        self.max_memory_fraction = max_memory_fraction
        self.cache_lifetime = cache_lifetime
        self.caches: Dict[str, MemoryAwareCache] = {}
        
    def get_cache(self, name: str) -> 'MemoryAwareCache':
        """Get or create a cache by name"""
        if name not in self.caches:
            self.caches[name] = MemoryAwareCache(
                self.max_memory_fraction,
                self.cache_lifetime
            )
        return self.caches[name]
        
    def clear_all(self):
        """Clear all caches"""
        logger.info("Clearing all attention caches")
        for name, cache in self.caches.items():
            cache.clear()
            logger.debug(f"Cache '{name}' cleared.")
            
    def check_memory_pressure(self):
        """Clear caches if memory pressure is high"""
        mem_free_total, mem_free_torch = memory_management.get_free_memory(
            torch.device('cuda'), True)
        if mem_free_torch / mem_free_total < 0.2:  # Clear if less than 20% free
            logger.warning("High memory pressure detected. Clearing caches.")
            self.clear_all()

class MemoryAwareCache:
    """Thread-safe cache with memory monitoring"""
    
    def __init__(self, max_memory_fraction: float = 0.3, cache_lifetime: int = 300):
        self.max_memory_fraction = max_memory_fraction
        self.cache_lifetime = cache_lifetime
        self.cache = weakref.WeakValueDictionary()
        self.last_access_times: Dict[Tuple, float] = {}
        self.lock = threading.Lock()
        
    def get(self, key: Tuple) -> Optional[Tensor]:
        """Get cached value with timestamp update"""
        with self.lock:
            value = self.cache.get(key)
            if value is not None:
                self.last_access_times[key] = time.time()
            return value
            
    def set(self, key: Tuple, value: Tensor) -> None:
        """Set cache value with memory and lifetime checks"""
        with self.lock:
            current_time = time.time()
            # Clear old entries
            old_keys = [k for k, t in self.last_access_times.items() 
                       if current_time - t > self.cache_lifetime]
            for k in old_keys:
                logger.debug(f"Removing expired cache key: {k}")
                self.cache.pop(k, None)
                self.last_access_times.pop(k, None)
                
            # Check memory before adding
            if self._check_memory_usage():
                self.cache[key] = value
                self.last_access_times[key] = current_time
            else:
                logger.warning("Memory threshold exceeded, skipping cache set.")
            
    def clear(self) -> None:
        """Clear cache and timestamps"""
        with self.lock:
            self.cache.clear()
            self.last_access_times.clear()
            
    def _check_memory_usage(self) -> bool:
        """Check if memory usage is within limits"""
        mem_free_total, mem_free_torch = memory_management.get_free_memory(
            torch.device('cuda'), True)
        return mem_free_torch / mem_free_total > self.max_memory_fraction

class AttentionError(Exception):
    """Base exception for attention-related errors"""
    pass

class MemoryError(AttentionError):
    """Raised when running into memory issues"""
    pass

class InvalidConfigurationError(AttentionError):
    """Raised when attention configuration is invalid"""
    pass

class SpatialAttentionError(AttentionError):
    """Raised when spatial attention computation fails"""
    pass

# Initialize global configuration and cache manager
backend_config = BackendConfig.create()
cache_manager = CacheManager()

# Define attention strategy protocol
class AttentionStrategy(Protocol):
    def __call__(self, q: Tensor, k: Tensor, v: Tensor,
                 heads: int, mask: Optional[Tensor] = None,
                 attn_precision: Optional[torch.dtype] = None) -> Tensor:
        ...

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
class AttentionBase(ABC):
    """Base class for all attention implementations"""
    def __init__(self, config: AttentionConfig):
        self.config = config
        self._cache = MemoryAwareCache()
        self._metrics = AttentionMetrics() if config.enable_metrics else None

    @abstractmethod
    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                heads: int, mask: Optional[Tensor] = None) -> Tensor:
        pass

    def __call__(self, q: Tensor, k: Tensor, v: Tensor,
                 heads: int, mask: Optional[Tensor] = None) -> Tensor:
        cache_key = (q.shape, k.shape, v.shape, heads, 
                    getattr(mask, 'shape', None))
        
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        start = time.time()
        try:
            result = self.forward(q, k, v, heads, mask)
            self._cache.set(cache_key, result)
            
            if self._metrics:
                self._metrics.record_call(time.time() - start, True)
            
            return result
        except Exception as e:
            if isinstance(e, torch.cuda.OutOfMemoryError):
                self._cache.clear()
                torch.cuda.empty_cache()
                return self.forward(q, k, v, heads, mask)
            raise AttentionError(f"Attention computation failed: {str(e)}") from e

class LazyAttentionLoader:
    """Lazy loader for attention implementations to reduce memory usage"""
    _loaded_implementation = None
    _backend = None

    @classmethod
    def get_implementation(cls):
        """Get or load the appropriate attention implementation"""
        current_backend = cls._get_current_backend()
        
        # Return cached implementation if backend hasn't changed
        if cls._loaded_implementation and cls._backend == current_backend:
            logger.debug("Using cached attention implementation.")
            return cls._loaded_implementation

        logger.info("Loading new attention implementation.")
        # Clear old implementation
        cls._loaded_implementation = None
        cls._backend = None

        # Load new implementation based on backend
        if backend_config.xformers_enabled and XFORMERS_AVAILABLE:
            cls._loaded_implementation = XFormersAttention
            logger.info("Selected xformers attention implementation.")
        elif backend_config.pytorch_attention_enabled:
            cls._loaded_implementation = PyTorchAttention
            logger.info("Selected PyTorch attention implementation.")
        elif args.attention_split:
            cls._loaded_implementation = SplitAttention
            logger.info("Selected split attention implementation.")
        else:
            cls._loaded_implementation = SubQuadraticAttention
            logger.info("Selected sub-quadratic attention implementation.")

        cls._backend = current_backend
        return cls._loaded_implementation

    @staticmethod
    def _get_current_backend() -> str:
        """Get current backend configuration as string"""
        if backend_config.xformers_enabled and XFORMERS_AVAILABLE:
            return "xformers"
        elif backend_config.pytorch_attention_enabled:
            return "pytorch"
        elif args.attention_split:
            return "split"
        else:
            return "subquadratic"

# Modify AttentionFactory to use lazy loader
class AttentionFactory:
    @staticmethod
    def create(config: AttentionConfig) -> AttentionBase:
        implementation = LazyAttentionLoader.get_implementation()
        return implementation(config)

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
    return backend_config.force_upcast_attention or attn_precision

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

@dataclass
class SpiralConfig:
    """Configuration for spiral attention pattern"""
    start_radius: float = 0.0
    growth_rate: float = 0.1
    decay_factor: float = 5.0
    min_value: float = -100.0
    use_exponential: bool = True
    cache_size: int = 32

@torch.jit.script
def create_spiral_pattern(height: int, width: int, config: Dict[str, float]) -> Tensor:
    """JIT-optimized spiral pattern creation"""
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, height),
        torch.linspace(-1, 1, width),
        indexing='ij'
    )
    
    r = torch.sqrt(x_coords**2 + y_coords**2)
    theta = torch.atan2(y_coords, x_coords)
    theta_pos = theta + math.pi
    
    spiral_r = config["start_radius"] + config["growth_rate"] * theta_pos
    spiral_dist = torch.abs(r - spiral_r)
    
    if config["use_exponential"]:
        spiral_score = torch.exp(-spiral_dist * config["decay_factor"])
    else:
        spiral_score = 1.0 / (1.0 + spiral_dist * config["decay_factor"])
    
    spiral_score = spiral_score.clamp(min=math.exp(config["min_value"]))
    return spiral_score

class SpiralAttention:
    """Efficient spiral attention implementation with caching"""
    
    def __init__(self, config: Optional[SpiralConfig] = None):
        self.config = config or SpiralConfig()
        self._cache = LRUCache(self.config.cache_size)
        
    @staticmethod
    def _make_cache_key(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tuple:
        return (height, width, str(device), str(dtype))
        
    def get_spiral_bias(self, height: int, width: int, 
                       heads: int, device: torch.device, 
                       dtype: torch.dtype) -> Tensor:
        """Get cached or compute spiral attention bias"""
        cache_key = self._make_cache_key(height, width, device, dtype)
        
        if cache_key in self._cache:
            bias = self._cache[cache_key]
        else:
            config_dict = {
                "start_radius": self.config.start_radius,
                "growth_rate": self.config.growth_rate,
                "decay_factor": self.config.decay_factor,
                "min_value": self.config.min_value,
                "use_exponential": float(self.config.use_exponential)  # Convert bool to float for JIT
            }
            
            with torch.device(device):
                bias = create_spiral_pattern(height, width, config_dict)
                flat_bias = bias.view(-1)
                bias = (flat_bias.unsqueeze(0) + flat_bias.unsqueeze(1)) / 2.0
                bias = bias.to(dtype=dtype)
                self._cache[cache_key] = bias
                
        # Expand for multiple heads
        return bias.unsqueeze(0).expand(heads, -1, -1)

def attention_with_spiral_bias(q: Tensor, k: Tensor, v: Tensor,
                             heads: int, mask: Optional[Tensor] = None,
                             spiral_config: Optional[SpiralConfig] = None,
                             skip_reshape: bool = False) -> Tensor:
    """Enhanced attention with configurable spiral bias"""
    b, seq_len, _ = q.shape
    height = width = int(math.sqrt(seq_len))
    
    if height * width != seq_len:
        # Fall back to regular attention if input is not square
        return attention_basic(q, k, v, heads, mask, skip_reshape=skip_reshape)
    
    spiral = SpiralAttention(spiral_config)
    spiral_bias = spiral.get_spiral_bias(height, width, heads, q.device, q.dtype)
    
    # Regular attention computation
    scale = get_scale(q.shape[-1], None, heads)
    q = simple_rearrange_qkv(q, b, heads, q.shape[-1] // heads, skip_reshape)
    k = simple_rearrange_qkv(k, b, heads, k.shape[-1] // heads, skip_reshape)
    v = simple_rearrange_qkv(v, b, heads, v.shape[-1] // heads, skip_reshape)
    
    sim = torch.bmm(q, k.transpose(-2, -1))
    sim *= scale
    
    # Add spiral bias (in log space)
    sim = sim + spiral_bias.log()
    
    # Apply mask if provided
    if mask is not None:
        sim = sim + mask
    
    # Compute attention and reshape
    attn = torch.softmax(sim, dim=-1)
    out = torch.bmm(attn, v)
    
    return out.view(b, heads, seq_len, v.shape[-1] // heads).transpose(1, 2).reshape(b, seq_len, -1)

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

def attention_basic(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                   heads: int, mask: Optional[torch.Tensor] = None, 
                   attn_precision: torch.dtype = torch.float32,
                   skip_reshape: bool = False,
                   use_spiral: bool = True) -> torch.Tensor:
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
        sim = q.bmm(k.transpose(-2, -1))
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
    """Optimized xformers attention implementation with fallback"""
    if not XFORMERS_AVAILABLE:
        return attention_pytorch(q, k, v, heads, mask, attn_precision, skip_reshape)
        
    if skip_reshape:
        b, seq, _, dim_head = q.shape
    else:
        b, seq, total_dim = q.shape
        dim_head = total_dim // heads

    if backend_config.broken_xformers and b * heads > 65535:
        return attention_pytorch(q, k, v, heads, mask, skip_reshape=skip_reshape)

    try:
        q = simple_rearrange_qkv(q, b, heads, dim_head, skip_reshape)
        k = simple_rearrange_qkv(k, b, heads, dim_head, skip_reshape)
        v = simple_rearrange_qkv(v, b, heads, dim_head, skip_reshape)

        if mask is not None:
            mask = get_mask_for_heads(mask.shape, heads, mask.device, mask.dtype)

        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask)
        return out.view(b, heads, seq, dim_head).transpose(1, 2).reshape(b, seq, heads * dim_head)
    except Exception as e:
        logger.warning(f"xformers attention failed: {str(e)}, falling back to PyTorch attention")
        return attention_pytorch(q, k, v, heads, mask, attn_precision, skip_reshape)

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
    """Memory-efficient spatial attention slicing with smart buffering"""
    with torch.inference_mode():
        dtype = torch.float32 if q.dtype == torch.float16 else q.dtype
        b, hw, c = q.shape
        scale = 1.0 / math.sqrt(q.shape[-1] + (1e-5 if dtype == torch.float32 else 1e-4))

        # Pre-allocate output and intermediate buffers
        r1 = torch.zeros((v.shape[0], v.shape[1], q.shape[1]), dtype=q.dtype, device=q.device)
        
        # Calculate optimal slice size based on available memory
        mem_free_total, mem_free_torch = memory_management.get_free_memory(q.device, True)
        tensor_size = q.numel() * k.shape[2] * q.element_size()
        steps = max(1, min(128, 2 ** math.ceil(math.log2(tensor_size * 3.0 / mem_free_total))))
        slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]

        # Allocate a reusable buffer for intermediate results
        # This avoids creating new tensors in the loop
        buffer_size = (b, slice_size, k.shape[1])
        intermediate_buffer = torch.empty(buffer_size, dtype=q.dtype, device=q.device)
        softmax_buffer = torch.empty_like(intermediate_buffer)

        for i in range(0, q.shape[1], slice_size):
            end = i + slice_size
            current_slice = min(slice_size, q.shape[1] - i)
            
            # Use buffers with correct size view
            if current_slice != slice_size:
                curr_buffer = intermediate_buffer[:, :current_slice]
                curr_softmax = softmax_buffer[:, :current_slice]
            else:
                curr_buffer = intermediate_buffer
                curr_softmax = softmax_buffer

            # Calculate attention scores using buffer
            torch.bmm(q[:, i:end], k, out=curr_buffer)
            curr_buffer *= scale
            
            # Softmax in-place using buffer
            torch.softmax(curr_buffer, dim=-1, out=curr_softmax)
            
            # Final multiplication directly into output
            torch.bmm(v, curr_softmax.transpose(-2, -1), out=r1[:, :, i:end])

        return r1

def normal_attention_single_head_spatial(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Standard spatial attention implementation"""
    b, c, h, w = q.shape
    q_ = q.view(b, c, h*w).transpose(1, 2)
    k_ = k.view(b, c, h*w)
    v_ = v.view(b, c, h*w)

    r1 = slice_attention_single_head_spatial(q_, k_, v_)
    return r1.view(b, c, h, w)

def safe_spatial_attention(func: Callable) -> Callable:
    """Decorator for safe spatial attention computation"""
    def wrapper(*args, **kwargs) -> Tensor:
        try:
            return func(*args, **kwargs)
        except (RuntimeError, memory_management.OOM_EXCEPTION) as e:
            logger.warning(f"Spatial attention failed: {e}, falling back to slice attention")
            return slice_attention_single_head_spatial(*args, **kwargs)
        except Exception as e:
            raise SpatialAttentionError(f"Unexpected error in spatial attention: {str(e)}") from e
    return wrapper

@safe_spatial_attention
def xformers_attention_single_head_spatial(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Safer xformers spatial attention implementation"""
    B, C, H, W = q.shape
    q_ = q.view(B, C, H*W).transpose(1, 2).contiguous()
    k_ = k.view(B, C, H*W).transpose(1, 2).contiguous()
    v_ = v.view(B, C, H*W).transpose(1, 2).contiguous()
    out = xformers.ops.memory_efficient_attention(q_, k_, v_, attn_bias=None)
    return out.transpose(1, 2).view(B, C, H, W)

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
if backend_config.xformers_enabled:
    attention_function = attention_xformers
    attention_function_single_head_spatial = xformers_attention_single_head_spatial
elif backend_config.pytorch_attention_enabled:
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
            chunk_output = attention_pytorch(chunk_q, k, v, heads, chunk_mask, attn_precision)
            output[:, i:i + chunk_size] = chunk_output
            
        return output

class AttentionProcessorForge:
    """Memory-optimized attention processor with better error handling"""
    
    def __init__(self, config: Optional[AttentionConfig] = None):
        self.config = config or AttentionConfig(
            dim_head=64,
            attn_precision=torch.float32,
            heads=8,
            device=torch.device('cuda'),
            dtype=torch.float32
        )
        self.attention = AttentionFactory.create(self.config)
        self.use_spiral = True
        
    def _process_attention_safely(self, *args, **kwargs) -> Tensor:
        """Process attention with proper error handling"""
        try:
            return self._process_attention(*args, **kwargs)
        except MemoryError as e:
            logger.warning(f"Memory error in attention processing: {e}")
            # Try to free memory and retry
            cache_manager.clear_all()
            memory_management.soft_empty_cache(force=True)
            return self._process_attention(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in attention processing: {e}")
            raise AttentionError(f"Attention processing failed: {str(e)}") from e
            
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
        value = attn.to_v(hidden_states if encoder_hidden_states is not None else hidden_states)
        
        hidden_states = attention_memory_efficient(
            query, key, value,
            heads=attn.heads,
            mask=attention_mask,
            chunk_size=512
        )
        
        hidden_states = attn.to_out[0](hidden_states)
        return hidden_states / attn.rescale_output_factor

# Only define base classes and interfaces here
# ...existing AttentionBase, AttentionConfig, etc...

# Move actual implementations to separate files
from .attention_impl import (
    XFormersAttention, 
    PyTorchAttention,
    SplitAttention,
    SubQuadraticAttention
)

# Export only what's needed
__all__ = [
    'AttentionFactory',
    'AttentionBase',
    'AttentionConfig',
    'BackendConfig',
    'AttentionProcessorForge'
]
