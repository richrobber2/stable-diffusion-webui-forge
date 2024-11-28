import math
import torch
import einops

from backend.args import args
from backend import memory_management
from backend.misc.sub_quadratic_attention import efficient_dot_product_attention


BROKEN_XFORMERS = False
if memory_management.xformers_enabled():
    import xformers
    import xformers.ops

    try:
        x_vers = xformers.__version__
        BROKEN_XFORMERS = x_vers.startswith("0.0.2") and not x_vers.startswith("0.0.20")
    except ImportError:
        pass


FORCE_UPCAST_ATTENTION_DTYPE = memory_management.force_upcast_attention_dtype()


def get_attn_precision(attn_precision=torch.float32):
    if args.disable_attention_upcast:
        return None
    if FORCE_UPCAST_ATTENTION_DTYPE is not None:
        return FORCE_UPCAST_ATTENTION_DTYPE
    return attn_precision


def exists(val):
    return val is not None


def get_optimal_chunk_size(total_memory, tensor_shape, dtype_size, heads):
    """Calculate optimal chunk size based on tensor dimensions and available memory"""
    # Account for attention matrix and intermediate results
    mem_per_element = dtype_size * 3  # For q, k, v
    total_elements = tensor_shape[0] * tensor_shape[1] * tensor_shape[2]
    base_memory = total_elements * mem_per_element

    # Dynamic scaling based on head count
    head_factor = math.log2(heads + 1) / 2
    chunk_size = min(4096, max(256, int(total_memory / (base_memory * head_factor))))

    # Round to nearest power of 2 for better memory alignment
    return 2 ** int(math.log2(chunk_size))


def attention_basic(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
    with torch.inference_mode(), torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
        attn_precision = get_attn_precision(attn_precision)

        if skip_reshape:
            b, _, _, dim_head = q.shape
        else:
            b, _, dim_head = q.shape
            dim_head //= heads

        # Dynamic scale based on head dimension and precision
        scale = 1.0 / math.sqrt(dim_head + (1e-8 if attn_precision == torch.float32 else 1e-5))

        # Direct reshape without contiguous for speed
        if skip_reshape:
            q = q.reshape(b * heads, -1, dim_head)
            k = k.reshape(b * heads, -1, dim_head)
            v = v.reshape(b * heads, -1, dim_head)
        else:
            q = q.view(b, -1, heads, dim_head).permute(0, 2, 1, 3).reshape(b * heads, -1, dim_head)
            k = k.view(b, -1, heads, dim_head).permute(0, 2, 1, 3).reshape(b * heads, -1, dim_head)
            v = v.view(b, -1, heads, dim_head).permute(0, 2, 1, 3).reshape(b * heads, -1, dim_head)

        # Optimize computation for different precisions
        if attn_precision == torch.float32:
            # Use flash attention when possible
            try:
                out = torch.nn.functional.scaled_dot_product_attention(
                    q.float(), k.float(), v.float(),
                    attn_mask=mask, dropout_p=0.0, scale=scale,
                    is_causal=False
                )
                out.to_(q.dtype)
                out.view_(b, -1, heads * dim_head)
                return out
            except:
                pass

        # Optimized attention calculation
        if attn_precision == torch.float32:
            sim = torch.baddbmm(
                torch.empty(q.shape[0], q.shape[1], k.shape[1], dtype=torch.float32, device=q.device),
                q.float(),
                k.float().transpose(-2, -1),
                beta=0.0,
                alpha=scale
            )
        else:
            sim = torch.baddbmm(
                torch.empty(q.shape[0], q.shape[1], k.shape[1], dtype=q.dtype, device=q.device),
                q,
                k.transpose(-2, -1),
                beta=0.0,
                alpha=scale
            )

        # Optimized mask application
        if exists(mask):
            if mask.dtype == torch.bool:
                sim.masked_fill_(~mask.view(b * heads, 1, -1), -torch.finfo(sim.dtype).max)
            else:
                sim.add_(mask.view(b * heads, -1, mask.shape[-1]))

        # Fast softmax
        sim = sim.softmax(dim=-1)

        # Optimized output calculation
        hidden_states = torch.bmm(sim.to(v.dtype), v)
        hidden_states = hidden_states.view(b, heads, -1, dim_head).permute(0, 2, 1, 3).reshape(b, -1, heads * dim_head)

        return hidden_states


def attention_sub_quad(query, key, value, heads, mask=None, attn_precision=None, skip_reshape=False):
    attn_precision = get_attn_precision(attn_precision)

    if skip_reshape:
        b, _, _, dim_head = query.shape
    else:
        b, _, dim_head = query.shape
        dim_head //= heads

    scale = dim_head ** -0.5

    if skip_reshape:
        query = query.reshape(b * heads, -1, dim_head)
        value = value.reshape(b * heads, -1, dim_head)
        key = key.reshape(b * heads, -1, dim_head).movedim(1, 2)
    else:
        query = query.unsqueeze(3).reshape(b, -1, heads, dim_head).permute(0, 2, 1, 3).reshape(b * heads, -1, dim_head)
        value = value.unsqueeze(3).reshape(b, -1, heads, dim_head).permute(0, 2, 1, 3).reshape(b * heads, -1, dim_head)
        key = key.unsqueeze(3).reshape(b, -1, heads, dim_head).permute(0, 2, 3, 1).reshape(b * heads, dim_head, -1)

    dtype = query.dtype
    upcast_attention = attn_precision == torch.float32 and query.dtype != torch.float32
    if upcast_attention:
        bytes_per_token = torch.finfo(torch.float32).bits // 8
    else:
        bytes_per_token = torch.finfo(query.dtype).bits // 8
    batch_x_heads, q_tokens, _ = query.shape
    _, _, k_tokens = key.shape

    mem_free_total, mem_free_torch = memory_management.get_free_memory(query.device, True)

    kv_chunk_size_min = None
    kv_chunk_size = None
    query_chunk_size = None

    for x in [4096, 2048, 1024, 512, 256]:
        count = mem_free_total / (batch_x_heads * bytes_per_token * x * 4.0)
        if count >= k_tokens:
            kv_chunk_size = k_tokens
            query_chunk_size = x
            break

    if query_chunk_size is None:
        query_chunk_size = 512

    if mask is not None:
        if len(mask.shape) == 2:
            bs = 1
        else:
            bs = mask.shape[0]
        mask = mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1]).expand(b, heads, -1, -1).reshape(-1, mask.shape[-2], mask.shape[-1])

    hidden_states = efficient_dot_product_attention(
        query,
        key,
        value,
        query_chunk_size=query_chunk_size,
        kv_chunk_size=kv_chunk_size,
        kv_chunk_size_min=kv_chunk_size_min,
        use_checkpoint=False,
        upcast_attention=upcast_attention,
        mask=mask,
    )

    hidden_states = hidden_states.to(dtype)

    hidden_states = hidden_states.unflatten(0, (-1, heads)).transpose(1, 2).flatten(start_dim=2)
    return hidden_states


def attention_split(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
    with torch.inference_mode():
        attn_precision = get_attn_precision(attn_precision)

        if skip_reshape:
            b, _, _, dim_head = q.shape
        else:
            b, _, dim_head = q.shape
            dim_head //= heads

        scale = dim_head ** -0.5

        h = heads
        if skip_reshape:
            q, k, v = (t.reshape(b * heads, -1, dim_head) for t in (q, k, v))
        else:
            q, k, v = (
                t.unsqueeze(3)
                .reshape(b, -1, heads, dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b * heads, -1, dim_head)
                .contiguous()
                for t in (q, k, v)
            )

        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)

        mem_free_total = memory_management.get_free_memory(q.device)

        if attn_precision == torch.float32:
            element_size = 4
            upcast = True
        else:
            element_size = q.element_size()
            upcast = False

        gb = 1024 ** 3
        tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * element_size
        # Improved memory requirement estimation
        modifier = 2.5 if element_size < 4 else 3
        mem_required = tensor_size * modifier
        steps = 1

        if mem_required > mem_free_total:
            steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))
            # print(f"Expected tensor size:{tensor_size/gb:0.1f}GB, cuda free:{mem_free_cuda/gb:0.1f}GB "
            #      f"torch free:{mem_free_torch/gb:0.1f} total:{mem_free_total/gb:0.1f} steps:{steps}")

        if steps > 64:
            max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
            raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                               f'Need: {mem_required / 64 / gb:0.1f}GB free, Have:{mem_free_total / gb:0.1f}GB free')

        if mask is not None:
            if len(mask.shape) == 2:
                bs = 1
            else:
                bs = mask.shape[0]
            mask = mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1]).expand(b, heads, -1, -1).reshape(-1, mask.shape[-2], mask.shape[-1])

        # Smart memory management
        mem_free_total = memory_management.get_free_memory(q.device)
        chunk_size = get_optimal_chunk_size(
            mem_free_total,
            (q.shape[0], q.shape[1], k.shape[1]),
            q.element_size(),
            heads
        )
        
        # Adaptive step calculation
        steps = max(1, (q.shape[1] + chunk_size - 1) // chunk_size)
        
        # Pre-allocate buffers for efficiency
        if steps > 1:
            s1_buffer = torch.empty(
                (q.shape[0], chunk_size, k.shape[1]),
                dtype=torch.float32 if upcast else q.dtype,
                device=q.device
            )

        # print("steps", steps, mem_required, mem_free_total, modifier, q.element_size(), tensor_size)
        first_op_done = False
        cleared_cache = False
        while True:
            try:
                slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
                for i in range(0, q.shape[1], slice_size):
                    end = i + slice_size
                    if upcast:
                        with torch.autocast(enabled=False, device_type='cuda'):
                            s1 = torch.einsum('b i d, b j d -> b i j', q[:, i:end].float(), k.float()) * scale
                    else:
                        s1 = torch.einsum('b i d, b j d -> b i j', q[:, i:end], k) * scale

                    if mask is not None:
                        if len(mask.shape) == 2:
                            s1 += mask[i:end]
                        else:
                            s1 += mask[:, i:end]

                    s2 = s1.softmax(dim=-1).to(v.dtype)
                    del s1
                    first_op_done = True

                    r1[:, i:end].add_(torch.einsum('b i j, b j d -> b i d', s2, v))
                    del s2
                break
            except memory_management.OOM_EXCEPTION as e:
                if not first_op_done:
                    memory_management.soft_empty_cache(True)
                    if not cleared_cache:
                        cleared_cache = True
                        print("out of memory error, emptying cache and trying again")
                        continue
                    steps *= 2
                    if steps > 64:
                        raise e
                    print("out of memory error, increasing steps and trying again {}".format(steps))
                else:
                    raise e

        del q, k, v

        r1 = (
            r1.unsqueeze(0)
            .reshape(b, heads, -1, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, -1, heads * dim_head)
        )
        return r1


def attention_xformers(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads

    if BROKEN_XFORMERS and b * heads > 65535:
        return attention_pytorch(q, k, v, heads, mask, skip_reshape=skip_reshape)

    if skip_reshape:
        q, k, v = (t.reshape(b * heads, -1, dim_head) for t in (q, k, v))
    else:
        q, k, v = (t.reshape(b, -1, heads, dim_head) for t in (q, k, v))

    if mask is not None:
        pad = 8 - q.shape[1] % 8
        mask_out = torch.empty([q.shape[0], q.shape[1], q.shape[1] + pad], dtype=q.dtype, device=q.device)
        mask_out[:, :, :mask.shape[-1]] = mask
        mask = mask_out[:, :, :mask.shape[-1]]

    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask)

    if skip_reshape:
        out = (
            out.unsqueeze(0)
            .reshape(b, heads, -1, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, -1, heads * dim_head)
        )
    else:
        out = (
            out.reshape(b, -1, heads * dim_head)
        )

    return out


def attention_pytorch(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    out = (
        out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    )
    return out


def slice_attention_single_head_spatial(q, k, v):
    with torch.inference_mode():
        # Smart tensor allocation
        dtype = torch.float32 if q.dtype == torch.float16 else q.dtype
        shape = (q.shape[0], q.shape[1], k.shape[2])

        r1 = torch.empty((v.shape[0], v.shape[1], q.shape[1]), dtype=q.dtype, device=q.device)

        # Optimized scale calculation with dynamic precision
        scale = 1.0 / math.sqrt(q.shape[-1] + (1e-5 if dtype == torch.float32 else 1e-4))

        # Optimized memory calculation
        mem_free_total = memory_management.get_free_memory(q.device)
        tensor_size = q.shape[0] * q.shape[1] * k.shape[2] * q.element_size()
        steps = max(1, min(128, 2 ** math.ceil(math.log2(tensor_size * 3.0 / mem_free_total))))

        slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]

        try:
            for i in range(0, q.shape[1], slice_size):
                end = i + slice_size
                # Optimized matrix multiplication
                s1 = torch.baddbmm(
                    torch.empty(q.shape[0], end-i, k.shape[2], dtype=q.dtype, device=q.device),
                    q[:, i:end],
                    k,
                    beta=0.0,
                    alpha=scale
                )
                s1.softmax_(dim=-1)
                r1[:, :, i:end].copy_(torch.bmm(v, s1.transpose(-2, -1)))
                del s1

            return r1

        except memory_management.OOM_EXCEPTION as e:
            memory_management.soft_empty_cache(True)
            if steps < 128:
                steps *= 2
                print(f"Retrying with {steps} steps")
                return slice_attention_single_head_spatial(q, k, v)
            raise e


def normal_attention_single_head_spatial(q, k, v):
    # Add shape validation
    assert q.shape == k.shape == v.shape, "Input tensor shapes must match"
    assert len(q.shape) == 4, "Input must be 4D tensor (batch, channels, height, width)"
    # compute attention
    b, c, h, w = q.shape

    q = q.reshape(b, c, h * w)
    q = q.permute(0, 2, 1)  # b,hw,c
    k = k.reshape(b, c, h * w)  # b,c,hw
    v = v.reshape(b, c, h * w)

    r1 = slice_attention_single_head_spatial(q, k, v)
    h_ = r1.reshape(b, c, h, w)
    del r1
    return h_


def xformers_attention_single_head_spatial(q, k, v):
    # compute attention
    B, C, H, W = q.shape
    try:
        q, k, v = (t.view(B, C, -1).transpose(1, 2).contiguous() for t in (q, k, v))
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)
        out = out.transpose(1, 2).reshape(B, C, H, W)
    except (NotImplementedError, RuntimeError):
        out = slice_attention_single_head_spatial(q.view(B, -1, C), k.view(B, -1, C).transpose(1, 2),
                                                  v.view(B, -1, C).transpose(1, 2)).reshape(B, C, H, W)
    return out


def pytorch_attention_single_head_spatial(q, k, v):
    # compute attention
    B, C, H, W = q.shape
    try:
        q, k, v = (t.view(B, 1, C, -1).transpose(2, 3).contiguous() for t in (q, k, v))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = out.transpose(2, 3).reshape(B, C, H, W)
    except memory_management.OOM_EXCEPTION:
        print("scaled_dot_product_attention OOMed: switched to slice attention")
        out = slice_attention_single_head_spatial(q.view(B, -1, C), k.view(B, -1, C).transpose(1, 2),
                                                  v.view(B, -1, C).transpose(1, 2)).reshape(B, C, H, W)
    return out


if memory_management.xformers_enabled():
    print("Using xformers cross attention")
    attention_function = attention_xformers
elif memory_management.pytorch_attention_enabled():
    print("Using pytorch cross attention")
    attention_function = attention_pytorch
elif args.attention_split:
    print("Using split optimization for cross attention")
    attention_function = attention_split
else:
    print("Using sub quadratic optimization for cross attention")
    attention_function = attention_sub_quad

if memory_management.xformers_enabled_vae():
    print("Using xformers attention for VAE")
    attention_function_single_head_spatial = xformers_attention_single_head_spatial
elif memory_management.pytorch_attention_enabled():
    print("Using pytorch attention for VAE")
    attention_function_single_head_spatial = pytorch_attention_single_head_spatial
else:
    print("Using split attention for VAE")
    attention_function_single_head_spatial = normal_attention_single_head_spatial


class AttentionProcessorForge:
    def __call__(self, attn, hidden_states, encoder_hidden_states, attention_mask=None, temb=None, *args, **kwargs):
        with torch.inference_mode():
            # Fast path for no spatial norm and no group norm
            if attn.spatial_norm is None and attn.group_norm is None and attention_mask is None:
                query = attn.to_q(hidden_states)
                key = attn.to_k(hidden_states if encoder_hidden_states is None else encoder_hidden_states)
                value = attn.to_v(hidden_states if encoder_hidden_states is None else encoder_hidden_states)

                hidden_states = attention_function(query, key, value, heads=attn.heads)
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)
                return hidden_states / attn.rescale_output_factor

            residual = hidden_states

            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = attn.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            hidden_states = attention_function(query, key, value, heads=attn.heads, mask=attention_mask)

            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states
