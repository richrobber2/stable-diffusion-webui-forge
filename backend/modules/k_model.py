import torch

from backend import memory_management, attention
from backend.modules.k_prediction import k_prediction_from_diffusers_scheduler


class KModel(torch.nn.Module):
    def __init__(self, model, diffusers_scheduler, k_predictor=None, config=None):
        super().__init__()

        self.config = config

        self.storage_dtype = model.storage_dtype
        self.computation_dtype = model.computation_dtype

        print(f'K-Model Created: {dict(storage_dtype=self.storage_dtype, computation_dtype=self.computation_dtype)}')

        self.diffusion_model = model

        if k_predictor is None:
            self.predictor = k_prediction_from_diffusers_scheduler(diffusers_scheduler)
        else:
            self.predictor = k_predictor

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kwargs):
        if transformer_options is None:
            transformer_options = {}
        sigma = t
        xc = self.predictor.calculate_input(sigma, x)
        if c_concat is not None:
            xc = torch.cat([xc] + [c_concat], dim=1)

        context = c_crossattn
        dtype = self.computation_dtype

        xc = xc.to(dtype=dtype, non_blocking=True)
        t = self.predictor.timestep(t).float()
        context = context.to(dtype=dtype, non_blocking=True)

        extra_conds = {}
        for o, extra in kwargs.items():
            if hasattr(extra, "dtype") and extra.dtype not in [torch.int, torch.long]:
                extra = extra.to(dtype=dtype, non_blocking=True)
            extra_conds[o] = extra

        model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)
        model_output = model_output.float()  # Changed from in-place float_() to regular float()
        return self.predictor.calculate_denoised(sigma, model_output, x)

    def memory_required(self, input_shape):
        pixel_area = input_shape[0] * input_shape[2] * input_shape[3]  # batch * height * width
        base_memory_per_pixel = memory_management.dtype_size(self.computation_dtype)

        # Attention memory scaling
        if attention.attention_function in [attention.attention_pytorch, attention.attention_xformers]:
            attention_scale = 1.28  # Optimized attention implementations
        else:
            attention_scale = 1.65  # Standard attention implementation
            if attention.get_attn_precision() == torch.float32:
                base_memory_per_pixel = 4  # Override for float32 precision

        # Total memory calculation (base_memory * scaling * feature_dimension)
        return pixel_area * base_memory_per_pixel * attention_scale * 16384
