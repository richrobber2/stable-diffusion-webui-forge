from typing import Optional, List, Dict, Any, Generator, Callable, Union
from dataclasses import dataclass
import copy
import itertools
import torch

from backend.modules.k_model import KModel
from backend.patcher.base import ModelPatcher

@dataclass
class PatchConfig:
    filename: str
    patches: Dict[str, List[tuple]]
    strength_patch: float
    strength_model: float

class UnetPatcher(ModelPatcher):
    @classmethod
    def from_model(cls, model, diffusers_scheduler, config, k_predictor=None):
        model = KModel(model=model, diffusers_scheduler=diffusers_scheduler, k_predictor=k_predictor, config=config)
        return UnetPatcher(
            model,
            load_device=model.diffusion_model.load_device,
            offload_device=model.diffusion_model.offload_device,
            current_device=model.diffusion_model.initial_device
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.controlnet_linked_list = None
        self.extra_preserved_memory_during_sampling = 0
        self.extra_model_patchers_during_sampling = []
        self.extra_concat_condition = None

    def clone(self) -> 'UnetPatcher':
        n = UnetPatcher(self.model, self.load_device, self.offload_device, self.size, self.current_device)
        n.lora_patches = self.lora_patches.copy()
        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.controlnet_linked_list = self.controlnet_linked_list
        n.extra_preserved_memory_during_sampling = self.extra_preserved_memory_during_sampling
        n.extra_model_patchers_during_sampling = self.extra_model_patchers_during_sampling.copy()
        n.extra_concat_condition = self.extra_concat_condition
        return n

    def add_extra_preserved_memory_during_sampling(self, memory_in_bytes: int):
        # Use this to ask Forge to preserve a certain amount of memory during sampling.
        # If GPU VRAM is 8 GB, and memory_in_bytes is 2GB, i.e., memory_in_bytes = 2 * 1024 * 1024 * 1024
        # Then the sampling will always use less than 6GB memory by dynamically offload modules to CPU RAM.
        # You can estimate this using memory_management.module_size(any_pytorch_model) to get size of any pytorch models.
        if isinstance(memory_in_bytes, int) and memory_in_bytes > 0:
            self.extra_preserved_memory_during_sampling += memory_in_bytes
        else:
            raise ValueError("memory_in_bytes must be a positive integer")
        return

    def add_extra_model_patcher_during_sampling(self, model_patcher: ModelPatcher):
        # Use this to ask Forge to move extra model patchers to GPU during sampling.
        # This method will manage GPU memory perfectly.
        self.extra_model_patchers_during_sampling.append(model_patcher)
        return

    def add_extra_torch_module_during_sampling(self, m: torch.nn.Module, cast_to_unet_dtype: bool = True):
        # Use this method to bind an extra torch.nn.Module to this UNet during sampling.
        # This model `m` will be delegated to Forge memory management system.
        # `m` will be loaded to GPU everytime when sampling starts.
        # `m` will be unloaded if necessary.
        # `m` will influence Forge's judgement about use GPU memory or
        # capacity and decide whether to use module offload to make user's batch size larger.
        # Use cast_to_unet_dtype if you want `m` to have same dtype with unet during sampling.

        if cast_to_unet_dtype:
            m.to(self.model.diffusion_model.dtype)

        patcher = ModelPatcher(model=m, load_device=self.load_device, offload_device=self.offload_device)

        self.add_extra_model_patcher_during_sampling(patcher)
        return patcher

    def add_patched_controlnet(self, cnet):
        cnet.set_previous_controlnet(self.controlnet_linked_list)
        self.controlnet_linked_list = cnet
        return

    def list_controlnets(self) -> Generator:
        pointer = self.controlnet_linked_list
        while pointer is not None:
            yield pointer
            pointer = pointer.previous_controlnet

    def _ensure_transformer_options(self) -> Dict:
        if 'transformer_options' not in self.model_options:
            self.model_options['transformer_options'] = {}
        return self.model_options['transformer_options']

    def _append_option(self, option_dict: Dict, key: str, value: Any, ensure_uniqueness: bool = False) -> None:
        if key not in option_dict:
            option_dict[key] = []
        if not ensure_uniqueness or value not in option_dict[key]:
            option_dict[key].append(value)

    def append_model_option(self, k: str, v: Any, ensure_uniqueness: bool = False) -> None:
        self._append_option(self.model_options, k, v, ensure_uniqueness)

    def append_transformer_option(self, k: str, v: Any, ensure_uniqueness: bool = False) -> None:
        to = self._ensure_transformer_options()
        self._append_option(to, k, v, ensure_uniqueness)

    def set_transformer_option(self, k: str, v: Any) -> None:
        to = self._ensure_transformer_options()
        to[k] = v

    def add_conditioning_modifier(self, modifier, ensure_uniqueness=False):
        self.append_model_option('conditioning_modifiers', modifier, ensure_uniqueness)
        return

    def add_sampler_pre_cfg_function(self, modifier, ensure_uniqueness=False):
        self.append_model_option('sampler_pre_cfg_function', modifier, ensure_uniqueness)
        return

    def set_memory_peak_estimation_modifier(self, modifier):
        self.model_options['memory_peak_estimation_modifier'] = modifier
        return

    def add_alphas_cumprod_modifier(self, modifier, ensure_uniqueness=False):
        """

        For some reasons, this function only works in A1111's Script.process_batch(self, p, *args, **kwargs)

        For example, below is a worked modification:

        class ExampleScript(scripts.Script):

            def process_batch(self, p, *args, **kwargs):
                unet = p.sd_model.forge_objects.unet.clone()

                def modifier(x):
                    return x ** 0.5

                unet.add_alphas_cumprod_modifier(modifier)
                p.sd_model.forge_objects.unet = unet

                return

        This add_alphas_cumprod_modifier is the only patch option that should be used in process_batch()
        All other patch options should be called in process_before_every_sampling()

        """

        self.append_model_option('alphas_cumprod_modifiers', modifier, ensure_uniqueness)
        return

    def add_block_modifier(self, modifier, ensure_uniqueness=False):
        self.append_transformer_option('block_modifiers', modifier, ensure_uniqueness)
        return

    def add_block_inner_modifier(self, modifier, ensure_uniqueness=False):
        self.append_transformer_option('block_inner_modifiers', modifier, ensure_uniqueness)
        return

    def add_controlnet_conditioning_modifier(self, modifier, ensure_uniqueness=False):
        self.append_transformer_option('controlnet_conditioning_modifiers', modifier, ensure_uniqueness)
        return

    def set_group_norm_wrapper(self, wrapper):
        self.set_transformer_option('group_norm_wrapper', wrapper)
        return

    def set_controlnet_model_function_wrapper(self, wrapper):
        self.set_transformer_option('controlnet_model_function_wrapper', wrapper)
        return

    def set_model_replace_all(self, patch, target="attn1"):
        for block_name in ['input', 'middle', 'output']:
            for number, transformer_index in itertools.product(range(16), range(16)):
                self.set_model_patch_replace(patch, target, block_name, number, transformer_index)
        return

    def load_frozen_patcher(self, filename: str, state_dict: Dict, strength: float) -> None:
        patch_dict: Dict[str, Dict[str, List[Optional[torch.Tensor]]]] = {}

        # Process state dict into structured format
        for key, weight in state_dict.items():
            model_key, patch_type, weight_index = key.split('::')
            patch_dict.setdefault(model_key, {}).setdefault(patch_type, [None] * 16)[int(weight_index)] = weight

        # Flatten patches for add_patches method
        patches = {
            model_key: list(patch_data.items())
            for model_key, patch_data in patch_dict.items()
        }

        config = PatchConfig(
            filename=filename,
            patches=patches,
            strength_patch=strength,
            strength_model=1.0,
        )
        
        self.add_patches(
            filename=config.filename,
            patches=config.patches,
            strength_patch=config.strength_patch,
            strength_model=config.strength_model
        )
        return
