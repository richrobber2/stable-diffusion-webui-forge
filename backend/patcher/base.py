# Model Patching System, Copyright Forge 2024

# API Templates partially extracted From ComfyUI, the actual implementation for those APIs
# are from Forge, implemented from scratch (after forge-v1.0.1), and may have
# certain level of differences.


import copy
import inspect

from backend import memory_management, utils
from backend.patcher.lora import LoraLoader


def set_model_options_patch_replace(model_options, patch, name, block_name, number, transformer_index=None):
    to = model_options["transformer_options"]

    if "patches_replace" not in to:
        to["patches_replace"] = {}

    if name not in to["patches_replace"]:
        to["patches_replace"][name] = {}

    if transformer_index is not None:
        block = (block_name, number, transformer_index)
    else:
        block = (block_name, number)
    to["patches_replace"][name][block] = patch
    return model_options


def set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=False):
    sampler_post_cfg = model_options.setdefault("sampler_post_cfg_function", [])
    sampler_post_cfg.append(post_cfg_function)
    if disable_cfg1_optimization:
        model_options["disable_cfg1_optimization"] = True
    return model_options


def set_model_options_pre_cfg_function(model_options, pre_cfg_function, disable_cfg1_optimization=False):
    sampler_pre_cfg = model_options.setdefault("sampler_pre_cfg_function", [])
    sampler_pre_cfg.append(pre_cfg_function)
    if disable_cfg1_optimization:
        model_options["disable_cfg1_optimization"] = True
    return model_options


class ModelPatcher:
    def __init__(self, model, load_device, offload_device, size=0, current_device=None, **kwargs):
        self.size = size
        self.model = model
        self.lora_patches = {}
        self.object_patches = {}
        self.object_patches_backup = {}
        self.model_options = {"transformer_options": {}}
        self.model_size()
        self.load_device = load_device
        self.offload_device = offload_device

        if not hasattr(model, 'lora_loader'):
            model.lora_loader = LoraLoader(model)

        self.lora_loader: LoraLoader = model.lora_loader

        if current_device is None:
            self.current_device = self.offload_device
        else:
            self.current_device = current_device

    def model_size(self):
        if self.size > 0:
            return self.size
        self.size = memory_management.module_size(self.model)
        return self.size

    def clone(self):
        n = ModelPatcher(
            self.model,
            self.load_device,
            self.offload_device,
            self.size,
            self.current_device
        )
        n.lora_patches = self.lora_patches.copy()
        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        return n

    def is_clone(self, other):
        return hasattr(other, 'model') and self.model is other.model

    def add_patches(self, *, filename, patches, strength_patch=1.0, strength_model=1.0, online_mode=False):
        lora_identifier = (filename, strength_patch, strength_model, online_mode)
        this_patches = {}

        p = set()
        model_keys = {k for k, _ in self.model.named_parameters()}

        for k in patches:
            offset = None
            function = None

            if isinstance(k, str):
                key = k
            else:
                offset = k[1]
                key = k[0]
                if len(k) > 2:
                    function = k[2]

            if key in model_keys:
                p.add(k)
                current_patches = this_patches.setdefault(key, [])
                current_patches.append([strength_patch, patches[k], strength_model, offset, function])

        self.lora_patches[lora_identifier] = this_patches
        return p

    def has_online_lora(self):
        return any(online_mode for (_, _, _, online_mode) in self.lora_patches.keys())

    def refresh_loras(self):
        self.lora_loader.refresh(lora_patches=self.lora_patches, offload_device=self.offload_device)

    def memory_required(self, input_shape):
        return self.model.memory_required(input_shape=input_shape)

    def set_model_sampler_cfg_function(self, sampler_cfg_function, disable_cfg1_optimization=False):
        if len(inspect.signature(sampler_cfg_function).parameters) == 3:
            self.model_options["sampler_cfg_function"] = lambda args: sampler_cfg_function(
                args["cond"], args["uncond"], args["cond_scale"]
            )  # Old way
        else:
            self.model_options["sampler_cfg_function"] = sampler_cfg_function
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True

    def set_model_sampler_post_cfg_function(self, post_cfg_function, disable_cfg1_optimization=False):
        self.model_options = set_model_options_post_cfg_function(
            self.model_options, post_cfg_function, disable_cfg1_optimization
        )

    def set_model_sampler_pre_cfg_function(self, pre_cfg_function, disable_cfg1_optimization=False):
        self.model_options = set_model_options_pre_cfg_function(
            self.model_options, pre_cfg_function, disable_cfg1_optimization
        )

    def set_model_unet_function_wrapper(self, unet_wrapper_function):
        self.model_options["model_function_wrapper"] = unet_wrapper_function

    def set_model_vae_encode_wrapper(self, wrapper_function):
        self.model_options["model_vae_encode_wrapper"] = wrapper_function

    def set_model_vae_decode_wrapper(self, wrapper_function):
        self.model_options["model_vae_decode_wrapper"] = wrapper_function

    def set_model_vae_regulation(self, vae_regulation):
        self.model_options["model_vae_regulation"] = vae_regulation

    def set_model_denoise_mask_function(self, denoise_mask_function):
        self.model_options["denoise_mask_function"] = denoise_mask_function

    def set_model_patch(self, patch, name):
        patches = self.model_options["transformer_options"].setdefault("patches", {})
        patches.setdefault(name, []).append(patch)

    def set_model_patch_replace(self, patch, name, block_name, number, transformer_index=None):
        self.model_options = set_model_options_patch_replace(
            self.model_options, patch, name, block_name, number, transformer_index=transformer_index
        )

    def set_model_attn1_patch(self, patch):
        self.set_model_patch(patch, "attn1_patch")

    def set_model_attn2_patch(self, patch):
        self.set_model_patch(patch, "attn2_patch")

    def set_model_attn1_replace(self, patch, block_name, number, transformer_index=None):
        self.set_model_patch_replace(patch, "attn1", block_name, number, transformer_index)

    def set_model_attn2_replace(self, patch, block_name, number, transformer_index=None):
        self.set_model_patch_replace(patch, "attn2", block_name, number, transformer_index)

    def set_model_attn1_output_patch(self, patch):
        self.set_model_patch(patch, "attn1_output_patch")

    def set_model_attn2_output_patch(self, patch):
        self.set_model_patch(patch, "attn2_output_patch")

    def set_model_input_block_patch(self, patch):
        self.set_model_patch(patch, "input_block_patch")

    def set_model_input_block_patch_after_skip(self, patch):
        self.set_model_patch(patch, "input_block_patch_after_skip")

    def set_model_output_block_patch(self, patch):
        self.set_model_patch(patch, "output_block_patch")

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def get_model_object(self, name):
        if name in self.object_patches:
            return self.object_patches[name]
        elif name in self.object_patches_backup:
            return self.object_patches_backup[name]
        else:
            return utils.get_attr(self.model, name)

    def model_patches_to(self, device):
        transformer_options = self.model_options.get("transformer_options", {})
        patches = transformer_options.get("patches", {})
        patches_replace = transformer_options.get("patches_replace", {})

        for patch_list in patches.values():
            for i in range(len(patch_list)):
                patch = patch_list[i]
                if hasattr(patch, "to"):
                    patch_list[i] = patch.to(device)

        for patch_dict in patches_replace.values():
            for key, patch in patch_dict.items():
                if hasattr(patch, "to"):
                    patch_dict[key] = patch.to(device)

        if "model_function_wrapper" in self.model_options:
            wrap_func = self.model_options["model_function_wrapper"]
            if hasattr(wrap_func, "to"):
                self.model_options["model_function_wrapper"] = wrap_func.to(device)

    def model_dtype(self):
        if hasattr(self.model, "get_dtype"):
            return self.model.get_dtype()

    def get_key_patches(self, filter_prefix=None):
        memory_management.unload_model_clones(self)
        model_sd = self.model_state_dict(filter_prefix)
        return {
            k: (
                [model_sd[k]] + self.patches[k]
                if k in self.patches
                else (model_sd[k],)
            )
            for k in model_sd
        }

    def model_state_dict(self, filter_prefix=None):
        sd = self.model.state_dict()
        if filter_prefix is not None:
            keys_to_remove = [k for k in sd if not k.startswith(filter_prefix)]
            for k in keys_to_remove:
                sd.pop(k)
        return sd

    def forge_patch_model(self, target_device=None):
        for k, item in self.object_patches.items():
            old = utils.get_attr(self.model, k)

            if k not in self.object_patches_backup:
                self.object_patches_backup[k] = old

            utils.set_attr_raw(self.model, k, item)

        if target_device is not None:
            self.model.to(target_device)
            self.current_device = target_device

        return self.model

    def forge_unpatch_model(self, target_device=None):
        if target_device is not None:
            self.model.to(target_device)
            self.current_device = target_device

        for k, backup_item in self.object_patches_backup.items():
            utils.set_attr_raw(self.model, k, backup_item)

        self.object_patches_backup.clear()
        return
