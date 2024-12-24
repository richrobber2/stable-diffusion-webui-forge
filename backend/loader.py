import os
import torch
import logging
import importlib

import backend.args
import contextlib
import huggingface_guess

from diffusers import DiffusionPipeline
from transformers import modeling_utils

from backend import memory_management
from backend.utils import read_arbitrary_config, load_torch_file, beautiful_print_gguf_state_dict_statics
from backend.state_dict import try_filter_state_dict, load_state_dict
from backend.operations import using_forge_operations
from backend.nn.vae import IntegratedAutoencoderKL
from backend.nn.clip import IntegratedCLIP
from backend.nn.unet import IntegratedUNet2DConditionModel

from backend.diffusion_engine.sd15 import StableDiffusion
from backend.diffusion_engine.sd20 import StableDiffusion2
from backend.diffusion_engine.sdxl import StableDiffusionXL
from backend.diffusion_engine.flux import Flux

possible_models = [StableDiffusion, StableDiffusion2, StableDiffusionXL, Flux]

logging.getLogger("diffusers").setLevel(logging.ERROR)
dir_path = os.path.dirname(__file__)


def load_huggingface_component(guess, component_name, lib_name, cls_name, repo_path, state_dict):
    """
    Load a Hugging Face component given its configuration and optional state_dict.
    Components can be tokenizers, schedulers, VAE, text encoders, UNets, etc.
    """
    config_path = os.path.join(repo_path, component_name)

    # Skip irrelevant components
    if component_name in ['feature_extractor', 'safety_checker']:
        return None

    if lib_name in ['transformers', 'diffusers']:
        # Handle known component types
        if component_name == 'scheduler':
            cls = getattr(importlib.import_module(lib_name), cls_name)
            return cls.from_pretrained(os.path.join(repo_path, component_name))

        if component_name.startswith('tokenizer'):
            # Load tokenizer and suppress length warnings
            cls = getattr(importlib.import_module(lib_name), cls_name)
            comp = cls.from_pretrained(os.path.join(repo_path, component_name))
            comp._eventual_warn_about_too_long_sequence = lambda *args, **kwargs: None
            return comp

        # Specialized components based on class names
        if cls_name == 'AutoencoderKL':
            return _extracted_from_load_huggingface_component_17(state_dict, config_path)

        if component_name.startswith('text_encoder') and cls_name in ['CLIPTextModel', 'CLIPTextModelWithProjection']:
            return _extracted_from_load_huggingface_component_19(state_dict, config_path, cls_name)

        if cls_name == 'T5EncoderModel':
            return _extracted_from_load_huggingface_component_38(state_dict, config_path, cls_name)

        if cls_name == 'UNet2DConditionModel':
            # UNet model (or similar conditional diffusion models)
            assert isinstance(state_dict, dict) and len(state_dict) > 16, 'You do not have a proper UNet state dict!'
            model_loader = lambda c: IntegratedUNet2DConditionModel.from_config(c)
            return _extracted_from_load_huggingface_component_88(guess, state_dict, model_loader)

        elif cls_name == 'FluxTransformer2DModel':
            assert isinstance(state_dict, dict) and len(state_dict) > 16, 'You do not have a proper Flux state dict!'
            from backend.nn.flux import IntegratedFluxTransformer2DModel
            model_loader = lambda c: IntegratedFluxTransformer2DModel(**c)
            return _extracted_from_load_huggingface_component_88(guess, state_dict, model_loader)

    # If none of the conditions matched
    print(f'Skipped: {component_name} = {lib_name}.{cls_name}')
    return None


def _decide_storage_dtype_and_log(state_dict, default_dtype, component_name):
    """
    Decide the storage dtype for a model based on the given state_dict and default dtype.
    Also logs details about the chosen dtype and handles special quantization formats.

    Args:
        state_dict (dict): The component's state dict.
        default_dtype (torch.dtype or str): The default dtype decided by heuristics.
        component_name (str): For logging purposes, the name of the component (e.g. 'T5', 'UNet', 'VAE').

    Returns:
        torch.dtype or str: The selected storage dtype.
    """
    state_dict_dtype = memory_management.state_dict_dtype(state_dict)
    # If special quantization is detected
    special_types = [torch.float8_e4m3fn, torch.float8_e5m2, 'nf4', 'fp4', 'gguf']
    if state_dict_dtype in special_types:
        print(f'Using Detected {component_name} Data Type: {state_dict_dtype}')
        # For gguf, print additional info
        if state_dict_dtype == 'gguf':
            beautiful_print_gguf_state_dict_statics(state_dict)
        return state_dict_dtype
    else:
        print(f'Using Default {component_name} Data Type: {default_dtype}')
        return default_dtype


def _init_model_with_forge(config, device, storage_dtype, computation_dtype=None, manual_cast_enabled=True, bnb_dtype=None, model_loader=None):
    """
    Initialize a model with Forge operations considering device, dtype, and optional quantization parameters.

    Args:
        config (dict): The model config.
        device (torch.device): The device to initially load parameters on.
        storage_dtype (torch.dtype or str): The storage dtype for the model.
        computation_dtype (torch.dtype, optional): The dtype for computations. Defaults to storage_dtype if None.
        manual_cast_enabled (bool): Whether manual casting is enabled.
        bnb_dtype (str, optional): If using bitsandbytes or special quantization formats.
        model_loader (callable, optional): A function that returns a model instance when given a config.

    Returns:
        nn.Module: The initialized model.
    """
    if computation_dtype is None:
        computation_dtype = storage_dtype

    op_args = dict(device=device, dtype=storage_dtype, manual_cast_enabled=manual_cast_enabled)
    if bnb_dtype:
        op_args['bnb_dtype'] = bnb_dtype

    with modeling_utils.no_init_weights():
        with using_forge_operations(**op_args):
            model = model_loader(config) if model_loader else None
            return model


# Specialized loading functions:

def _extracted_from_load_huggingface_component_38(state_dict, config_path, cls_name):
    # Load T5 encoder model
    from backend.nn.t5 import IntegratedT5
    config = read_arbitrary_config(config_path)

    default_dtype = memory_management.text_encoder_dtype()
    storage_dtype = _decide_storage_dtype_and_log(state_dict, default_dtype, 'T5')

    # Determine if we use bnb quantization
    quant_formats = ['nf4', 'fp4', 'gguf']
    if storage_dtype in quant_formats:
        # No manual cast in these quant modes
        model = _init_model_with_forge(config, memory_management.cpu, memory_management.text_encoder_dtype(), manual_cast_enabled=False, bnb_dtype=storage_dtype, model_loader=IntegratedT5)
    else:
        model = _init_model_with_forge(config, memory_management.cpu, storage_dtype, manual_cast_enabled=True, model_loader=IntegratedT5)

    load_state_dict(model, state_dict, log_name=cls_name, ignore_errors=['transformer.encoder.embed_tokens.weight', 'logit_scale'])
    return model


def _extracted_from_load_huggingface_component_19(state_dict, config_path, cls_name):
    # Load CLIP text encoder model
    from transformers import CLIPTextConfig, CLIPTextModel
    config = CLIPTextConfig.from_pretrained(config_path)

    model = _init_model_with_forge(config, memory_management.cpu, memory_management.text_encoder_dtype(), manual_cast_enabled=True,
                                   model_loader=lambda c: IntegratedCLIP(CLIPTextModel, c, add_text_projection=True))

    load_state_dict(model, state_dict, ignore_errors=[
        'transformer.text_projection.weight',
        'transformer.text_model.embeddings.position_ids',
        'logit_scale'
    ], log_name=cls_name)
    return model


def _extracted_from_load_huggingface_component_17(state_dict, config_path):
    # Load VAE
    assert isinstance(state_dict, dict) and len(state_dict) > 16, 'Invalid VAE state dict!'
    config = IntegratedAutoencoderKL.load_config(config_path)

    # Initialize VAE directly with chosen dtype (usually on CPU)
    model = _init_model_with_forge(config, memory_management.cpu, memory_management.vae_dtype(), model_loader=IntegratedAutoencoderKL.from_config)

    # Convert diffusers format if needed
    if 'decoder.up_blocks.0.resnets.0.norm1.weight' in state_dict:
        state_dict = huggingface_guess.diffusers_convert.convert_vae_state_dict(state_dict)

    load_state_dict(model, state_dict, ignore_start='loss.')
    return model


def _extracted_from_load_huggingface_component_88(guess, state_dict, model_loader):
    # Load UNet or Flux model
    unet_config = guess.unet_config.copy()
    state_dict_parameters = memory_management.state_dict_parameters(state_dict)

    # Decide storage dtype based on state dict
    default_storage_dtype = memory_management.unet_dtype(model_params=state_dict_parameters, supported_dtypes=guess.supported_inference_dtypes)
    unet_storage_dtype_overwrite = backend.args.dynamic_args.get('forge_unet_storage_dtype')
    if unet_storage_dtype_overwrite is not None:
        default_storage_dtype = unet_storage_dtype_overwrite

    storage_dtype = _decide_storage_dtype_and_log(state_dict, default_storage_dtype, 'UNet')

    load_device = memory_management.get_torch_device()
    computation_dtype = memory_management.get_computation_dtype(load_device, parameters=state_dict_parameters, supported_dtypes=guess.supported_inference_dtypes)
    offload_device = memory_management.unet_offload_device()

    quant_formats = ['nf4', 'fp4', 'gguf']
    if storage_dtype in quant_formats:
        # When using quantized formats
        initial_device = memory_management.unet_inital_load_device(parameters=state_dict_parameters, dtype=computation_dtype)
        model = _init_model_with_forge(unet_config, initial_device, computation_dtype, manual_cast_enabled=False, bnb_dtype=storage_dtype, model_loader=model_loader)
    else:
        # Non-quantized scenario
        initial_device = memory_management.unet_inital_load_device(parameters=state_dict_parameters, dtype=storage_dtype)
        need_manual_cast = (storage_dtype != computation_dtype)
        model = _init_model_with_forge(unet_config, initial_device, storage_dtype, computation_dtype=computation_dtype,
                                       manual_cast_enabled=need_manual_cast, model_loader=model_loader)

    load_state_dict(model, state_dict)

    # Store config and dtype info in model
    if hasattr(model, '_internal_dict'):
        model._internal_dict = unet_config
    else:
        model.config = unet_config

    model.storage_dtype = storage_dtype
    model.computation_dtype = computation_dtype
    model.load_device = load_device
    model.initial_device = initial_device
    model.offload_device = offload_device

    return model


def replace_state_dict(sd, asd, guess):
    """
    Merge or replace parts of sd with asd, handling special naming conventions for T5, VAE, CLIP.
    """
    vae_key_prefix = guess.vae_key_prefix[0]
    text_encoder_key_prefix = guess.text_encoder_key_prefix[0]

    # Handle weird T5 format if present
    if 'enc.blk.0.attn_k.weight' in asd:
        wierd_t5_format_from_city96 = {
            "enc.": "encoder.",
            ".blk.": ".block.",
            "token_embd": "shared",
            "output_norm": "final_layer_norm",
            "attn_q": "layer.0.SelfAttention.q",
            "attn_k": "layer.0.SelfAttention.k",
            "attn_v": "layer.0.SelfAttention.v",
            "attn_o": "layer.0.SelfAttention.o",
            "attn_norm": "layer.0.layer_norm",
            "attn_rel_b": "layer.0.SelfAttention.relative_attention_bias",
            "ffn_up": "layer.1.DenseReluDense.wi_1",
            "ffn_down": "layer.1.DenseReluDense.wo",
            "ffn_gate": "layer.1.DenseReluDense.wi_0",
            "ffn_norm": "layer.1.layer_norm",
        }
        wierd_t5_pre_quant_keys_from_city96 = ['shared.weight']

        # Rename keys in asd
        for key in list(asd.keys()):
            new_key = key
            for old, new in wierd_t5_format_from_city96.items():
                new_key = new_key.replace(old, new)
            asd[new_key] = asd.pop(key)

        # Dequantize special keys
        for key in wierd_t5_pre_quant_keys_from_city96:
            asd[key] = asd[key].dequantize_as_pytorch_parameter()

    # Handle merging VAE weights if needed
    if "decoder.conv_in.weight" in asd:
        keys_to_delete = [k for k in sd if k.startswith(vae_key_prefix)]
        for k in keys_to_delete:
            del sd[k]
        for k, v in asd.items():
            sd[vae_key_prefix + k] = v

    # Handle merging CLIP weights
    if 'text_model.encoder.layers.0.layer_norm1.weight' in asd:
        keys_to_delete = [k for k in sd if k.startswith(f"{text_encoder_key_prefix}clip_l.")]
        for k in keys_to_delete:
            del sd[k]
        for k, v in asd.items():
            sd[f"{text_encoder_key_prefix}clip_l.transformer.{k}"] = v

    # Handle merging T5 XXL weights
    if 'encoder.block.0.layer.0.SelfAttention.k.weight' in asd:
        keys_to_delete = [k for k in sd if k.startswith(f"{text_encoder_key_prefix}t5xxl.")]
        for k in keys_to_delete:
            del sd[k]
        for k, v in asd.items():
            sd[f"{text_encoder_key_prefix}t5xxl.transformer.{k}"] = v

    return sd


def preprocess_state_dict(sd):
    """
    Preprocess the given state_dict for uniformity.
    Some models need a "model.diffusion_model." prefix if missing.
    """
    if any("double_block" in k for k in sd.keys()) and not any(k.startswith("model.diffusion_model") for k in sd.keys()):
        sd = {f"model.diffusion_model.{k}": v for k, v in sd.items()}
    return sd


def split_state_dict(sd, additional_state_dicts: list = None):
    """
    Split the loaded state_dict into UNet, VAE, and text encoder components based on guessed model architecture.
    """
    sd = load_torch_file(sd)
    sd = preprocess_state_dict(sd)
    guess = huggingface_guess.guess(sd)

    if isinstance(additional_state_dicts, list):
        for asd in additional_state_dicts:
            asd = load_torch_file(asd)
            sd = replace_state_dict(sd, asd, guess)

    guess.clip_target = guess.clip_target(sd)
    guess.model_type = guess.model_type(sd)
    guess.ztsnr = 'ztsnr' in sd

    state_dict = {
        guess.unet_target: try_filter_state_dict(sd, guess.unet_key_prefix),
        guess.vae_target: try_filter_state_dict(sd, guess.vae_key_prefix)
    }

    sd = guess.process_clip_state_dict(sd)

    for k, v in guess.clip_target.items():
        state_dict[v] = try_filter_state_dict(sd, [f'{k}.'])

    state_dict['ignore'] = sd

    print_dict = {k: len(v) for k, v in state_dict.items()}
    print(f'StateDict Keys: {print_dict}')

    del state_dict['ignore']

    return state_dict, guess


@torch.inference_mode()
def forge_loader(sd, additional_state_dicts=None):
    """
    Main loader function: from a given state_dict and optional additional dicts,
    identify the model type, load huggingface components, and return a stable diffusion model instance.
    """
    state_dicts, estimated_config = get_state_dicts_and_config(sd, additional_state_dicts)
    repo_name = estimated_config.huggingface_repo
    local_path = os.path.join(dir_path, 'huggingface', repo_name)
    config = DiffusionPipeline.load_config(local_path)

    # Load components
    huggingface_components = load_components(config, state_dicts, estimated_config, local_path)
    yaml_config_prediction_type = get_yaml_config_prediction_type(sd)
    update_prediction_type(huggingface_components, yaml_config_prediction_type, estimated_config)

    return get_model(estimated_config, huggingface_components)


def get_state_dicts_and_config(sd, additional_state_dicts):
    try:
        return split_state_dict(sd, additional_state_dicts=additional_state_dicts)
    except Exception as e:
        raise ValueError(f'Failed to recognize model type! Error: {e}')


def load_components(config, state_dicts, estimated_config, local_path):
    """
    Load required huggingface components (tokenizer, scheduler, UNet, VAE, etc.) based on the pipeline config.
    """
    huggingface_components = {}
    for component_name, v in config.items():
        if isinstance(v, list) and len(v) == 2:
            lib_name, cls_name = v
            component_sd = state_dicts.get(component_name, None)
            component = load_huggingface_component(estimated_config, component_name, lib_name, cls_name, local_path, component_sd)
            if component_sd is not None:
                del state_dicts[component_name]
            if component is not None:
                huggingface_components[component_name] = component
    return huggingface_components


def get_yaml_config_prediction_type(sd):
    """
    Check if there's a .yaml config file next to the model file to determine the prediction_type ('v_prediction', 'epsilon', etc.)
    """
    yaml_config = None
    with contextlib.suppress(ImportError):
        import yaml
        from pathlib import Path
        config_filename = f'{os.path.splitext(sd)[0]}.yaml'
        if Path(config_filename).is_file():
            with open(config_filename, 'r') as stream:
                yaml_config = yaml.safe_load(stream)
    if yaml_config is not None:
        yaml_config_prediction_type = (
            yaml_config.get('model', {}).get('params', {}).get('parameterization', '')
            or yaml_config.get('model', {}).get('params', {}).get('denoiser_config', {}).get('params', {}).get('scaling_config', {}).get('target', '')
        )
        if yaml_config_prediction_type == 'v' or yaml_config_prediction_type.endswith(".VScaling"):
            return 'v_prediction'
    return ''


def update_prediction_type(huggingface_components, yaml_config_prediction_type, estimated_config):
    """
    Update the scheduler's prediction_type based on YAML config or fallback to the model_type guess.
    """
    if (
        'scheduler' in huggingface_components
        and hasattr(huggingface_components['scheduler'], 'config')
        and 'prediction_type' in huggingface_components['scheduler'].config
    ):
        if yaml_config_prediction_type:
            huggingface_components['scheduler'].config.prediction_type = yaml_config_prediction_type
        else:
            prediction_types = {
                'EPS': 'epsilon',
                'V_PREDICTION': 'v_prediction',
                'EDM': 'edm',
            }
            huggingface_components['scheduler'].config.prediction_type = prediction_types.get(
                estimated_config.model_type.name,
                huggingface_components['scheduler'].config.prediction_type
            )


def get_model(estimated_config, huggingface_components):
    """
    Given an estimated_config and loaded huggingface_components, try to instantiate one of the known stable diffusion model classes.
    """
    for M in possible_models:
        if any(isinstance(estimated_config, x) for x in M.matched_guesses):
            return M(estimated_config=estimated_config, huggingface_components=huggingface_components)

    print('Failed to recognize model type!')
    return None
