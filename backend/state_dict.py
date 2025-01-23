import torch
from typing import Set, Dict, Any
from copy import deepcopy


def load_state_dict(model, sd, ignore_errors=None, log_name=None, ignore_start=None):
    if ignore_errors is None:
        ignore_errors = []
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Convert lists to sets for faster membership testing
    ignore_errors_set = set(ignore_errors)
    missing = set(missing) - ignore_errors_set
    unexpected = set(unexpected) - ignore_errors_set

    if isinstance(ignore_start, str):
        missing = {x for x in missing if not x.startswith(ignore_start)}
        unexpected = {x for x in unexpected if not x.startswith(ignore_start)}

    log_name = log_name or type(model).__name__
    if missing:
        print(f'{log_name} Missing: {missing}')
    if unexpected:
        print(f'{log_name} Unexpected: {unexpected}')
    return


def state_dict_has(sd, prefix):
    # Use any() with generator expression instead of creating a list
    return any(k.startswith(prefix) for k in sd)


def filter_state_dict_with_prefix(sd, prefix, new_prefix=''):
    prefix_len = len(prefix)
    # Create new dict instead of modifying in place
    return {new_prefix + k[prefix_len:]: v for k, v in sd.items()
            if k.startswith(prefix)}


def try_filter_state_dict(sd, prefix_list, new_prefix=''):
    # Make a copy of sd to avoid modifying the original
    sd_copy = deepcopy(sd)
    return next(
        (
            filter_state_dict_with_prefix(sd_copy, prefix, new_prefix)
            for prefix in prefix_list
            if state_dict_has(sd_copy, prefix)
        ),
        {},
    )


def transformers_convert(sd, prefix_from, prefix_to, number):
    # Create a new dict instead of modifying in place
    new_sd = {}

    keys_to_replace = {
        f"{prefix_from}positional_embedding": f"{prefix_to}embeddings.position_embedding.weight",
        f"{prefix_from}token_embedding.weight": f"{prefix_to}embeddings.token_embedding.weight",
        f"{prefix_from}ln_final.weight": f"{prefix_to}final_layer_norm.weight",
        f"{prefix_from}ln_final.bias": f"{prefix_to}final_layer_norm.bias",
    }

    # Copy non-replaced keys
    new_sd |= {k: v for k, v in sd.items() if k not in keys_to_replace}

    # Add replaced keys
    new_sd |= {
        new_key: sd[old_key]
        for old_key, new_key in keys_to_replace.items()
        if old_key in sd
    }

    resblock_to_replace = {
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
        "attn.out_proj": "self_attn.out_proj",
    }

    # Pre-compute format strings
    k_from_template = f"{prefix_from}transformer.resblocks.{{}}.{{}}.{{}}"
    k_to_template = f"{prefix_to}encoder.layers.{{}}.{{}}.{{}}"

    for resblock in range(number):
        # Process resblock replacements
        for old_name, new_name in resblock_to_replace.items():
            for y in ("weight", "bias"):
                k = k_from_template.format(resblock, old_name, y)
                k_to = k_to_template.format(resblock, new_name, y)
                if k in sd:
                    new_sd[k_to] = sd.pop(k)

        # Process attention projections
        for y in ("weight", "bias"):
            k_from = f"{prefix_from}transformer.resblocks.{resblock}.attn.in_proj_{y}"
            if k_from in sd:
                weights = sd.pop(k_from)
                shape_from = weights.shape[0] // 3
                proj_names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]

                # Batch process the three projections
                for x, proj in enumerate(proj_names):
                    k_to = k_to_template.format(resblock, proj, y)
                    new_sd[k_to] = weights[shape_from*x:shape_from*(x + 1)]

    return new_sd


def state_dict_key_replace(state_dict, keys_to_replace):
    return {keys_to_replace.get(k, k): v for k, v in state_dict.items()}


def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    result = {} if filter_keys else dict(state_dict)
    # Convert keys to set for faster lookup
    prefix_set = set(replace_prefix.keys())

    for key, value in state_dict.items():
        if matching_prefix := next(
            (p for p in prefix_set if key.startswith(p)), None
        ):
            new_key = f"{replace_prefix[matching_prefix]}{key[len(matching_prefix):]}"
            result[new_key] = value
        elif not filter_keys:
            result[key] = value

    return result
