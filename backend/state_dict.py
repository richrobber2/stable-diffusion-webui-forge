import torch


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
    # Use dictionary comprehension instead of loop
    new_sd = {new_prefix + k[prefix_len:]: v for k, v in sd.items()
              if k.startswith(prefix)}

    # Batch delete keys
    keys_to_delete = [k for k in sd if k.startswith(prefix)]
    for k in keys_to_delete:
        del sd[k]

    return new_sd


def try_filter_state_dict(sd, prefix_list, new_prefix=''):
    for prefix in prefix_list:
        if state_dict_has(sd, prefix):
            return filter_state_dict_with_prefix(sd, prefix, new_prefix)
    return {}


def transformers_convert(sd, prefix_from, prefix_to, number):
    keys_to_replace = {
        f"{prefix_from}positional_embedding": f"{prefix_to}embeddings.position_embedding.weight",
        f"{prefix_from}token_embedding.weight": f"{prefix_to}embeddings.token_embedding.weight",
        f"{prefix_from}ln_final.weight": f"{prefix_to}final_layer_norm.weight",
        f"{prefix_from}ln_final.bias": f"{prefix_to}final_layer_norm.bias",
    }

    # Batch process key replacements
    for old_key, new_key in keys_to_replace.items():
        if old_key in sd:
            sd[new_key] = sd.pop(old_key)

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
                    sd[k_to] = sd.pop(k)

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
                    sd[k_to] = weights[shape_from*x:shape_from*(x + 1)]

    return sd


def state_dict_key_replace(state_dict, keys_to_replace):
    for x in keys_to_replace:
        if x in state_dict:
            state_dict[keys_to_replace[x]] = state_dict.pop(x)
    return state_dict


def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    if filter_keys:
        out = {}
    else:
        out = state_dict
    for rp in replace_prefix:
        replace = [(a, "{}{}".format(replace_prefix[rp], a[len(rp):])) for a in filter(lambda a: a.startswith(rp), state_dict.keys())]
        for x in replace:
            w = state_dict.pop(x[0])
            out[x[1]] = w
    return out
