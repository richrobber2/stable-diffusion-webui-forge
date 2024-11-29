"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Comfy

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from packages_3rdparty.comfyui_lora_collection import utils


LORA_CLIP_MAP = {
    "mlp.fc1": "mlp_fc1",
    "mlp.fc2": "mlp_fc2",
    "self_attn.k_proj": "self_attn_k_proj",
    "self_attn.q_proj": "self_attn_q_proj",
    "self_attn.v_proj": "self_attn_v_proj",
    "self_attn.out_proj": "self_attn_out_proj",
}


def get_lora_key_pattern(prefix, suffix):
    """Helper to generate various lora key patterns"""
    patterns = [
        f"{prefix}.{suffix}",
        f"{prefix}_lora.{suffix}",
        f"{prefix}.lora_{suffix}",
        f"{prefix}.lora.{suffix}",
        f"{prefix}.lora_linear_layer.{suffix}"
    ]
    return patterns

def load_lora(lora, to_load):
    patch_dict = {}
    loaded_keys = set()

    def get_value_and_track(key, default=None):
        """Helper to get value from lora dict and track loaded keys"""
        if key in lora:
            loaded_keys.add(key)
            return lora[key]
        return default

    for x in to_load:
        # Load alpha and dora_scale
        alpha = get_value_and_track(f"{x}.alpha")
        alpha = alpha.item() if alpha is not None else None
        dora_scale = get_value_and_track(f"{x}.dora_scale")

        # Regular LoRA
        for up_pattern in ["lora_up.weight", "lora_B.weight"]:
            A_name = f"{x}.{up_pattern}"
            if A_name in lora:
                B_name = f"{x}.{up_pattern.replace('up', 'down').replace('B', 'A')}"
                mid_name = f"{x}.lora_mid.weight" if "lora_up" in up_pattern else None
                mid = get_value_and_track(mid_name) if mid_name else None

                patch_dict[to_load[x]] = ("lora", (
                    lora[A_name],
                    lora[B_name],
                    alpha,
                    mid,
                    dora_scale
                ))
                loaded_keys.update([A_name, B_name])
                break

        # LoHA
        loha_keys = {
            f"{x}.hada_w1_a": None,
            f"{x}.hada_w1_b": None,
            f"{x}.hada_w2_a": None,
            f"{x}.hada_w2_b": None
        }
        if all(k in lora for k in loha_keys):
            hada_t1 = get_value_and_track(f"{x}.hada_t1")
            hada_t2 = get_value_and_track(f"{x}.hada_t2")

            patch_dict[to_load[x]] = ("loha", (
                lora[f"{x}.hada_w1_a"],
                lora[f"{x}.hada_w1_b"],
                alpha,
                lora[f"{x}.hada_w2_a"],
                lora[f"{x}.hada_w2_b"],
                hada_t1,
                hada_t2,
                dora_scale
            ))
            loaded_keys.update(loha_keys.keys())

        # LoKR
        lokr_components = {
            'w1': get_value_and_track(f"{x}.lokr_w1"),
            'w2': get_value_and_track(f"{x}.lokr_w2"),
            'w1_a': get_value_and_track(f"{x}.lokr_w1_a"),
            'w1_b': get_value_and_track(f"{x}.lokr_w1_b"),
            'w2_a': get_value_and_track(f"{x}.lokr_w2_a"),
            'w2_b': get_value_and_track(f"{x}.lokr_w2_b"),
            't2': get_value_and_track(f"{x}.lokr_t2")
        }

        if any(v is not None for v in lokr_components.values()):
            patch_dict[to_load[x]] = ("lokr", (
                lokr_components['w1'],
                lokr_components['w2'],
                alpha,
                lokr_components['w1_a'],
                lokr_components['w1_b'],
                lokr_components['w2_a'],
                lokr_components['w2_b'],
                lokr_components['t2'],
                dora_scale
            ))

        # GLoRA
        glora_keys = [f"{x}.{suffix}.weight" for suffix in ['a1', 'a2', 'b1', 'b2']]
        if all(k in lora for k in glora_keys):
            patch_dict[to_load[x]] = ("glora", tuple(
                [lora[k] for k in glora_keys] + [alpha, dora_scale]
            ))
            loaded_keys.update(glora_keys)

        # Handle norms and diffs
        for norm_type in ['w_norm', 'b_norm']:
            norm_value = get_value_and_track(f"{x}.{norm_type}")
            if norm_value is not None:
                target = to_load[x] if norm_type == 'w_norm' else f"{to_load[x][:-len('.weight')]}.bias"
                patch_dict[target] = ("diff", (norm_value,))

    remaining_dict = {k: v for k, v in lora.items() if k not in loaded_keys}
    return patch_dict, remaining_dict


def model_lora_keys_clip(model, key_map=None):
    if key_map is None:
        key_map = {}

    sdk = model.state_dict().keys()

    text_model_lora_key = "lora_te_text_model_encoder_layers_{}_{}"
    clip_l_present = False
    for b in range(32): #TODO: clean up
        for c in LORA_CLIP_MAP:
            k = "clip_h.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                lora_key = text_model_lora_key.format(b, LORA_CLIP_MAP[c])
                key_map[lora_key] = k
                lora_key = "lora_te1_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c])
                key_map[lora_key] = k
                lora_key = "text_encoder.text_model.encoder.layers.{}.{}".format(b, c) #diffusers lora
                key_map[lora_key] = k

            k = "clip_l.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                lora_key = text_model_lora_key.format(b, LORA_CLIP_MAP[c])
                key_map[lora_key] = k
                lora_key = "lora_te1_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c]) #SDXL base
                key_map[lora_key] = k
                clip_l_present = True
                lora_key = "text_encoder.text_model.encoder.layers.{}.{}".format(b, c) #diffusers lora
                key_map[lora_key] = k

            k = "clip_g.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                if clip_l_present:
                    lora_key = "lora_te2_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c]) #SDXL base
                    key_map[lora_key] = k
                    lora_key = "text_encoder_2.text_model.encoder.layers.{}.{}".format(b, c) #diffusers lora
                    key_map[lora_key] = k
                else:
                    lora_key = "lora_te_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c]) #TODO: test if this is correct for SDXL-Refiner
                    key_map[lora_key] = k
                    lora_key = "text_encoder.text_model.encoder.layers.{}.{}".format(b, c) #diffusers lora
                    key_map[lora_key] = k
                    lora_key = "lora_prior_te_text_model_encoder_layers_{}_{}".format(b, LORA_CLIP_MAP[c]) #cascade lora: TODO put lora key prefix in the model config
                    key_map[lora_key] = k

    for k in sdk:
        if k.endswith(".weight"):
            if k.startswith("t5xxl.transformer."):#OneTrainer SD3 lora
                l_key = k[len("t5xxl.transformer."):-len(".weight")]
                lora_key = "lora_te3_{}".format(l_key.replace(".", "_"))
                key_map[lora_key] = k

                #####
                lora_key = "lora_te2_{}".format(l_key.replace(".", "_"))#OneTrainer Flux lora, by Forge
                key_map[lora_key] = k
                #####
    #         elif k.startswith("hydit_clip.transformer.bert."): #HunyuanDiT Lora
    #             l_key = k[len("hydit_clip.transformer.bert."):-len(".weight")]
    #             lora_key = "lora_te1_{}".format(l_key.replace(".", "_"))
    #             key_map[lora_key] = k


    k = "clip_g.transformer.text_projection.weight"
    if k in sdk:
    #    key_map["lora_prior_te_text_projection"] = k #cascade lora?
        key_map["text_encoder.text_projection"] = k #TODO: check if other lora have the text_projection too
        key_map["lora_te2_text_projection"] = k #OneTrainer SD3 lora

    k = "clip_l.transformer.text_projection.weight"
    if k in sdk:
        key_map["lora_te1_text_projection"] = k #OneTrainer SD3 lora, not necessary but omits warning

    return sdk, key_map


def model_lora_keys_unet(model, key_map=None):
    if key_map is None:
        key_map = {}
  # noqa: W293
    sd = model.state_dict()
    sdk = sd.keys()

    for k in sdk:
        if k.startswith("diffusion_model.") and k.endswith(".weight"):
            key_lora = k[len("diffusion_model."):-len(".weight")].replace(".", "_")
            key_map["lora_unet_{}".format(key_lora)] = k
            key_map["lora_prior_unet_{}".format(key_lora)] = k #cascade lora: TODO put lora key prefix in the model config
            key_map["{}".format(k[:-len(".weight")])] = k #generic lora format without any weird key names

    diffusers_keys = utils.unet_to_diffusers(model.diffusion_model.config)
    for k in diffusers_keys:
        if k.endswith(".weight"):
            unet_key = "diffusion_model.{}".format(diffusers_keys[k])
            key_lora = k[:-len(".weight")].replace(".", "_")
            key_map["lora_unet_{}".format(key_lora)] = unet_key

            diffusers_lora_prefix = ["", "unet."]
            for p in diffusers_lora_prefix:
                diffusers_lora_key = "{}{}".format(p, k[:-len(".weight")].replace(".to_", ".processor.to_"))
                if diffusers_lora_key.endswith(".to_out.0"):
                    diffusers_lora_key = diffusers_lora_key[:-2]
                key_map[diffusers_lora_key] = unet_key

    # if isinstance(model, comfy.model_base.SD3): #Diffusers lora SD3
    #     diffusers_keys = utils.mmdit_to_diffusers(model.diffusion_model.config, output_prefix="diffusion_model.")
    #     for k in diffusers_keys:
    #         if k.endswith(".weight"):
    #             to = diffusers_keys[k]
    #             key_lora = "transformer.{}".format(k[:-len(".weight")]) #regular diffusers sd3 lora format
    #             key_map[key_lora] = to
    #
    #             key_lora = "base_model.model.{}".format(k[:-len(".weight")]) #format for flash-sd3 lora and others?
    #             key_map[key_lora] = to
    #
    #             key_lora = "lora_transformer_{}".format(k[:-len(".weight")].replace(".", "_")) #OneTrainer lora
    #             key_map[key_lora] = to
    #
    # if isinstance(model, comfy.model_base.AuraFlow): #Diffusers lora AuraFlow
    #     diffusers_keys = utils.auraflow_to_diffusers(model.diffusion_model.config, output_prefix="diffusion_model.")
    #     for k in diffusers_keys:
    #         if k.endswith(".weight"):
    #             to = diffusers_keys[k]
    #             key_lora = "transformer.{}".format(k[:-len(".weight")]) #simpletrainer and probably regular diffusers lora format
    #             key_map[key_lora] = to
    #
    # if isinstance(model, comfy.model_base.HunyuanDiT):
    #     for k in sdk:
    #         if k.startswith("diffusion_model.") and k.endswith(".weight"):
    #             key_lora = k[len("diffusion_model."):-len(".weight")]
    #             key_map["base_model.model.{}".format(key_lora)] = k #official hunyuan lora format

    if 'flux' in model.config.huggingface_repo.lower(): #Diffusers lora Flux
        diffusers_keys = utils.flux_to_diffusers(model.diffusion_model.config, output_prefix="diffusion_model.")
        for k in diffusers_keys:
            if k.endswith(".weight"):
                to = diffusers_keys[k]
                key_map["transformer.{}".format(k[:-len(".weight")])] = to  # simpletrainer and probably regular diffusers flux lora format
                key_map["lycoris_{}".format(k[:-len(".weight")].replace(".", "_"))] = to  # simpletrainer lycoris
                key_map["lora_transformer_{}".format(k[:-len(".weight")].replace(".", "_"))] = to  # onetrainer

    return sdk, key_map
