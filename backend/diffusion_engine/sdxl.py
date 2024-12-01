import torch

from huggingface_guess import model_list
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.vae import VAE
from backend.patcher.unet import UnetPatcher
from backend.text_processing.classic_engine import ClassicTextProcessingEngine
from backend.args import dynamic_args
from backend import memory_management
from backend.nn.unet import Timestep


class StableDiffusionXL(ForgeDiffusionEngine):
    matched_guesses = [model_list.SDXL]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(
            model_dict={
                'clip_l': huggingface_components['text_encoder'],
                'clip_g': huggingface_components['text_encoder_2']
            },
            tokenizer_dict={
                'clip_l': huggingface_components['tokenizer'],
                'clip_g': huggingface_components['tokenizer_2']
            }
        )

        vae = VAE(model=huggingface_components['vae'])

        unet = UnetPatcher.from_model(
            model=huggingface_components['unet'],
            diffusers_scheduler=huggingface_components['scheduler'],
            config=estimated_config
        )

        self.text_processing_engine_l = ClassicTextProcessingEngine(
            text_encoder=clip.cond_stage_model.clip_l,
            tokenizer=clip.tokenizer.clip_l,
            embedding_dir=dynamic_args['embedding_dir'],
            embedding_key='clip_l',
            embedding_expected_shape=2048,
            emphasis_name=dynamic_args['emphasis_name'],
            text_projection=False,
            minimal_clip_skip=2,
            clip_skip=2,
            return_pooled=False,
            final_layer_norm=False,
        )

        self.text_processing_engine_g = ClassicTextProcessingEngine(
            text_encoder=clip.cond_stage_model.clip_g,
            tokenizer=clip.tokenizer.clip_g,
            embedding_dir=dynamic_args['embedding_dir'],
            embedding_key='clip_g',
            embedding_expected_shape=2048,
            emphasis_name=dynamic_args['emphasis_name'],
            text_projection=True,
            minimal_clip_skip=2,
            clip_skip=2,
            return_pooled=True,
            final_layer_norm=False,
        )

        self.embedder = Timestep(256)

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        # WebUI Legacy
        self.is_sdxl = True

    def set_clip_skip(self, clip_skip):
        self.text_processing_engine_l.clip_skip = clip_skip
        self.text_processing_engine_g.clip_skip = clip_skip

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)

        cond_l = self.text_processing_engine_l(prompt)
        cond_g, clip_pooled = self.text_processing_engine_g(prompt)

        if cond_l.dim() != 3 or cond_g.dim() != 3:
            raise ValueError("Invalid conditioning tensor dimensions")

        width = getattr(prompt, 'width', 1024) or 1024
        height = getattr(prompt, 'height', 1024) or 1024
        is_negative_prompt = getattr(prompt, 'is_negative_prompt', False)

        # Optimize tensor creation by using torch.empty and in-place assignment
        embedding_values = torch.empty(6, 1, device=clip_pooled.device)
        embedding_values[0].fill_(height)
        embedding_values[1].fill_(width)
        embedding_values[2].fill_(0)
        embedding_values[3].fill_(0)
        embedding_values[4].fill_(height)
        embedding_values[5].fill_(width)

        embedded = self.embedder(embedding_values)
        flat = embedded.view(1, -1).expand(clip_pooled.shape[0], -1)

        if is_negative_prompt and all(x == '' for x in prompt):
            zeros = torch.zeros_like
            clip_pooled, cond_l, cond_g = map(zeros, (clip_pooled, cond_l, cond_g))

        return {
            'crossattn': torch.cat([cond_l, cond_g], dim=2),
            'vector': torch.cat([clip_pooled, flat], dim=1),
        }

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        _, token_count = self.text_processing_engine_l.process_texts([prompt])
        return token_count, self.text_processing_engine_l.get_target_prompt_token_count(token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x):
        x = x.movedim(1, -1)
        x.mul_(0.5).add_(0.5)  # in-place operations
        sample = self.forge_objects.vae.encode(x)
        return self.forge_objects.vae.first_stage_model.process_in(sample).to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        processed = self.forge_objects.vae.first_stage_model.process_out(x)
        decoded = self.forge_objects.vae.decode(processed)
        decoded = decoded.movedim(-1, 1)
        decoded.mul_(2.0).sub_(1.0)  # in-place operations
        return decoded.to(x)
