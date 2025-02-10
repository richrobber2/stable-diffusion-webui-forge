import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from huggingface_guess import model_list
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.vae import VAE
from backend.patcher.unet import UnetPatcher
from backend.text_processing.classic_engine import ClassicTextProcessingEngine
from backend.args import dynamic_args
from backend import memory_management
from backend.nn.unet import Timestep
from backend.text_processing.prompt_structs import PromptConfig, TextualPrompt

@dataclass
class PromptConfig:
    width: int = 1024
    height: int = 1024
    is_negative_prompt: bool = False

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
    def get_learned_conditioning(self, prompt: Union[List[str], List[PromptConfig]]) -> Dict[str, torch.Tensor]:
        """Process prompt into conditioning tensors for image generation.
        
        Args:
            prompt: List of prompt strings or PromptConfig objects
            
        Returns:
            Dict containing 'crossattn' and 'vector' tensors
        """
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)

        # Handle different prompt types
        if isinstance(prompt, list) and isinstance(prompt[0], str):
            # Convert list of strings to TextualPrompt
            prompt = [TextualPrompt(text=p) for p in prompt]
        elif not isinstance(prompt, list):
            raise ValueError("Prompt must be a list of strings or PromptConfig objects")

        cond_l_list = []
        cond_g_list = []
        clip_pooled_list = []

        for p in prompt:
            if isinstance(p, TextualPrompt):
                cond_l = self.text_processing_engine_l([p.text])
                cond_g, clip_pooled = self.text_processing_engine_g([p.text])
                width, height = p.width, p.height
                is_negative_prompt = p.is_negative_prompt
            elif isinstance(p, PromptConfig):
                cond_l = self.text_processing_engine_l([p.prompt])
                cond_g, clip_pooled = self.text_processing_engine_g([p.prompt])
                width, height = p.width, p.height
                is_negative_prompt = p.is_negative_prompt
            else:
                raise ValueError(f"Unsupported prompt type: {type(p)}")

            cond_l_list.append(cond_l)
            cond_g_list.append(cond_g)
            clip_pooled_list.append(clip_pooled)

        cond_l = torch.cat(cond_l_list, dim=0)
        cond_g = torch.cat(cond_g_list, dim=0)
        clip_pooled = torch.cat(clip_pooled_list, dim=0)

        if cond_l.dim() != 3 or cond_g.dim() != 3:
            raise ValueError("Invalid conditioning tensor dimensions")

        # Create embedding tensor directly on target device
        embedding_values = torch.tensor([
            height, width,    # Image dimensions
            0, 0,            # Crop coordinates 
            height, width    # Crop dimensions
        ], device=clip_pooled.device, dtype=clip_pooled.dtype).view(6, 1)
        
        embedded = self.embedder(embedding_values)
        flat = embedded.view(1, -1).expand(clip_pooled.shape[0], -1)

        # Streamline zeroing logic
        if is_negative_prompt and all(isinstance(p, TextualPrompt) and p.text == '' for p in prompt):
            clip_pooled = torch.zeros_like(clip_pooled)
            cond_l = torch.zeros_like(cond_l) 
            cond_g = torch.zeros_like(cond_g)

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
        # Ensure channels are in correct dimension
        if x.shape[1] != 3:
            # If channels first and RGBA, convert to RGB
            if x.shape[1] == 4:
                x = x[:, :3, :, :]
            # If channels last, move to channels first
            elif x.shape[-1] in (3, 4):
                # Take only first 3 channels if RGBA
                x = x[..., :3]
                x = x.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Unexpected channel dimension. Expected 3 or 4 channels, got shape {x.shape}")
        
        # Normalize pixel values to [0, 1] range
        x = x.movedim(1, -1)
        x = x * 0.5 + 0.5
        sample = self.forge_objects.vae.encode(x)
        return self.forge_objects.vae.first_stage_model.process_in(sample).to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        processed = self.forge_objects.vae.first_stage_model.process_out(x)
        decoded = self.forge_objects.vae.decode(processed)
        decoded = decoded.movedim(-1, 1)
        decoded.mul_(2.0).sub_(1.0)  # in-place operations
        return decoded.to(x)
