from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from typing import Optional, Union, List
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from diffusers.models.attention_processor import AttnProcessor2_0
from dataclasses import dataclass


class PixelSmithVAE(AutoencoderKL):
    """Custom VAE implementation with tiled operations."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_tiling()
        
    def enable_tiling(self):
        """Enable tiled operations for memory efficiency."""
        self.use_tiling = True
        self.tile_size = 512
        self.tile_stride = 512
        
    def tiled_encode(self, x):
        # Handle encoding with tiling
        b, c, h, w = x.shape

        # Use standard encode if small enough
        if h <= self.tile_size and w <= self.tile_size:
            return self.encode(x)

        # Calculate number of tiles needed
        n_h = (h - 1) // self.tile_stride + 1
        n_w = (w - 1) // self.tile_stride + 1

        # Process tiles
        latents = []
        for i in range(n_h):
            row = []
            for j in range(n_w):
                # Extract tile
                start_h = i * self.tile_stride
                start_w = j * self.tile_stride
                end_h = min(start_h + self.tile_size, h)
                end_w = min(start_w + self.tile_size, w)

                tile = x[:, :, start_h:end_h, start_w:end_w]
                with torch.no_grad():
                    tile_latent = self.encode(tile)
                row.append(tile_latent)
            latents.append(row)

        return self._extracted_from_tiled_decode_32(latents)
        
    def tiled_decode(self, latents, return_dict=False):
        # Handle decoding with tiling
        b, c, h, w = latents.shape

        # Use standard decode if small enough
        if h <= self.tile_size and w <= self.tile_size:
            return self.decode(latents, return_dict=return_dict)

        # Calculate tiles needed
        n_h = (h - 1) // self.tile_stride + 1
        n_w = (w - 1) // self.tile_stride + 1

        # Process tiles
        decoded = []
        for i in range(n_h):
            row = []
            for j in range(n_w):
                # Extract tile
                start_h = i * self.tile_stride
                start_w = j * self.tile_stride
                end_h = min(start_h + self.tile_size, h)
                end_w = min(start_w + self.tile_size, w)

                tile = latents[:, :, start_h:end_h, start_w:end_w]
                with torch.no_grad():
                    tile_decoded = self.decode(tile, return_dict=return_dict)
                if isinstance(tile_decoded, tuple):
                    tile_decoded = tile_decoded[0]
                row.append(tile_decoded)
            decoded.append(row)

        final_decoded = self._extracted_from_tiled_decode_32(decoded)
        return (final_decoded,) if return_dict else final_decoded

    def _extracted_from_tiled_decode_32(self, arg0):
        rows = []
        rows.extend(torch.cat(row, dim=3) for row in arg0)
        return torch.cat(rows, dim=2)


class PixelSmithXLPipeline(StableDiffusionXLPipeline):
    """Pipeline for PixelSmith image generation."""
    
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
