import gradio as gr
from PIL import Image
import numpy as np
import torch
import logging
from modules import scripts, processing, shared, sd_models, devices, images
from modules.pixelsmith import PixelSmithXLPipeline, PixelSmithVAE

logger = logging.getLogger(__name__)

class Script(scripts.Script):
    def __init__(self):
        super().__init__()
        self.orig_pipe = None
        self.orig_vae = None
        self.vae = None

    def title(self):
        return 'PixelSmith'

    def show(self, is_img2img):
        return True  # Show in both txt2img and img2img

    def ui(self, _is_img2img):
        with gr.Row():
            gr.HTML('<a href="https://github.com/Thanos-DB/Pixelsmith">&nbsp PixelSmith</a><br>')
        with gr.Row():
            slider = gr.Slider(
                label="Model Freedom",
                value=20,
                minimum=0,
                maximum=100,
                step=1,
                info="Higher values allow more freedom in generation",
            )
        return [slider]

    def encode(self, p: processing.StableDiffusionProcessing, image: Image.Image):
        if image is None:
            return None
        if p.width is None or p.width == 0:
            p.width = int(8 * (image.width * p.scale_by // 8))
        if p.height is None or p.height == 0:
            p.height = int(8 * (image.height * p.scale_by // 8))
        image = images.resize_image(p.resize_mode, image, p.width, p.height, upscaler_name=p.resize_name, context=p.resize_context)
        tensor = np.array(image).astype(np.float16) / 255.0
        tensor = tensor[None].transpose(0, 3, 1, 2)
        tensor = torch.from_numpy(tensor).to(device=devices.device, dtype=devices.dtype)
        tensor = 2.0 * tensor - 1.0
        with devices.inference_context():
            latent = shared.sd_model.vae.tiled_encode(tensor)
            latent = shared.sd_model.vae.config.scaling_factor * latent.latent_dist.sample()
        logger.info(
            f"PixelSmith encode: image={image} latent={latent.shape} width={p.width} height={p.height} vae={shared.sd_model.vae.__class__.__name__}"
        )
        return latent

    def run(self, p: processing.StableDiffusionProcessing, slider: int = 20):
        # Check if model is SDXL by inspecting UNet configuration
        is_sdxl = (
            hasattr(shared.sd_model, "forge_objects")
            and hasattr(shared.sd_model.forge_objects.unet.model, "model")
            and shared.sd_model.forge_objects.unet.model.model.num_in_channels == 9
        )

        if not is_sdxl:
            logger.warning("PixelSmith: Current model is not SDXL. This script requires an SDXL model.")
            return

        self.orig_pipe = shared.sd_model
        self.orig_vae = shared.sd_model.vae

        if self.vae is None:
            self.vae = PixelSmithVAE.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", torch_dtype=devices.dtype
            ).to(devices.device)

        shared.sd_model = sd_models.switch_pipe(PixelSmithXLPipeline, shared.sd_model)
        shared.sd_model.vae = self.vae
        shared.sd_model.vae.enable_tiling()

        p.extra_generation_params["PixelSmith"] = f"Freedom={slider}"
        p.sampler_name = "DDIM"
        p.task_args["slider"] = slider

        if hasattr(p, "init_images") and p.init_images and len(p.init_images) > 0:
            p.task_args["image"] = self.encode(p, p.init_images[0])
            p.init_images = None

        logger.info(
            f"PixelSmith apply: freedom={slider} class={shared.sd_model.__class__.__name__} vae={shared.sd_model.vae.__class__.__name__}"
        )

        return processing.process_images(p)

    def after(self, p, processed, slider):
        if self.orig_pipe is None:
            return processed
        if shared.sd_model.__class__.__name__ == "PixelSmithXLPipeline":
            shared.sd_model = self.orig_pipe
            shared.sd_model.vae = self.orig_vae
        self.orig_pipe = None
        self.orig_vae = None
        return processed
