import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
)
from huggingface_hub import hf_hub_download
from PIL import Image

from stylerank.models.csgo.src.ip_adapter import CSGO

from ..base.base_model import BaseModel
from .csgo_config import CSGOConfig


class CSGOModel(BaseModel):
    def __init__(
        self, config: CSGOConfig, device: Optional[Union[str, torch.device]] = None
    ):
        super().__init__(config)
        self.config = config

        # download CSGO weights
        csgo_path = hf_hub_download(
            repo_id=config.csgo_repo_id,
            filename=config.csgo_filename,
        )

        # download controlnet weights
        hf_hub_download(
            repo_id=config.controlnet_repo_id,
            filename=config.controlnet_filename,
            force_filename="models--TTPlanet--TTPLanet_SDXL_Controlnet_Tile_Realistic/diffusion_pytorch_model.safetensors",
        )
        controlnet_config_path = hf_hub_download(
            repo_id=config.controlnet_repo_id,
            filename=config.controlnet_config_filename,
            force_filename="models--TTPlanet--TTPLanet_SDXL_Controlnet_Tile_Realistic/config.json",
        )
        controlnet_path = os.path.dirname(controlnet_config_path)

        # download image encoder weights
        hf_hub_download(
            repo_id=config.image_encoder_repo_id, filename=config.image_encoder_filename
        )
        image_encoder_path = hf_hub_download(
            repo_id=config.image_encoder_repo_id,
            filename=config.image_encoder_config_filename,
        )
        image_encoder_path = os.path.dirname(image_encoder_path)

        # load models
        self.vae = AutoencoderKL.from_pretrained(
            self.config.vae_version, torch_dtype=torch.float16
        )
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_path, torch_dtype=torch.float16, use_safetensors=True
        )

        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.config.sdxl_version,
            vae=self.vae,
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            add_watermarker=False,
        )
        self.pipe.enable_vae_tiling()

        self.target_content_blocks = self.config.target_content_blocks
        self.target_style_blocks = self.config.target_style_blocks
        self.controlnet_target_content_blocks = (
            self.config.controlnet_target_content_blocks
        )
        self.controlnet_target_style_blocks = self.config.controlnet_target_style_blocks

        if device is None:
            self.device = "cpu"
        else:
            self.device = device

        self.csgo = CSGO(
            self.pipe,
            image_encoder_path,
            csgo_path,
            self.device,
            num_content_tokens=4,
            num_style_tokens=32,
            target_content_blocks=self.target_content_blocks,
            target_style_blocks=self.target_style_blocks,
            controlnet_adapter=True,
            controlnet_target_content_blocks=self.controlnet_target_content_blocks,
            controlnet_target_style_blocks=self.controlnet_target_style_blocks,
            content_model_resampler=True,
            style_model_resampler=True,
        )

    def to(self, device: Union[str, torch.device]):
        self.pipe = self.csgo.to(device)
        self.device = device
        return self

    def sample(
        self,
        batch: Dict[str, Any],
        prompts: Union[str, List[str]],
        num_inference_steps: int = 50,
        guidance_scale: float = 10.0,
        controlnet_conditioning_scale: float = 0.01,
        *args,
        **kwargs
    ) -> List[Image.Image]:

        negative_prompt = "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry"

        if isinstance(prompts, str):
            prompts = [prompts]
            negative_prompts = [negative_prompt]
        else:
            negative_prompts = [negative_prompt] * len(prompts)

        style_image = batch[self.input_key].cpu().numpy()
        style_image = (style_image + 1) / 2
        style_image = (style_image * 255).astype("uint8")
        style_image = style_image[0].transpose(1, 2, 0)

        content_image = Image.fromarray(
            np.zeros((1024, 1024, 3), dtype=np.uint8)
        ).convert("RGB")

        images = []

        for prompt, negative_prompt in zip(prompts, negative_prompts):

            # Text Driven + Style Image generation
            image = self.csgo.generate(
                pil_content_image=content_image,
                pil_style_image=style_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                content_scale=1.0,
                style_scale=1.0,
                guidance_scale=guidance_scale,
                num_images_per_prompt=1,
                num_samples=1,
                num_inference_steps=num_inference_steps,
                seed=42,
                image=content_image,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                *args,
                **kwargs
            )[0]

            images.append(image)

        return images
