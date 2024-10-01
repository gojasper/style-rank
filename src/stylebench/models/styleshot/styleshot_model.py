from typing import Dict, List, Union

import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download
from PIL import Image

from ..base.base_model import BaseModel
from .src.ip_adapter import StyleShot
from .styleshot_config import StyleShotConfig


class StyleShotModel(BaseModel):
    def __init__(self, config: StyleShotConfig):
        super().__init__(config)
        self.config = config
        base_model_path = config.base_model_path
        transformer_block_path = config.transformer_block_path
        device = config.device

        ip_ckpt_path = hf_hub_download(
            repo_id="Gaojunyao/StyleShot", filename="pretrained_weight/ip.bin"
        )
        style_aware_encoder_path = hf_hub_download(
            repo_id="Gaojunyao/StyleShot",
            filename="pretrained_weight/style_aware_encoder.bin",
        )

        stable_diffusion_pipe = StableDiffusionPipeline.from_pretrained(
            base_model_path, safety_checker=None
        )
        self.pipe = StyleShot(
            device,
            stable_diffusion_pipe,
            ip_ckpt_path,
            style_aware_encoder_path,
            transformer_block_path,
        )

    def to(self, device: Union[str, torch.device]):
        self.pipe = self.pipe.to(device)
        self.device = device
        return self

    def sample(
        self,
        batch: Dict[str, Image.Image],
        prompts: Union[str, List[str]],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        *args,
        **kwargs
    ):

        style_image = batch[self.input_key]
        images = self.pipe.generate(
            style_image=style_image,
            prompts=prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        images = [image[0] for image in images]
        return images
