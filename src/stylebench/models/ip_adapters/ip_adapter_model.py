from typing import Any, Dict, List, Optional, Union

import torch
from diffusers import StableDiffusionXLPipeline

from ..base.base_model import BaseModel
from .ip_adapter_config import IPAdapterConfig


class IPAdapterModel(BaseModel):
    def __init__(self, config: IPAdapterConfig):
        super().__init__(config)

        self.config = config
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        self.pipe.load_ip_adapter(
            config.version,
            subfolder=config.subfolder,
            weight_name=config.weight_name,
        )

        self.pipe.set_ip_adapter_scale(config.adapter_scale)

    def to(self, device: Union[str, torch.device]):
        self.pipe = self.pipe.to(device)
        self.device = device
        return self

    def sample(
        self,
        batch: Dict[str, torch.Tensor],
        prompts: Union[str, List[str]],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        *args,
        **kwargs
    ):
        """Sample a batch of images from a reference style image using the IP-Adapter implementation

        Args:
            batch (Dict[str, Any]): Reference to generate samples from
            prompts (List[str]): A list of prompts to guide the sampling
            num_inference_steps (int): The number of inference steps
            guidance_scale (float): The guidance scale for the generation

        Returns:
           images List[PIL.Image]: A list of generated images
        """

        style_image = batch[self.input_key].cpu().numpy()
        style_image = (style_image + 1) / 2
        style_image = (style_image * 255).astype("uint8")
        style_image = style_image[0].transpose(1, 2, 0)

        images = self.pipe(
            prompt=prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            ip_adapter_image=style_image,
            *args,
            **kwargs
        ).images

        return images
