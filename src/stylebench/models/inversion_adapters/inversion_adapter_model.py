from typing import Dict, List, Union

import torch
from diffusers import StableDiffusionXLPipeline, DDIMInverseScheduler, AutoencoderKL

from ..base.base_model import BaseModel
from .inversion_adapter_config import InversionAdapterConfig


class InversionAdapterModel(BaseModel):
    def __init__(self, config: InversionAdapterConfig):
        super().__init__(config)

        self.config = config
        self.vae = AutoencoderKL.from_pretrained(
            self.config.vae_version, torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.config.sdxl_version,
            vae=self.vae,
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

        self.forward_scheduler = self.pipe.scheduler
        self.inverse_scheduler = DDIMInverseScheduler(**self.forward_scheduler.config)

    def to(self, device: Union[str, torch.device]):
        self.pipe = self.pipe.to(device)
        self.device = device
        return self

    def invert(self, batch, num_inversion_steps):
        """
        Invert an image using the DDIM inversion
        """
        # cast dtype
        image_tensor = batch[self.input_key].to(device=self.device)
        image_tensor = image_tensor.to(dtype=torch.float16)

        style_image = (image_tensor.cpu().numpy() + 1) / 2
        style_image = (style_image * 255).astype("uint8")
        style_image = style_image[0].transpose(1, 2, 0)

        posterior = self.pipe.vae.encode(image_tensor).latent_dist
        latent = posterior.mean * 0.18215

        # Change the setup
        self.pipe.scheduler = self.inverse_scheduler
        self.pipe.set_ip_adapter_scale(0)

        latent = self.pipe(
            prompt="",
            negative_prompt="",
            ip_adapter_image=style_image,
            guidance_scale=1.0,
            output_type="latent",
            return_dict=False,
            num_inference_steps=num_inversion_steps,
            latents=latent,
        )[0]

        return latent

    def sample(
        self,
        batch: Dict[str, torch.Tensor],
        prompts: Union[str, List[str]],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        num_inversion_steps: int = 30,
        noise_scale: float = 1,
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

        latent = self.invert(batch, num_inversion_steps)

        # Add random noise to latent inversion
        random_noise = torch.randn_like(latent).to(self.device, dtype=torch.float16)

        latent = latent + noise_scale * random_noise
        latent = latent / torch.sqrt(
            torch.tensor(1 + noise_scale**2).to(self.device, dtype=torch.float16)
        )

        self.pipe.scheduler = self.forward_scheduler
        self.pipe.set_ip_adapter_scale(self.config.adapter_scale)

        latents = latent.repeat(len(prompts), 1, 1, 1)

        images = self.pipe(
            prompt=prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            ip_adapter_image=style_image,
            latents=latents,
            *args,
            **kwargs
        ).images

        return images
