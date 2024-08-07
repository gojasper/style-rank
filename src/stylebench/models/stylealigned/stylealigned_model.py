from typing import Any, Dict, List, Optional, Union

import torch
from diffusers import DDIMScheduler, StableDiffusionXLPipeline

from stylebench.models.stylealigned.src.inversion import (
    ddim_inversion,
    make_inversion_callback,
)
from stylebench.models.stylealigned.src.sa_handler import Handler

from ..base.base_model import BaseModel
from .stylealigned_config import StyleAlignedConfig


class StyleAlignedModel(BaseModel):
    def __init__(self, config: StyleAlignedConfig):
        super().__init__(config)
        self.config = config
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            scheduler=DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
            ),
        )
        self.pipe.enable_vae_tiling()

        self.handler = Handler(self.pipe)
        self.handler.register(self.config.args)
        self.device: Union[str, torch.device] = "cpu"

    def to(self, device: Union[str, torch.device]):
        self.pipe = self.pipe.to(device)
        self.device = device
        return self

    def invert(
        self,
        batch: Dict[str, torch.Tensor],
        prompt: str,
        num_inference_steps: int = 30,
        guidance_scale: float = 2.0,
    ):
        """Invert an image using the StyleAligned oringial DDIM implementation

        Args:
            batch (Dict[str, torch.Tensor]): A batch of 1 image in the range [-1, 1]
            prompt (str): The prompt guide the inversion
            num_inference_steps (int): The number of inference steps  for the inversion
            guidance_scale (float): The guidance scale for the inversion

        Returns:
            torch.Tensor: The inverted latent trajectory (zN, zN-1, ..., z0)

        """

        assert batch[self.input_key].shape[0] == 1, "Batch size should be 1"

        x0 = batch[self.input_key].cpu().numpy()
        x0 = (x0 + 1) / 2
        x0 = (x0 * 255).astype("uint8")
        x0 = x0[0].transpose(1, 2, 0)

        return ddim_inversion(
            self.pipe, x0, prompt, num_inference_steps, guidance_scale
        )

    def sample(
        self,
        batch: Dict[str, Any],
        prompts: Union[str, List[str]],
        style_prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 10.0,
        num_inversion_steps: int = 50,
        inversion_guidance_scale: float = 2.0,
        offset: int = 5,
        *args,
        **kwargs
    ):
        """Sample a batch of images from a reference style image using the StyleAligned implementation

        Args:
            batch (Dict[str, Any]): Reference to generate samples from
            prompts (List[str]): A list of prompts to guide the sampling
            style_prompt (str): The prompt to guide the style (here used of inversion)
            num_inference_steps (int): The number of inference steps
            guidance_scale (float): The guidance scale for the generation
            num_inversion_steps (int): The number of inversion steps
            inversion_guidance_scale (float): The guidance scale for the inversion
            offset (int): The offset for the inversion

        Returns:
            images List[PIL.Image]: A list of generated images
        """

        assert (
            batch[self.input_key].shape[0] == 1
        ), "Batch size should be 1, only 1 style image can be processed at a time"

        # Invert the image
        zts = self.invert(
            batch, style_prompt, num_inversion_steps, inversion_guidance_scale
        )
        zT, inversion_callback = make_inversion_callback(zts, offset=offset)

        if isinstance(prompts, str):
            prompts = [prompts]

        prompts = [style_prompt] + prompts

        latents = torch.randn(
            len(prompts), 4, 128, 128, device="cpu", dtype=self.pipe.unet.dtype
        ).to(self.device)
        latents[0] = zT

        images = self.pipe(
            prompts,
            latents=latents,
            callback_on_step_end=inversion_callback,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            *args,
            **kwargs,
        ).images

        return images
