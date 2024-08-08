from typing import Any, Dict, List, Optional, Union

import torch

from ..base.base_model import BaseModel
from .src.pipelines.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from .visualstyle_config import VisualStyleConfig


class VisualStyleModel(BaseModel):

    def __init__(self, config: VisualStyleConfig):
        super().__init__(config)
        self.config = config

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            use_safetensors=True,
        )

        self.pipe.activate_layer(
            activate_layer_indices=self.config.activate_layer_indices,
            attn_map_save_steps=self.config.attn_map_save_steps,
            activate_step_indices=self.config.activate_step_indices,
            use_shared_attention=self.config.use_shared_attention,
        )

        self.device = "cpu"

    def to(self, device: Union[str, torch.device]):
        self.pipe = self.pipe.to(device)
        self.device = device
        return self

    def sample(
        self,
        batch: Dict[str, torch.Tensor],
        prompts: Union[str, List[str]],
        style_prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        *args,
        **kwargs
    ):
        """Sample a batch of images from a reference style image using the VisualStyle implementation

        Args:
            batch (Dict[str, Any]): Reference to generate samples from
            prompts (List[str]): A list of prompts to guide the sampling
            style_prompt (str): The prompt to guide the style
            num_inference_steps (int): The number of inference steps
            guidance_scale (float): The guidance scale for the generation

        Returns:
           images List[PIL.Image]: A list of generated images
        """

        assert (
            batch[self.input_key].shape[0] == 1
        ), "Batch size should be 1, only 1 style image can be processed at a time"

        style_image = batch[self.input_key]
        # Convert the image back to PIL format
        style_image = (style_image + 1) / 2
        style_image = (style_image * 255).cpu().numpy().astype("uint8")
        style_image = style_image[0].transpose(1, 2, 0)

        latents = torch.randn([2, 4, 128, 128], device="cpu").to(self.device)

        outputs = []

        if isinstance(prompts, str):
            # VisualStyle doesn't support batching
            prompts = [prompts]

        for prompt in prompts:

            images = self.pipe(
                prompt=style_prompt,  # The style ref caption
                latents=latents,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                image=style_image,
                target_prompt=prompt,
                num_images_per_prompt=2,
                use_inf_negative_prompt=self.config.use_inf_negative_prompt,
                use_advanced_sampling=self.config.use_advanced_sampling,
                use_prompt_as_null=self.config.use_prompt_as_null,
                *args,
                **kwargs,
            )[0]

            generated_image = images[1]

            outputs.append(generated_image)

        return outputs
