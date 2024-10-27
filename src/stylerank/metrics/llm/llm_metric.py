from typing import Dict, List, Union

import torch
from PIL import Image
import openai
import base64
import io

from ..base import BaseMetric
from .llm_metric_config import LlmMetricConfig


def encode_image(image_array):
   """
   Encode a numpy array image to base64 string.
   
   Args:
       image_array: Numpy array representing an image (H,W,C) with values 0-255
       
   Returns:
       base64 encoded string of the image
   """
   # Convert numpy array to PIL Image
   image = Image.fromarray(image_array.astype('uint8'))
   
   # Create bytes buffer
   buffer = io.BytesIO()
   
   # Save image to buffer in PNG format
   image.save(buffer, format='PNG')
   
   # Encode buffer to base64
   img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
   
   buffer.close()
   
   return img_str


class LlmMetric(BaseMetric):
    def __init__(self, config: LlmMetricConfig):
        super().__init__(config)
        self.config = config
        self.client = openai.OpenAI()

    def forward(
        self,
        batch_1: Dict[str, Union[str, List[str]]],
        batch_2: Dict[str, torch.Tensor],
        *args,
        **kwargs,
    ):
        assert (
            len(batch_1[self.config.input_key_1])
            == batch_2[self.config.input_key_2].shape[0]
        ), (
            f"Batch size mismatch: {len(batch_1[self.config.input_key_1])} prompts and "
            f"{batch_2[self.config.input_key_2].shape[0]} images"
        )

        batches = {}
        indexes = {}

        for i, (prompt, image) in enumerate(
            zip(batch_1[self.config.input_key_1], batch_2[self.config.input_key_2])
        ):
            # Transform tensor image into PIL image
            image = image.cpu().detach().numpy()
            base64_image = encode_image(image)
            
            # if prompt key is missing create a list with current image
            if prompt not in batches:
                batches[prompt] = [image]
                indexes[prompt] = [i]
            else:
                batches[prompt].append(image)
                indexes[prompt].append(i)

        rewards = {}
        for prompt, images in batches.items():
            score = self.model.score(prompt, images)
            rewards[prompt] = score if isinstance(score, list) else [score]

        score = torch.zeros(len(batch_1[self.config.input_key_1]), device=device)
        for prompt, reward_list in rewards.items():
            for index, reward in zip(indexes[prompt], reward_list):
                score[index] = reward

        output = {"score": score}
        return output
