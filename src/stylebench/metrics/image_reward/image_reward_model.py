from typing import Dict, List, Union

import torch
from ImageReward import RM
from PIL import Image

from ..base import BaseMetric
from .image_reward_config import ImageRewardMetricConfig


class ImageRewardMetric(BaseMetric):
    def __init__(self, config: ImageRewardMetricConfig):
        super().__init__(config)
        self.config = config

        self.model = RM.load("ImageReward-v1.0")

    def forward(
        self,
        batch_1: Dict[str, Union[str, List[str]]],
        batch_2: Dict[str, torch.Tensor],
        device: str = "cpu",
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
            image = (image + 1) / 2
            image = (image * 255).astype("uint8")
            image = image.transpose(1, 2, 0)
            image = Image.fromarray(image)

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
