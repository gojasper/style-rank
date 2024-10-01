import os
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from stylerank.metrics import ImageRewardMetric, ImageRewardMetricConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATH = os.path.dirname(os.path.abspath(__file__))


class TestImageRewardMetric:

    @pytest.fixture()
    def metric_config(self):
        """
        Return ImageRewardMetricConfig
        """
        metric_config = ImageRewardMetricConfig()
        return metric_config

    def conditioner_input(self, path):
        image_pil = Image.open(os.path.join(Path(PATH).parent, path)).convert("RGB")
        image_tensor = pil_to_tensor(image_pil).unsqueeze(0) / 255.0
        return image_tensor

    def test_metric_forward(self, metric_config):

        metric = ImageRewardMetric(metric_config)

        # Assert the images are matching the prompt and can run multiple images at once
        prompt = "a raccoon reading a book in a lush forest"
        image_tensor = torch.cat(
            [
                self.conditioner_input("reference_images/raccoon_1.png").to(DEVICE) * 2
                - 1,
                self.conditioner_input("reference_images/raccoon_2.png").to(DEVICE) * 2
                - 1,
            ]
        )
        image_batch = {"image": image_tensor}
        text_batch = {"text": [prompt, prompt]}

        output_real = metric.forward(text_batch, image_batch, device=DEVICE)

        assert torch.all(output_real["score"] >= torch.ones_like(output_real["score"]))

        image_tensor = (
            self.conditioner_input("reference_images/astronaut.png").to(DEVICE) * 2 - 1
        )
        image_batch = {"image": image_tensor}
        text_batch = {"text": [prompt]}

        output_single = metric.forward(text_batch, image_batch, device=DEVICE)

        assert torch.all(
            output_single["score"] <= torch.zeros_like(output_single["score"])
        )

        # Assert the images are matching the prompt and can run multiple images at once
        prompt = "a raccoon reading a book in a lush forest"
        prompt_2 = "an astronaut in a jungle"
        image_tensor = torch.cat(
            [
                self.conditioner_input("reference_images/raccoon_1.png").to(DEVICE) * 2
                - 1,
                self.conditioner_input("reference_images/raccoon_2.png").to(DEVICE) * 2
                - 1,
                self.conditioner_input("reference_images/astronaut.png").to(DEVICE) * 2
                - 1,
            ]
        )
        image_batch = {"image": image_tensor}
        text_batch = {"text": [prompt, prompt, prompt_2]}

        output_real = metric.forward(text_batch, image_batch, device=DEVICE)

        assert torch.all(output_real["score"] >= torch.ones_like(output_real["score"]))
