import os
from pathlib import Path

import pillow_avif
import pytest
import torch
from PIL import Image

from stylebench.models.styleshot import StyleShotConfig, StyleShotModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATH = os.path.dirname(os.path.abspath(__file__))


class TestStyleShotModel:

    @pytest.fixture()
    def model_config(self):
        return StyleShotConfig(device="cuda")

    def conditioner_input(self):
        image_pil = Image.open(
            os.path.join(Path(PATH).parent, "reference_images/raccoon_1.png")
        ).convert("RGB")
        return image_pil

    def test_model(self, model_config):
        model = StyleShotModel(model_config)
        model.to(DEVICE)

        prompts = ["a dog reading a book in a lush forest"]

        reference_image = self.conditioner_input()

        images = model.sample(
            batch={"image": reference_image},
            prompts=prompts,
            num_inference_steps=30,
        )

        generated_sample = images[0]

        assert generated_sample is not None
        assert generated_sample.size == (512, 512)
