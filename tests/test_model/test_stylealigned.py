import os
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from stylerank.embedders import ClipImageEmbedderConfig
from stylerank.metrics import ClipMetric, ClipMetricConfig
from stylerank.models.stylealigned import StyleAlignedConfig, StyleAlignedModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATH = os.path.dirname(os.path.abspath(__file__))


class TestStyleAlignedModel:

    @pytest.fixture
    def model_config(self):
        return StyleAlignedConfig()

    def conditioner_input(self):
        image_pil = Image.open(
            os.path.join(Path(PATH).parent, "reference_images/raccoon_1.png")
        ).convert("RGB")
        image_tensor = pil_to_tensor(image_pil).unsqueeze(0) / 255.0
        return image_tensor

    @pytest.fixture()
    def metric_config(self):
        embedder_image = ClipImageEmbedderConfig(
            version="openai/clip-vit-large-patch14",
            always_return_pooled=True,
        )
        clip_config = ClipMetricConfig(
            embedder_config_1=embedder_image,
            embedder_config_2=embedder_image,
        )

        return clip_config

    def test_model(self, model_config, metric_config):
        model = StyleAlignedModel(model_config)
        model.to(DEVICE)

        reference_image_caption = "a raccoon reading a book in a lush forest"
        prompts = "a dog reading a book in a lush forest"

        reference_image_tensor = self.conditioner_input()
        reference_image_tensor = reference_image_tensor * 2 - 1
        reference_image_tensor = reference_image_tensor.to(DEVICE)

        images = model.sample(
            batch={"image": reference_image_tensor},
            prompts=prompts,
            style_prompt=reference_image_caption,
        )

        # Assert 1st image is closed enough to reference to make sure inversion is done properly
        inverted_image = images[0]
        inverted_image = pil_to_tensor(inverted_image).unsqueeze(0) / 255.0
        inverted_image = inverted_image * 2 - 1
        inverted_image = inverted_image.to(DEVICE)

        metric = ClipMetric(metric_config)

        batch_1 = {"image": reference_image_tensor}
        batch_2 = {"image": inverted_image}

        output = metric.forward(batch_1, batch_2)

        assert torch.all(output["score"] >= 95 * torch.ones_like(output["score"]))
