import os
from pathlib import Path

import pillow_avif
import pytest
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from stylebench.embedders import ClipImageEmbedderConfig
from stylebench.metrics import ClipMetric, ClipMetricConfig
from stylebench.models.inversion_adapters import InversionAdapterConfig, InversionAdapterModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATH = os.path.dirname(os.path.abspath(__file__))


class TestInversionAdapterModel:

    @pytest.fixture(
        params=[
            [1],
            [
                {
                    "up": {"block_0": [0.0, 1.0, 0.0]},
                }
            ],
            [
                {
                    "down": {"block_2": [0.0, 1.0]},
                    "up": {"block_0": [0.0, 1.0, 0.0]},
                }
            ],
        ]
    )
    def model_config(self, request):
        return InversionAdapterConfig(
            adapter_scale=request.param,
        )

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
        model = InversionAdapterModel(model_config)
        model.to(DEVICE)

        prompts = ["a dog reading a book in a lush forest"]

        reference_image_tensor = self.conditioner_input()
        reference_image_tensor = reference_image_tensor * 2 - 1
        reference_image_tensor = reference_image_tensor.to(DEVICE)

        images = model.sample(
            batch={"image": reference_image_tensor},
            prompts=prompts,
            num_inference_steps=30,
        )

        # Assert sample is closed enough to reference to make sure styling is done properly
        generated_sample = images[0]
        generated_sample = pil_to_tensor(generated_sample).unsqueeze(0) / 255.0
        generated_sample = generated_sample * 2 - 1
        generated_sample = generated_sample.to(DEVICE)

        metric = ClipMetric(metric_config)

        batch_1 = {"image": reference_image_tensor}
        batch_2 = {"image": generated_sample}

        output = metric.forward(batch_1, batch_2)

        assert torch.all(output["score"] >= 70 * torch.ones_like(output["score"]))
