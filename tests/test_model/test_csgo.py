import os
from pathlib import Path

import pillow_avif
import pytest
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from stylerank.embedders import ClipImageEmbedderConfig
from stylerank.metrics import ClipMetric, ClipMetricConfig
from stylerank.models.csgo import CSGOConfig, CSGOModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATH = os.path.dirname(os.path.abspath(__file__))


class TestCSGOModel:

    @pytest.fixture
    def model_config(self):
        config = CSGOConfig(
            sdxl_version="stabilityai/stable-diffusion-xl-base-1.0",
            vae_version="madebyollin/sdxl-vae-fp16-fix",
            csgo_repo_id="InstantX/CSGO",
            csgo_filename="csgo_4_32.bin",
            controlnet_repo_id="TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic",
            controlnet_filename="TTPLANET_Controlnet_Tile_realistic_v2_fp16.safetensors",
            controlnet_config_filename="config.json",
            image_encoder_repo_id="h94/IP-Adapter",
            image_encoder_filename="sdxl_models/image_encoder/model.safetensors",
            image_encoder_config_filename="sdxl_models/image_encoder/config.json",
        )
        return config

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
        model = CSGOModel(model_config, device=DEVICE)

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
