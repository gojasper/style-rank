import pytest
import torch
from pydantic import ValidationError
from torchmetrics.multimodal.clip_score import CLIPScore

from stylebench.embedders import ClipEmbedderConfig, ClipImageEmbedderConfig
from stylebench.metrics import ClipMetric, ClipMetricConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestClipMetric:

    def test_wrong_config(self):
        # Check we cannot have always_return_pooled as False
        with pytest.raises(ValidationError):
            embedder_image = ClipImageEmbedderConfig(always_return_pooled=False)
            embedder_text = ClipEmbedderConfig(always_return_pooled=True)
            ClipMetricConfig(
                embedder_config_1=embedder_image,
                embedder_config_2=embedder_text,
            )

        with pytest.raises(ValidationError):
            embedder_image = ClipImageEmbedderConfig(always_return_pooled=True)
            embedder_text = ClipEmbedderConfig(always_return_pooled=False)
            ClipMetricConfig(
                embedder_config_1=embedder_image,
                embedder_config_2=embedder_text,
            )

        with pytest.raises(ValidationError):
            embedder_image = ClipImageEmbedderConfig(
                input_key="thisIsNotAValidKey", always_return_pooled=False
            )
            embedder_text = ClipEmbedderConfig(always_return_pooled=False)
            ClipMetricConfig(
                embedder_config_1=embedder_image,
                embedder_config_2=embedder_text,
            )

        with pytest.raises(ValidationError):
            embedder_image = ClipImageEmbedderConfig(always_return_pooled=False)
            embedder_text = ClipEmbedderConfig(
                always_return_pooled=False,
                input_key="thisIsNotAValidKey",
            )
            ClipMetricConfig(
                embedder_config_1=embedder_image,
                embedder_config_2=embedder_text,
            )

    @pytest.fixture(
        params=[
            [
                "image",
                "openai/clip-vit-large-patch14",
            ],
            [
                "text",
                "openai/clip-vit-large-patch14",
            ],
            [
                "image-text",
                "openai/clip-vit-large-patch14",
            ],
            [
                "image-text",
                "openai/clip-vit-large-patch14-336",
            ],
            [
                "text-image",
                "openai/clip-vit-large-patch14",
            ],
            [
                "image",
                "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
            ],
        ]
    )
    def metric_config(self, request):
        """
        Define ClipMetricConfig
        """
        image_config = ClipImageEmbedderConfig(
            version=request.param[1],
            always_return_pooled=True,
        )
        text_config = ClipEmbedderConfig(
            version=request.param[1],
            always_return_pooled=True,
        )
        if request.param[0] == "image":
            clip_config = ClipMetricConfig(
                embedder_config_1=image_config,
                embedder_config_2=image_config,
            )
        elif request.param[0] == "text":
            clip_config = ClipMetricConfig(
                embedder_config_1=text_config,
                embedder_config_2=text_config,
            )
        elif request.param[0] == "image-text":
            clip_config = ClipMetricConfig(
                embedder_config_1=image_config,
                embedder_config_2=text_config,
            )
        elif request.param[0] == "text-image":
            clip_config = ClipMetricConfig(
                embedder_config_1=text_config,
                embedder_config_2=image_config,
            )
        return clip_config

    def test_metrics_forward(
        self,
        metric_config,
    ):

        if metric_config.input_key_1 == "image":
            # input is assumed to be in [-1, 1]
            image_tensor = torch.rand(10, 3, 224, 224).to(DEVICE)
            image_tensor = 2 * image_tensor - 1
            batch_1 = {"image": image_tensor}

        if metric_config.input_key_1 == "text":
            text_1 = "a photo of a cat"
            text_2 = "a photo of a dog"
            text_tensor = [text_1 if i % 2 == 0 else text_2 for i in range(10)]
            batch_1 = {"text": text_tensor}

        if metric_config.input_key_2 == "image":
            # input is assumed to be in [-1, 1]
            image_tensor = torch.rand(10, 3, 224, 224).to(DEVICE)
            image_tensor = 2 * image_tensor - 1
            batch_2 = {"image": image_tensor}

        if metric_config.input_key_2 == "text":
            text_1 = "a photo of a cat"
            text_2 = "a photo of a dog"
            text_tensor = [text_1 if i % 2 == 0 else text_2 for i in range(10)]
            batch_2 = {"text": text_tensor}

        clip_metric = ClipMetric(metric_config)
        output = clip_metric(batch_1, batch_2, device=DEVICE)

        # assert scores are in range 0,100
        assert torch.all(output["score"] >= torch.zeros_like(output["score"]))
        assert torch.all(
            output["score"] <= 100.0 * torch.ones_like(output["score"]) + 1e-4
        )

        if metric_config.input_key_1 != metric_config.input_key_2:
            torchmetrics_clip = CLIPScore(
                model_name_or_path=metric_config.embedder_config_1.version,
            )
            image_tensor_cpu = (image_tensor.to("cpu") + 1) / 2 * 255
            image_tensor_int = image_tensor_cpu.int()
            score = torchmetrics_clip(image_tensor_int, text_tensor)
            avg_score = output["score"].detach().mean(0)

            torch.testing.assert_close(
                avg_score,
                score,
                rtol=0,
                atol=1e-4,
                check_device=False,
                check_dtype=False,
                check_layout=False,
            ), (
                avg_score,
                score,
            )
