import os
from pathlib import Path

import pytest
import torch
from PIL import Image
from pydantic import ValidationError
from torchvision.transforms.functional import pil_to_tensor

from stylebench.embedders import Dinov2ImageEmbedderWithProjectionConfig
from stylebench.metrics import Dinov2Metric, Dinov2MetricConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATH = os.path.dirname(os.path.abspath(__file__))


class TestDinov2Metric:

    def test_wrong_config(self):
        with pytest.raises(ValidationError):
            Dinov2MetricConfig(
                embedder_config=Dinov2ImageEmbedderWithProjectionConfig(
                    input_key="thisIsNotAValidKey"
                )
            )

    @pytest.fixture(params=[True, False])
    def do_processor_test(self, request):
        return request.param

    @pytest.fixture(
        params=[
            # version for embedder and use_global_token
            ("facebook/dinov2-small", True),
            ("facebook/dinov2-base", True),
            ("facebook/dinov2-large", True),
            ("facebook/dinov2-giant", True),
            ("facebook/dinov2-small", False),
            ("facebook/dinov2-base", False),
            ("facebook/dinov2-large", False),
            ("facebook/dinov2-giant", False),
        ]
    )
    def metric_config(self, request, do_processor_test):
        """
        Return Dinov2MetricConfig
        """
        embedder_config = Dinov2ImageEmbedderWithProjectionConfig(
            version=request.param[0],
            do_processor=do_processor_test,
        )
        return Dinov2MetricConfig(
            embedder_config=embedder_config, use_global_token=request.param[1]
        )

    def conditioner_input_1(self):
        image_pil = Image.open(
            os.path.join(Path(PATH).parent, "reference_images/raccoon_1.png")
        ).convert("RGB")
        image_tensor = pil_to_tensor(image_pil).unsqueeze(0) / 255.0
        return image_tensor

    def conditioner_input_2(self):
        image_pil = Image.open(
            os.path.join(Path(PATH).parent, "reference_images/raccoon_2.png")
        ).convert("RGB")
        image_tensor = pil_to_tensor(image_pil).unsqueeze(0) / 255.0
        return image_tensor

    def test_metric_forward(self, metric_config, do_processor_test):

        metric = Dinov2Metric(metric_config)

        if do_processor_test:
            # compute metric for two real images which are both of a racoon
            # the images don't have the same size (1024x1024) vs (512x512)
            # Compute ONLY using the pre-processor
            # Change call method to avoid OOM cuda

            conditioner_input_1 = {
                "image": self.conditioner_input_1().to(DEVICE) * 2 - 1
            }
            conditioner_input_2 = {
                "image": self.conditioner_input_2().to(DEVICE) * 2 - 1
            }

            output_real = metric.forward(
                conditioner_input_1, conditioner_input_2, device=DEVICE
            )
            assert torch.all(
                output_real["score"] >= torch.zeros_like(output_real["score"])
            )
            assert torch.all(
                output_real["score"] <= 100 * torch.ones_like(output_real["score"])
            )

        # now compute for a batch of random images
        image_tensor_1 = torch.rand(10, 3, 224, 224)
        image_tensor_2 = torch.rand(10, 3, 224, 224)

        if do_processor_test:
            batch_1 = {"image": image_tensor_1.to(DEVICE) * 2 - 1}
            batch_2 = {"image": image_tensor_2.to(DEVICE) * 2 - 1}
        else:
            batch_1 = {"image": image_tensor_1.to(DEVICE)}
            batch_2 = {"image": image_tensor_2.to(DEVICE)}

        output = metric.forward(batch_1, batch_2, device=DEVICE)
        assert torch.all(output["score"] >= torch.zeros_like(output["score"]))
        assert torch.all(output["score"] <= 100 * torch.ones_like(output["score"]))

        # assert we can match 1 image with a batch of 10 images
        if do_processor_test:
            single_image_batch = {
                "image": torch.rand(1, 3, 224, 224).to(DEVICE) * 2 - 1
            }
        else:
            single_image_batch = {"image": torch.rand(1, 3, 224, 224).to(DEVICE)}

        output = metric.forward(single_image_batch, batch_2, device=DEVICE)
        assert output["score"].shape == (10,)
        assert torch.all(output["score"] >= torch.zeros_like(output["score"]))
        assert torch.all(output["score"] <= 100 * torch.ones_like(output["score"]))

        # assert we can match 10 images with 1 image reverse order
        output = metric.forward(batch_1, single_image_batch, device=DEVICE)
        assert output["score"].shape == (10,)
        assert torch.all(output["score"] >= torch.zeros_like(output["score"]))
        assert torch.all(output["score"] <= 100 * torch.ones_like(output["score"]))

        if do_processor_test:
            # assert random noise X racoon < racoon X racoon
            # usefull only using processor
            # Compute the score again
            output = metric.forward(conditioner_input_1, batch_2, device=DEVICE)

            assert output_real["score"] > output["score"].mean()
            assert output_real["score"] > output["score"].mean()

        # assert image1 x image1 == 100
        output = metric.forward(batch_1, batch_1, device=DEVICE)
        assert torch.allclose(output["score"], 100 * torch.ones_like(output["score"]))
