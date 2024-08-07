from typing import Any, Dict

import torch

from ...embedders.clip import (
    ClipEmbedderWithProjection,
    ClipImageEmbedderWithProjection,
)
from ..base import BaseMetric
from .clip_metric_config import ClipMetricConfig


class ClipMetric(BaseMetric):
    """This is the ClipMetric class which defines all the useful methods to compute the metric

    By default : Compute the following metric: CLIPScore
    CLIPScore(I,T) = max(100*cos(E_I,E_T),0)

    Which corresponds to the cosine similarity between visual CLIP embedding for Image and textual CLIP embeddings for a given caption.
    The score is bound between 0 and 100 and the closer to 100 the better

    To change the metric to compute Clip Image x Image or Text x Text, change the ClipMetricConfig

    Args:

        config (ClipMetricConfig): An instance of the ClipMetricConfig class containing the parameters to configure the metric.
    """

    def __init__(self, config: ClipMetricConfig):
        BaseMetric.__init__(self, config)
        self.config = config

        # Load embedders (Default is Image X Text)
        if self.config.input_key_1 == "text":
            self.embedder_1 = ClipEmbedderWithProjection(config.embedder_config_1)
        if self.config.input_key_1 == "image":
            self.embedder_1 = ClipImageEmbedderWithProjection(config.embedder_config_1)

        if self.config.input_key_2 == "text":
            self.embedder_2 = ClipEmbedderWithProjection(config.embedder_config_2)
        if self.config.input_key_2 == "image":
            self.embedder_2 = ClipImageEmbedderWithProjection(config.embedder_config_2)

    def forward(
        self,
        batch_1: Dict[str, Any],
        batch_2: Dict[str, Any],
        device: str = "cpu",
        *args,
        **kwargs
    ):
        """Computes the cosine sim between the embeddings of two inputs

        Args:

            batch_1 (Dict[str, Any]): The first input batch. If key "vector" is present, it will be used as the embeddings.
            batch_2 (Dict[str, Any]): The second input batch. If key "vector" is present, it will be used as the embeddings.
            device (str): The device to use. Defaults to "cpu".

        Returns:

            Dict[str, torch.Tensor]: The computed metric for each pair of inputs.
        """

        # Check if embeddings are already computed
        # this logic can be abtracted to a more generic function for all metrics I think
        if "vector" in batch_1:
            embeddings_1 = batch_1["vector"]
        else:
            embeddings_1 = self.embedder_1(batch_1, device=device)["vector"]
        if "vector" in batch_2:
            embeddings_2 = batch_2["vector"]
        else:
            embeddings_2 = self.embedder_2(batch_2, device=device)["vector"]

        cosine_tensor = torch.nn.functional.cosine_similarity(
            embeddings_1, embeddings_2, dim=-1
        )
        score = 100 * cosine_tensor

        output = {"score": score}

        return output
