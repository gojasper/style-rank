from typing import Any, Dict

import torch

from ...embedders import (
    Dinov2ImageEmbedderWithProjection,
    Dinov2ImageEmbedderWithProjectionConfig,
)
from ..base import BaseMetric
from .dinov2_metric_config import Dinov2MetricConfig


class Dinov2Metric(BaseMetric):
    """This is the Dinov2Metric class which defines all the useful methods to compute the metric

    By default : Compute the following metric: Dinov2Score
    Dinov2Score(I_1,I_2) = max(100*cos(E_1,E_2),0)

    Which corresponds to the cosine similarity between visual Dinov2 embeddings for Image 1 and Image 2.
    The Score is bound between 0 and 100 and the closer to 100 the better

    Args:

        config (Dinov2MetricConfig): An instance of the Dinov2MetricConfig class containing the parameters to configure the metric.
    """

    def __init__(self, config: Dinov2MetricConfig):
        BaseMetric.__init__(self, config)
        self.config = config

        self.use_global_token = config.use_global_token
        self.embedder = Dinov2ImageEmbedderWithProjection(config.embedder_config)

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

            batch_1 (Dict[str, Any]): The first input batch.
            batch_2 (Dict[str, Any]): The second input batch.
            device (str): The device to use. Defaults to "cpu".

        Returns:

            Dict[str, Any]: A dictionary containing the cosine similarity score between the embeddings of the two inputs.
        """

        embeddings_1 = self.embedder(batch_1, device=device)
        embeddings_2 = self.embedder(batch_2, device=device)

        if self.use_global_token:
            # The class token is the 1st token of the 256 + 1 tokens given by the .hidden_states of the embeddings
            vector_1 = embeddings_1["crossattn"][:, 0, :]
            vector_2 = embeddings_2["crossattn"][:, 0, :]
        else:
            vector_1 = embeddings_1["crossattn"].flatten(1)
            vector_2 = embeddings_2["crossattn"].flatten(1)

        cosine_tensor = torch.nn.functional.cosine_similarity(
            vector_1, vector_2, dim=-1
        )
        # Compute score
        score = 100 * torch.max(cosine_tensor, torch.zeros_like(cosine_tensor))
        output = {"score": score}

        return output
