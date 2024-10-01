from typing import Union

from pydantic.dataclasses import dataclass

from ...embedders.clip import ClipEmbedderConfig, ClipImageEmbedderConfig
from ..base.base_metric_config import BaseMetricConfig


@dataclass
class ClipMetricConfig(BaseMetricConfig):
    """This is the ClipMetricConfig class which defines all the useful parameters to instantiate the metric

    By default : Compute the following metric: CLIPScore
    CLIPScore(I,T) = max(100*cos(E_I,E_T),0)

    Which corresponds to the cosine similarity between visual CLIP embedding for Image and textual CLIP embeddings for a given caption.
    The score is bound between 0 and 100 and the closer to 100 the better

    Can be overriden to compute Clip Image x Image or Text x Text by changing the embedder configs

    Args:

        embedder_config_1 (Union[ClipImageEmbedderConfig, ClipEmbedderConfig]): The config of the first embedder to use for the 1st input. Default to ClipImageEmbedderConfig(always_return_pooled=True, do_rescale=True)
        embedder_config_2 (Union[ClipImageEmbedderConfig, ClipEmbedderConfig]): The config of the second embedder to use for the 2nd input. Default to ClipEmbedderConfig(always_return_pooled=True)
    """

    embedder_config_1: Union[ClipImageEmbedderConfig, ClipEmbedderConfig] = (
        ClipImageEmbedderConfig(always_return_pooled=True)
    )
    embedder_config_2: Union[ClipImageEmbedderConfig, ClipEmbedderConfig] = (
        ClipEmbedderConfig(always_return_pooled=True)
    )

    def __post_init__(self):
        super().__post_init__()

        self.input_key_1 = self.embedder_config_1.input_key
        self.input_key_2 = self.embedder_config_2.input_key

        # for Text-Text or Image-Image
        if self.input_key_1 == self.input_key_2:
            self.input_key = self.input_key_1

        assert self.input_key_1 in [
            "text",
            "image",
        ], f"Expected input_key to be either 'text' or 'image' for embedder 1, got {self.input_key_1}"

        assert self.input_key_2 in [
            "text",
            "image",
        ], f"Expected input_key to be either 'text' or 'image' for embedder 2, got {self.input_key_2}"

        assert (
            self.embedder_config_1.always_return_pooled
        ), "Expected always_return_pooled to be True for embedder 1"

        assert (
            self.embedder_config_2.always_return_pooled
        ), "Expected always_return_pooled to be True for embedder 2"
