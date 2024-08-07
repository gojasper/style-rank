from pydantic.dataclasses import dataclass

from ...embedders import Dinov2ImageEmbedderWithProjectionConfig
from ..base.base_metric_config import BaseMetricConfig


@dataclass
class Dinov2MetricConfig(BaseMetricConfig):
    """This is the Dinov2MetricConfig class which defines all the useful parameters to instantiate the metric

    Dinov2Score(I_1,I_2) = max(100*cos(E_1,E_2),0)

    Which corresponds to the cosine similarity between visual Dinov2 embeddings for Image 1 and Image 2.
    The Score is bound between 0 and 100 and the closer to 100 the better

    Args:

        embedder_config (Dinov2ImageEmbedderWithProjectionConfig): The config of the embedder to use for the input. Default to Dinov2ImageEmbedderWithProjectionConfig(always_return_pooled=True)
        use_global_token (bool): Whether to use only the global token for the embeddings or all patches. Defaults to False.
    """

    embedder_config: Dinov2ImageEmbedderWithProjectionConfig = (
        Dinov2ImageEmbedderWithProjectionConfig()
    )
    use_global_token: bool = False

    def __post_init__(self):
        super().__post_init__()

        self.input_key = self.embedder_config.input_key

        assert (
            self.input_key == "image"
        ), f"Expected input_key to be 'image' for embedder, got {self.input_key}"
