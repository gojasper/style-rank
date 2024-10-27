from pydantic.dataclasses import dataclass

from ..base.base_metric_config import BaseMetricConfig


@dataclass
class LlmMetricConfig(BaseMetricConfig):
    """This is the Llm Metric Config.  This method uses an LLM as a judge to use vision models
    to evaluate the style and content quality of a data transfer

    Args:

        embedder_config (Dinov2ImageEmbedderWithProjectionConfig): The config of the embedder to use for the input. Default to Dinov2ImageEmbedderWithProjectionConfig(always_return_pooled=True)
        use_global_token (bool): Whether to use only the global token for the embeddings or all patches. Defaults to False.
    """


    def __post_init__(self):
        super().__post_init__()