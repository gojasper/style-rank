from typing import Literal, Optional

from pydantic.dataclasses import dataclass

from ..base import BaseConditionerConfig


@dataclass
class Dinov2ImageEmbedderWithProjectionConfig(BaseConditionerConfig):
    """This is the Dinov2ImageEmbedderWithProjectionConfig class which defines all the useful parameters to instantiate the model

    Args:
        version (Literal["facebook/dinov2-small", "facebook/dinov2-base", "facebook/dinov2-large", "facebook/dinov2-giant"], optional)
            The version of the model on HF Hub. Defaults to "facebook/dinov2-small".
        input_key (str, optional): The key for the input. Defaults to "image".
        layer (Literal["last", "pooled"], optional): The layer to return. Defaults to "last".
        freeze_backbone (bool): Whether to freeze the backbone. Defaults to True.
        always_return_pooled (bool): Whether to always return the pooled output. Defaults to True.
        do_processor (bool): Whether to use the integrated pre-processor (resize, center-crop, normalize...). Defaults to True. When set to False, the input is assumed to be scaled in [0, 1].
        projection_dim (int, optional): The projection dimension to use to setup a pooling layer
    """

    version: Literal[
        "facebook/dinov2-small",
        "facebook/dinov2-base",
        "facebook/dinov2-large",
        "facebook/dinov2-giant",
    ] = "facebook/dinov2-giant"
    input_key: str = "image"
    layer: Literal["last", "pooled"] = "last"
    always_return_pooled: bool = True
    freeze_backbone: bool = True
    do_processor: bool = True
    projection_dim: Optional[int] = None
