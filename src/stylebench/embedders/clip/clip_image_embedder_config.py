from typing import Literal, Optional

from pydantic.dataclasses import dataclass

from ..base import BaseConditionerConfig


@dataclass
class ClipImageEmbedderConfig(BaseConditionerConfig):
    """This is the ClipImageEmbedderConfig class which defines all the useful parameters to instantiate the model

    Args:
        version (str): The version of the model on HF Hub. Defaults to "openai/clip-vit-large-patch14".
        always_return_pooled (bool): Whether to always return the pooled output. Defaults to False.
        input_key (str): The key for the input. Defaults to "image".
        repeat_length (int): The number of times to repeat the input along the second dimension when using the pooled output. Defaults to 1.
        layer (Literal["last", "pooled"]): The layer to return. Defaults to "last".
        do_processor (bool): Whether to use the integrated pre-processor (resize, center-crop, normalize...). Defaults to True.
    """

    version: Literal[
        "openai/clip-vit-large-patch14",
        "openai/clip-vit-large-patch14-336",
        "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    ] = "openai/clip-vit-large-patch14"
    always_return_pooled: bool = False
    input_key: str = "image"
    repeat_length: int = 1
    layer: Literal["last", "pooled"] = "last"
    do_processor: bool = True

    def __post_init__(self):
        super().__post_init__()
