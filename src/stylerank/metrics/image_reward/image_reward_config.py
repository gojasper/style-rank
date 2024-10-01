from typing import Union

from pydantic.dataclasses import dataclass

from ..base.base_metric_config import BaseMetricConfig


@dataclass
class ImageRewardMetricConfig(BaseMetricConfig):
    input_key_1 = "text"
    input_key_2 = "image"

    def __post_init__(self):
        return super().__post_init__()
