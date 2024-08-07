from pydantic.dataclasses import dataclass

from ...config import BaseConfig


@dataclass
class BaseMetricConfig(BaseConfig):
    input_key: str = "image"
