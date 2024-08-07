from typing import Dict, List, Union

import numpy as np
from pydantic.dataclasses import dataclass

from ..base.model_config import ModelConfig


@dataclass
class IPAdapterConfig(ModelConfig):
    """A configuration class for the IP-Adapter Model"""

    adapter_scale: List[Union[float, Dict[str, Dict]]] = 1.0
    # InstantStyle : [{"down": {"block_2": [0.0, 1.0]}, "up": {"block_0": [0.0, 1.0, 0.0]},}]
    version: str = "h94/IP-Adapter"
    subfolder: str = "sdxl_models"
    weight_name: str = "ip-adapter_sdxl.safetensors"
