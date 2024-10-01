from typing import Any, Dict

import torch.nn as nn

from .base_metric_config import BaseMetricConfig


class BaseMetric(nn.Module):
    def __init__(self, config: BaseMetricConfig):
        nn.Module.__init__(self)
        self.config = config
        self.input_key = config.input_key

    def forward(
        self, batch_1: Dict[str, Any], batch_2: Dict[str, Any] = None, *args, **kwargs
    ):
        raise NotImplementedError(
            "Forward method should be implemented in a child class"
        )
