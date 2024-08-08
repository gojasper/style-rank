from typing import Any, Dict

import torch.nn as nn

from .model_config import ModelConfig


class BaseModel(nn.Module):
    def __init__(self, config: ModelConfig):
        nn.Module.__init__(self)
        self.config = config
        self.input_key = config.input_key

    def forward(self, batch: Dict[str, Any], *args, **kwargs):
        raise NotImplementedError("forward method is not implemented")

    def compute_metrics(self, batch: Dict[str, Any], *args, **kwargs):
        """Compute the metrics"""
        return {}

    def sample(self, batch: Dict[str, Any], *args, **kwargs):
        """Sample from the model"""
        return {}
