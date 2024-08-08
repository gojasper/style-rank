import numpy as np
from pydantic.dataclasses import dataclass

from ..base.model_config import ModelConfig
from .src.sa_handler import StyleAlignedArgs


@dataclass
class StyleAlignedConfig(ModelConfig):
    """A configuration class for the StyleAligned Model

    Wrapper of the StyleAlignedArgs dataclass from the original implementation
    """

    share_group_norm: bool = True
    share_layer_norm: bool = True
    share_attention: bool = True
    adain_queries: bool = True
    adain_keys: bool = True
    adain_values: bool = False
    shared_score_shift: float = np.log(2)
    shared_score_scale: float = 1

    def __post_init__(self):
        self.args = StyleAlignedArgs(
            share_group_norm=self.share_group_norm,
            share_layer_norm=self.share_layer_norm,
            share_attention=self.share_attention,
            adain_queries=self.adain_queries,
            adain_keys=self.adain_keys,
            adain_values=self.adain_values,
            shared_score_shift=self.shared_score_shift,
            shared_score_scale=self.shared_score_scale,
        )
