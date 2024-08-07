import numpy as np
from pydantic.dataclasses import dataclass

from ..base.model_config import ModelConfig


@dataclass
class VisualStyleConfig(ModelConfig):
    """A configuration class for the VisualStyle Model"""

    use_inf_negative_prompt: bool = True
    use_advanced_sampling: bool = True
    use_prompt_as_null: bool = True

    activate_layer_indices = [[0, 0], [128, 140]]
    attn_map_save_steps = []
    activate_step_indices = [[0, 50]]
    use_shared_attention: bool = False
