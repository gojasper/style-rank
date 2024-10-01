from pydantic.dataclasses import dataclass

from ..base.model_config import ModelConfig


@dataclass
class StyleShotConfig(ModelConfig):
    """A configuration class for the StyleShot Model"""

    base_model_path: str = "runwayml/stable-diffusion-v1-5"
    transformer_block_path: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    device: str = "cuda"
