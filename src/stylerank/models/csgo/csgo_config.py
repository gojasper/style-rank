from pydantic.dataclasses import dataclass

from stylerank.models.csgo.src.utils import BLOCKS as BLOCKS
from stylerank.models.csgo.src.utils import controlnet_BLOCKS as controlnet_BLOCKS

from ..base.model_config import ModelConfig


@dataclass
class CSGOConfig(ModelConfig):

    sdxl_version: str = "stabilityai/stable-diffusion-xl-base-1.0"
    vae_version: str = "madebyollin/sdxl-vae-fp16-fix"

    csgo_repo_id: str = "InstantX/CSGO"
    csgo_filename: str = "csgo_4_32.bin"
    controlnet_repo_id: str = "TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic"
    controlnet_filename: str = "TTPLANET_Controlnet_Tile_realistic_v2_fp16.safetensors"
    controlnet_config_filename: str = "config.json"
    image_encoder_repo_id: str = "h94/IP-Adapter"
    image_encoder_filename: str = "sdxl_models/image_encoder/model.safetensors"
    image_encoder_config_filename: str = "sdxl_models/image_encoder/config.json"

    target_content_blocks = BLOCKS["content"]
    target_style_blocks = BLOCKS["style"]

    controlnet_target_content_blocks = controlnet_BLOCKS["content"]
    controlnet_target_style_blocks = controlnet_BLOCKS["style"]

    controlnet_adapter: bool = True
    content_model_resampler: bool = True
    style_model_resampler: bool = True
