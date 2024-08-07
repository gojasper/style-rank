import logging
from typing import Any, Dict, Tuple, Union

import einops
import torch
import torch.nn as nn
from transformers import AutoProcessor, Dinov2Model

from ..base import BaseConditioner
from .dinov2_embedder_config import Dinov2ImageEmbedderWithProjectionConfig


class Dinov2ImageEmbedderWithProjection(BaseConditioner):
    """Dinov2ImageEmbedderWithProjection

    This implements the Dinov2 embedder, with an optional projection layer to reduce/increase the dimensionality of the embeddings
    Dinov2 embeddings of shape (batch_size, 256 patch tokens + 1 cls token, hidden_size)
    To use DinoV2 to embed images bigger than 224x224, you must set the `do_processor` to False and set your own preprocessor using Mappers instead. The input is assumed to be scaled in [0, 1].

    The projection layer is trainable while the backbone is frozen by default
    To train the projection layer, make sure to add it to the optimizer: `trainable_params=["conditioner.conditioners.0.projection."]

    Configurations:
        - "dinov2-small": {
            "model": "vit_small_patch14_224",
            "hidden_size": 384,
            "depth": 12,
            "num_heads": 6,
        }
        - "dinov2-base": {
            "model": "vit_base_patch14_224",
            "hidden_size": 768,
            "depth": 12,
            "num_heads": 12,
        }
        - "dinov2-large": {
            "model": "vit_large_patch14_224",
            "hidden_size": 1024,
            "depth": 24,
            "num_heads": 16,
        }
        - "dinov2-giant": {
            "model": "vit_giant_patch14_224",
            "hidden_size": 1536,
            "depth": 40,
            "num_heads": 24,
        }

    Args:
        config (Dinov2ImageEmbedderWithProjectionConfig)
    """

    def __init__(self, config: Dinov2ImageEmbedderWithProjectionConfig):
        BaseConditioner.__init__(self, config)
        self.layer = config.layer
        self.always_return_pooled = config.always_return_pooled

        # on the model of CLIP Image Embedder, add a processor for the images
        self.image_processor = AutoProcessor.from_pretrained(config.version)
        self.do_processor = config.do_processor

        # build model
        self.model = Dinov2Model.from_pretrained(config.version)
        self.hidden_size = self.model.config.hidden_size

        # freeze the backbone
        if config.freeze_backbone:
            self.freeze()

        # build projector
        self.projection_dim = config.projection_dim
        if self.projection_dim:
            self.projection = nn.Linear(self.hidden_size, self.projection_dim)
            logging.info(
                f"Projecting DINO embeddings {self.hidden_size} to {self.projection_dim} dimensions."
            )
            logging.warning(
                "Make sure to add the projection layer to the optimizer if you want to train it: `conditioner.conditioners.0.projection.`"
            )

    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(
        self,
        batch: Dict[str, Any],
        force_zero_embedding: bool = False,
        device="cpu",
        *args,
        **kwargs,
    ) -> Union[dict, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the Dinov2ImageEmbedderWithProjection

        Args:
            batch (Dict[str, Any]): A dictionary containing the data. Assumed to be torch tensors.
                The image is assumed to be scaled in [-1, 1].
            force_zero_embedding (bool, optional): Whether to force the embedding to be zero. Defaults to False.
            device (str, optional): The device to use. Defaults to "cpu".

        Returns:
            Dict[str, torch.Tensor]: The ouput of the embedder.
        """
        images = batch[self.input_key]

        if self.do_processor:
            # transforms images from [-1, 1] to [0, 1]
            images = (images + 1) / 2
            dino_image_processed = self.image_processor(
                images=images, return_tensors="pt", do_rescale=False
            )
        else:
            # Expect the images here to be to the correct format in the range [0,1] for the clip embedder
            # Check the self.image_processor.config for the corresponding mappers
            dino_image_processed = {"pixel_values": images}

        dino_image_processed = {
            k: v.to(device) for k, v in dino_image_processed.items()
        }

        # extract features
        self.model.to(device)
        outputs = self.model(**dino_image_processed)

        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output
            z = einops.repeat(z, "b k -> b l k", l=257)

        # project features
        if self.projection_dim:
            z = self.projection(z)

        if force_zero_embedding:
            z = 0 * z
        output = {self.dim2outputkey[z.dim()]: z}

        if self.always_return_pooled:
            if self.projection_dim:
                outputs.pooler_output = self.projection(outputs.pooler_output)
            if force_zero_embedding:
                outputs.pooler_output = 0 * outputs.pooler_output
            output.update({self.dim2outputkey[2]: outputs.pooler_output})

        return output
