from typing import Any, Dict

from einops import repeat
from transformers import AutoProcessor, CLIPVisionModel, CLIPVisionModelWithProjection

from ..base import BaseConditioner
from .clip_image_embedder_config import ClipImageEmbedderConfig


class ClipImageEmbedder(BaseConditioner):
    """
    ClipImageEmbedder class which defines the CLIPVisionModel model

    This class expects the input to be a batch of images normalized to [-1, 1].
    See https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModel

    Args:

        config (ClipEmbedderConfig): The config class which defines all the required parameters.
    """

    def __init__(self, config: ClipImageEmbedderConfig):
        BaseConditioner.__init__(self, config)

        self.image_processor = AutoProcessor.from_pretrained(config.version)
        self.model = CLIPVisionModel.from_pretrained(config.version)
        self.always_return_pooled = config.always_return_pooled
        self.repeat_length = config.repeat_length
        self.layer = config.layer
        self.do_processor = config.do_processor

    def freeze(self):
        super().freeze()
        self.model = self.model.eval()

    def forward(
        self,
        batch: Dict[str, Any],
        force_zero_embedding: bool = False,
        device="cpu",
        *args,
        **kwargs,
    ):
        """
        Forward pass of the ClipImageEmbedder

        Args:

            batch (Dict[str, Any]): A dictionary containing the data. Assumed to be torch tensors.
                The image is assumed to be scaled in [-1, 1].
            force_zero_embedding (bool): Whether to force zero embedding.
                This will return an embedding with all entries set to 0. Defaults to False.
            device (str): The device to use. Defaults to "cpu".

        Returns:

            Dict[str, Any]: The output of the embedder. This embedder outputs a 2-dimensional conditioning (type "crossattn")
                and a 1-dimensional conditioning (type "vector") if always_return_pooled is True.
        """
        # get the batch of images as torch tensors
        images = batch[self.input_key]

        if self.do_processor:
            # transforms images from [-1, 1] to [0, 1]
            images = (images + 1) / 2
            clip_image_processed = self.image_processor(
                images=images, return_tensors="pt", do_rescale=False
            )
        else:
            # Expect the images here to be to the correct format in the range [0,1] for the clip embedder
            # Check the self.image_processor.config for the corresponding mappers
            clip_image_processed = {"pixel_values": images}

        clip_image_processed = {
            k: v.to(device) for k, v in clip_image_processed.items()
        }
        self.model = self.model.to(device)
        outputs = self.model(**clip_image_processed)

        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output
            z = repeat(z, "b d -> b l d", l=self.repeat_length)

        if force_zero_embedding:
            z = 0 * z

        output = {self.dim2outputkey[z.dim()]: z}

        if self.always_return_pooled:
            if force_zero_embedding:
                outputs.pooler_output = 0 * outputs.pooler_output
            output.update({self.dim2outputkey[2]: outputs.pooler_output})

        return output


class ClipImageEmbedderWithProjection(BaseConditioner):
    """
    ClipImageEmbedderWithProjection class which defines the CLIPVisionModelWithProjection model

    A projection layer is added to the CLIPVisionModel model and already trained.
    See https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModelWithProjection
    This class expects the input to be a batch of images normalized to [-1, 1].

    Args:

        config (ClipImageEmbedderConfig): The config class which defines all the required parameters.
    """

    def __init__(self, config: ClipImageEmbedderConfig):
        BaseConditioner.__init__(self, config)

        self.image_processor = AutoProcessor.from_pretrained(config.version)
        kwargs = {}
        kwargs["pretrained_model_name_or_path"] = config.version

        # hack to fix the projection dim for the L versions of laion
        if config.version in [
            "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
            "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
        ]:
            kwargs["projection_dim"] = 768

        self.model = CLIPVisionModelWithProjection.from_pretrained(**kwargs)
        self.always_return_pooled = config.always_return_pooled
        self.repeat_length = config.repeat_length
        self.layer = config.layer
        self.do_processor = config.do_processor

    def freeze(self):
        super().freeze()
        self.model = self.model.eval()

    def forward(
        self,
        batch: Dict[str, Any],
        force_zero_embedding: bool = False,
        device="cpu",
        *args,
        **kwargs,
    ):
        """
        Forward pass of the ClipImageEmbedderWithProjection

        Args:

            batch (Dict[str, Any]): A dictionary containing the data. Assumed to be torch tensors.
                The image is assumed to be scaled in [-1, 1].
            force_zero_embedding (bool): Whether to force zero embedding.
                This will return an embedding with all entries set to 0. Defaults to False.
            device (str): The device to use. Defaults to "cpu".

        Returns:

            Dict[str, Any]: The output of the embedder. This embedder outputs a 2-dimensional conditioning (type "crossattn")
                and a 1-dimensional conditioning (type "vector") if always_return_pooled is True.
        """
        # get the batch of images as torch tensors
        images = batch[self.input_key]

        if self.do_processor:
            # transforms images from [-1, 1] to [0, 1]
            images = (images + 1) / 2
            clip_image_processed = self.image_processor(
                images=images, return_tensors="pt", do_rescale=False
            )
        else:
            # Expect the images here to be to the correct format in the range [0,1] for the clip embedder
            # Check the self.image_processor.config for the corresponding mappers
            clip_image_processed = {"pixel_values": images}

        clip_image_processed = {
            k: v.to(device) for k, v in clip_image_processed.items()
        }
        self.model = self.model.to(device)
        outputs = self.model(**clip_image_processed)

        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.image_embeds
            z = repeat(z, "b d -> b l d", l=self.repeat_length)

        if force_zero_embedding:
            z = 0 * z

        output = {self.dim2outputkey[z.dim()]: z}

        if self.always_return_pooled:
            if force_zero_embedding:
                outputs.image_embeds = 0 * outputs.image_embeds
            output.update({self.dim2outputkey[2]: outputs.image_embeds})

        return output
