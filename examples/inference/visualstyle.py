import os
from pathlib import Path

import numpy as np
import pillow_avif
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from stylebench.data.datasets import DataModule, DataModuleConfig
from stylebench.data.filters import KeyFilter, KeyFilterConfig
from stylebench.data.mappers import (
    KeyRenameMapper,
    KeyRenameMapperConfig,
    MapperWrapper,
    RescaleMapper,
    RescaleMapperConfig,
    TorchvisionMapper,
    TorchvisionMapperConfig,
)
from stylebench.models.visualstyle import VisualStyleConfig, VisualStyleModel

# ENV VARIABLES
PATH = os.path.dirname(os.path.abspath(__file__))
PARENT_PATH = Path(PATH).parent.parent

DATA_PATH = os.path.join(PARENT_PATH, "data/papers.tar")
OUTPUT_PATH = os.path.join(PARENT_PATH, "output/results/visualstyle")

os.makedirs(OUTPUT_PATH, exist_ok=True)


def get_data_module(DATA_PATH):

    data_module_config = DataModuleConfig(
        shards_path_or_urls=str(DATA_PATH),
        per_worker_batch_size=1,
        num_workers=1,
        decoder="pil",
    )

    filters_map = [
        KeyFilter(KeyFilterConfig(keys=["jpg", "json"], mode="keep")),
        MapperWrapper(
            [
                KeyRenameMapper(KeyRenameMapperConfig(key_map={"jpg": "image"})),
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key="image",
                        transforms=["CenterCrop", "ToTensor"],
                        transforms_kwargs=[
                            {"size": (1024, 1024)},
                            {},
                        ],
                    )
                ),
                RescaleMapper(RescaleMapperConfig(key="image")),
            ]
        ),
    ]

    data_module = DataModule(data_module_config, filters_map)

    return data_module


def get_model():
    config = VisualStyleConfig()
    model = VisualStyleModel(config)
    return model


def process_image(image):
    image = image.convert("RGB").resize((1024, 1024), resample=Image.BICUBIC)
    image_tensor = pil_to_tensor(image).unsqueeze(0)
    image_tensor = image_tensor * 2 - 1

    return image_tensor


if __name__ == "__main__":

    data_module = get_data_module(DATA_PATH)
    data_module.setup()
    dataloader = data_module.train_dataloader()
    model = get_model()

    for batch in tqdm(dataloader):

        # Get images and data
        image, data, key = batch["image"][0], batch["json"][0], batch["__key__"][0]

        # format
        # image_tensor = process_image(image)
        image_tensor = image.unsqueeze(0)
        key = key.split("/")[-1].split(".")[0]

        reference_image_caption = data["caption_blip"]
        prompts = [
            "a robot",
            "a cat",
            "an airplane",
            "a human",
            "the eiffel tower",
            "a bird",
        ]

        input_batch = {"image": image_tensor.to("cuda")}
        model.to("cuda")

        images = model.sample(
            input_batch,
            prompts,
            style_prompt=reference_image_caption,
            num_inference_steps=50,
        )

        for i, img in enumerate(images):
            img.save(os.path.join(OUTPUT_PATH, f"new/{key}_{prompts[i]}.png"))

        raise NotImplementedError
