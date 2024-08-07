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
from stylebench.models.ip_adapters import IPAdapterConfig, IPAdapterModel

# ENV VARIABLES
PATH = os.path.dirname(os.path.abspath(__file__))  # gives examples/benchmark
PARENT_PATH = Path(PATH).parent.parent

DATA_PATH = os.path.join(PARENT_PATH, "data/fixed_papers.tar")
OUTPUT_PATH = os.path.join(PARENT_PATH, "output/results/ip_adapters")

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
    config = IPAdapterConfig(
        adapter_scale=[
            {
                "down": {"block_2": [0.0, 1.0]},
                "up": {"block_0": [0.0, 1.0, 0.0]},
            }
        ]
    )
    model = IPAdapterModel(config)
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

    valid_keys = []
    for batch in tqdm(dataloader):

        # Get images and data
        image, data, key = batch["image"][0], batch["json"][0], batch["__key__"][0]
        continue

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
            num_inference_steps=30,
        )

        for i, img in enumerate(images):
            img.save(os.path.join(OUTPUT_PATH, f"{key}_{prompts[i]}.png"))

        raise NotImplementedError
