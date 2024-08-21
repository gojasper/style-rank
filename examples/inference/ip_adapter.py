import json
import os
from pathlib import Path
from typing import Optional

import fire
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
PATH = os.path.dirname(os.path.abspath(__file__))
PARENT_PATH = Path(PATH).parent.parent


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
    # Instant Style Method
    config = IPAdapterConfig()
    model = IPAdapterModel(config)
    return model


def get_json(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def main(
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    json_path: Optional[str] = None,
):

    if input_path is None:
        input_path = os.path.join(PARENT_PATH, "data/stylebench_papers.tar")
    if output_path is None:
        output_path = os.path.join(PARENT_PATH, "output/inference/instant_style")
    if json_path is None:
        json_path = os.path.join(PARENT_PATH, "data/prompts.json")

    os.makedirs(output_path, exist_ok=True)

    data_module = get_data_module(input_path)
    data_module.setup()
    dataloader = data_module.train_dataloader()
    model = get_model()
    all_prompts = get_json(json_path)["content"]

    for batch in tqdm(dataloader):

        # Get images and data
        image, data, key = batch["image"][0], batch["json"][0], batch["__key__"][0]

        image_tensor = image.unsqueeze(0)
        key = key.split("/")[-1].split(".")[0]

        reference_image_caption = data["caption_blip"]
        prompts = np.random.choice(all_prompts, 4, replace=False)
        prompts = prompts.tolist()

        input_batch = {"image": image_tensor.to("cuda")}
        model.to("cuda")

        images = model.sample(
            input_batch,
            prompts,
            num_inference_steps=30,
        )
        os.makedirs(os.path.join(output_path, key), exist_ok=True)

        for i, img in enumerate(images):
            img.save(
                os.path.join(output_path, key, f"{prompts[i].replace(' ', '_')}.png")
            )

        # Save ref Image
        ref_image = ((image + 1) / 2) * 255
        ref_image = ref_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        ref_image = Image.fromarray(ref_image)
        ref_image.save(os.path.join(output_path, key, "reference.png"))


if __name__ == "__main__":
    fire.Fire(main)
