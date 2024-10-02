import json
import os
from pathlib import Path
from typing import List, Optional

import fire
import numpy as np
from PIL import Image
from tqdm import tqdm

from stylerank.data.datasets import DataModule, DataModuleConfig
from stylerank.data.filters import KeyFilter, KeyFilterConfig
from stylerank.data.mappers import (
    KeyRenameMapper,
    KeyRenameMapperConfig,
    MapperWrapper,
)
from stylerank.models.styleshot import StyleShotConfig, StyleShotModel

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
            ]
        ),
    ]

    data_module = DataModule(data_module_config, filters_map)

    return data_module


def get_model():
    config = StyleShotConfig()
    model = StyleShotModel(config)
    return model


def get_json(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def main(
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    json_path: Optional[str] = None,
    prompts: Optional[List[str]] = None,
):

    if input_path is None:
        input_path = os.path.join(PARENT_PATH, "data/stylerank_papers.tar")
    if output_path is None:
        output_path = os.path.join(PARENT_PATH, "output/inference/style_shot")
    if json_path is None:
        json_path = os.path.join(PARENT_PATH, "data/prompts.json")

    os.makedirs(output_path, exist_ok=True)

    data_module = get_data_module(input_path)
    data_module.setup()
    dataloader = data_module.train_dataloader()
    model = get_model()
    all_prompts = get_json(json_path)["content"]
    model.to("cuda")

    empty_prompt = prompts is None

    for batch in tqdm(dataloader):

        # Get images and data
        style_image, metadata, key = (
            batch["image"][0],
            batch["json"][0],
            batch["__key__"][0],
        )
        key = key.split("/")[-1].split(".")[0]

        if empty_prompt:
            prompts = np.random.choice(all_prompts, 4, replace=False)
            prompts = prompts.tolist()

        input_batch = {"image": style_image}

        images = model.sample(
            batch=input_batch,
            prompts=prompts,
            num_inference_steps=30,
        )
        os.makedirs(os.path.join(output_path, key), exist_ok=True)

        for i, img in enumerate(images):
            img.save(
                os.path.join(output_path, key, f"{prompts[i].replace(' ', '_')}.png")
            )

        # Save style reference image
        style_image.save(os.path.join(output_path, key, "reference.png"))


if __name__ == "__main__":
    fire.Fire(main)
