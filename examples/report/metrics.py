import os
import signal
import sys
from pathlib import Path
from typing import List, Literal, Optional, Union

import fire
import pandas as pd
import pillow_avif
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from stylerank.embedders import (
    ClipEmbedderConfig,
    ClipImageEmbedderConfig,
    Dinov2ImageEmbedderWithProjectionConfig,
)
from stylerank.metrics import (
    ClipMetric,
    ClipMetricConfig,
    Dinov2Metric,
    Dinov2MetricConfig,
    ImageRewardMetric,
    ImageRewardMetricConfig,
)

PATH = os.path.dirname(os.path.abspath(__file__))
PARENT_PATH = Path(PATH).parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = pil_to_tensor(image).unsqueeze(0)
    image_tensor = image_tensor / 255.0
    image_tensor = image_tensor * 2 - 1

    return image_tensor


def get_metric_config(
    metric_name: Literal["ClipImage", "ClipText", "Dinov2", "ImageReward"]
):
    clip_image_config = ClipImageEmbedderConfig(
        version="openai/clip-vit-large-patch14",
        always_return_pooled=True,
    )
    clip_text_config = ClipEmbedderConfig(
        version="openai/clip-vit-large-patch14",
        always_return_pooled=True,
    )
    dino_config = Dinov2ImageEmbedderWithProjectionConfig(
        version="facebook/dinov2-base",
    )

    match metric_name:
        case "ClipImage":
            return ClipMetricConfig(
                embedder_config_1=clip_image_config, embedder_config_2=clip_image_config
            )

        case "ClipText":
            return ClipMetricConfig(
                embedder_config_1=clip_text_config, embedder_config_2=clip_image_config
            )

        case "Dinov2":
            return Dinov2MetricConfig(embedder_config=dino_config)

        case "ImageReward":
            return ImageRewardMetricConfig()


def get_metric(metric_name: Literal["ClipImage", "ClipText", "Dinov2", "ImageReward"]):

    metric_config = get_metric_config(metric_name)

    match metric_name:
        case "ClipImage" | "ClipText":
            return ClipMetric(metric_config)
        case "Dinov2":
            return Dinov2Metric(metric_config)
        case "ImageReward":
            return ImageRewardMetric(metric_config)


def main(
    metrics: Union[
        List[Literal["ClipImage", "ClipText", "Dinov2", "ImageReward"]],
        Literal["ClipImage", "ClipText", "Dinov2", "ImageReward"],
    ],
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
):

    if isinstance(metrics, str):
        metrics = [metrics]

    # Load data
    if input_path is None:
        input_path = os.path.join(PARENT_PATH, "output", "inference")
    if output_path is None:
        output_path = os.path.join(PARENT_PATH, "output", "metrics")

    os.makedirs(output_path, exist_ok=True)

    # Load metrics
    metrics_model = [get_metric(metric) for metric in metrics]
    df = pd.DataFrame(columns=["key", "prompt", "model", *metrics])

    # Save metrics even if interrupted
    def signal_handler(*args):
        print("Interrupted by user")
        print("Exiting...")

        merged_df = (
            df.groupby(["key", "prompt", "model"])
            .agg({metric: "first" for metric in metrics})
            .reset_index()
        )
        print("Saving metrics at ", os.path.join(output_path, "interrupted.csv"))
        merged_df.to_csv(os.path.join(output_path, "interrupted.csv"), index=False)

        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    for metric, name in zip(metrics_model, metrics):

        for root, _, files in tqdm(os.walk(input_path), desc=f"Computing {name}"):
            folder_name = Path(root).parent
            key = Path(root).name

            images = []
            prompts = []

            if "reference.png" not in files:
                continue

            for file in files:

                if file == "reference.png":
                    reference_image = load_image(os.path.join(root, file))
                else:
                    images.append(load_image(os.path.join(root, file)))
                    prompt = file.split(".")[0]
                    prompts.append(prompt.replace("_", " "))

            # Create batches
            if metric.config.input_key_1 == "image":
                batch_1 = {"image": reference_image.expand(len(images), -1, -1, -1)}

            if metric.config.input_key_1 == "text":
                batch_1 = {"text": prompts}

            batch_2 = {"image": torch.cat(images)}

            # Compute metrics
            output = metric(batch_1, batch_2, device=DEVICE)

            # Save metrics by concat
            for i, prompt in enumerate(prompts):
                row = {
                    "key": key,
                    "prompt": prompt,
                    "model": str(folder_name).split("/")[-1],
                    name: output["score"][i].item(),
                }
                df = pd.concat([df, pd.DataFrame(row, index=[0])])

    merged_df = (
        df.groupby(["key", "prompt", "model"])
        .agg({metric: "first" for metric in metrics})
        .reset_index()
    )
    merged_df.to_csv(os.path.join(output_path, "metrics.csv"), index=False)
    merged_df.groupby("model").agg({metric: "mean" for metric in metrics}).to_csv(
        os.path.join(output_path, "report.csv"), index=True
    )
    print("Metrics saved to ", os.path.join(output_path, "metrics.csv"))
    print("Report saved to ", os.path.join(output_path, "report.csv"))


if __name__ == "__main__":
    fire.Fire(main)
