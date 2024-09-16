# Style Bench

<p align="center">
  <a href='https://creativecommons.org/licenses/by-nd/4.0/legalcode'>
    <img src="https://img.shields.io/badge/python-3.10+-purple" />
	</a>
  <a href="https://huggingface.co/datasets/jasperai/style-bench">
	    <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-StyleBench-yellow' />
	</a>
  <a href="https://github.com/psf/black">
    <img src='https://img.shields.io/badge/Code_style-black-black' />
	</a>
</p>


A unified benchmarking framework for generative styling models in PyTorch. This repository contains code wrapping the implementation of several papers in the field of generative styling models and implementation of metrics to evaluate the quality of the generated images. We also provide one dataset for comparison and evaluation of the models.

## Models

| **Model**    | **Arxiv**                                 | **Code**                                                   | **Project Page**                                               | **Notes**                                                                           |
| ------------ | ----------------------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| StyleAligned | [Arxiv](https://arxiv.org/abs/2312.02133) | [Code](https://github.com/google/style-aligned/)           | [Project Page](https://style-aligned-gen.github.io/)           |                                                                                     |
| VisualStyle  | [Arxiv](https://arxiv.org/abs/2402.12974) | [Code](https://github.com/naver-ai/Visual-Style-Prompting) | [Project Page](https://curryjung.github.io/VisualStylePrompt/) |                                                                                     |
| IP-Adapter   | [Arxiv](https://arxiv.org/abs/2308.06721) | [Code](https://github.com/tencent-ailab/IP-Adapter)        | [Project Page](https://ip-adapter.github.io/)                  | Using the implementation from [Diffusers](https://github.com/huggingface/diffusers) |
| InstantStyle | [Arxiv](https://arxiv.org/abs/2404.02733) | [Code](https://github.com/InstantStyle/InstantStyle)       | [Project Page](https://instantstyle.github.io/)                | Using the implementation from [Diffusers](https://github.com/huggingface/diffusers) |


## Metrics

We implemented several common metrics to evaluate the quality of the generated images:

- CLIP-Text metric : Cosine Similarity between a caption (embedded using `ClipTextModel`) and the generated image (embedded using `ClipVisionModel`) - Using the implpementation from [Transformers](https://github.com/huggingface/transformers)
- CLIP-Image metric : Cosine Similarity between two images (embedded using `ClipVisionModel`) - Using the implpementation from [Transformers](https://github.com/huggingface/transformers)
- Dino : Cosine Similarity between two images (embedded using `Dinov2Model`) - Using the implpementation from [Dino](https://github.com/facebookresearch/dinov2)
- ImageReward : Score from the [ImageReward](https://github.com/THUDM/ImageReward?tab=readme-ov-file#install-dependency) model 

## Dataset

The [dataset](https://huggingface.co/datasets/jasperai/style-bench) is an aggregation of images from multiple styling papers:
- [B-LoRA](https://b-lora.github.io/B-LoRA/)
- [InstantStyle](https://instantstyle.github.io/)
- [MagicInsert](https://magicinsert.github.io/)
- [StyleAligned](https://style-aligned-gen.github.io/)
- [StyleDrop](https://styledrop.github.io/)
- [VisualStyle](https://curryjung.github.io/VisualStylePrompt/)

## Setup

To be up and running, you need first to create a virtual env with at least `python3.10` installed and activate it

### With `venv`

```bash
python3.10 -m venv envs/style_bench
source envs/style_bench/bin/activate
```

### With `conda`

```bash
conda create -n style_bench python=3.10
conda activate style_bench 
```

### Install the dependencies

Then install the required dependencies (if on GPU) and the repo in editable mode

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Usage

Using the provided code, you can generate stylized images on the provided datasets (or your own given the right format) and evaluate them using the provided metrics. Results can fluctuate as the generation is not seeded and the default prompts are sampled from a list of prompts.

### Scripts

To generate images using one of the provided models, you can use the scripts provided in the `examples/inference` folder. For example, to generate images using the `StyleAligned` model, you can use the following command :

```bash
python examples/inference/stylealigned.py --input-path /path/to/dataset --output-path /path/to/output
```

Default output path is `output/inference/` and the default input path is `data/stylebench_papers.tar`.

Addtionally, you can provide the `--json_path` argument to use a different json file for the prompts or use the `--prompts` argument to provide a list of prompts to use for the generation.

Iterating throught the provided `.tar` file and generate 4 random images based on the prompts provided in the `prompts.json` file, following a similar evaluation process as the one described in the VisualStyle paper.

### Folder structure

The folder structure should be as follows :

```bash
.
├── README.md
├── data
│   ├── stylebench_papers.tar
│   └── prompts.json
├── examples
│   ├── inference
│   └── report
├── output
│   ├── inference
│   └── metrics
├── requirements.txt
├── setup.py
├── src
│   └── stylebench
└── tests
    ├── reference_images
    ├── test_metrics
    └── test_model
```

When running an inference script, the model will by default create a folder with its name to store the generated samples and the reference image using a new folder for each reference (with it's key as name) and the prompts used to generate it. The folder structure should look like this inside the `./output/` folder:

```bash
.
├── inference
│   ├── instant_style
│   │   ├── 0000
│   │   │   ├── prompt_1.png
│   │   │   ├── prompt_2.png
│   │   │   ├── prompt_3.png
│   │   │   ├── prompt_4.png
│   │   │   └── reference.png
│   │   ├── 0001
.   .   .   ....
│   │   └── 0111
│   ├── ip_adapter
│   │   ├── 0000
│   │   ├── 0001
.   .   .   ....
│   │   └── 0111
│   ├── stylealigned
.   .   └── ....
│   └── visualstyle
│       └── ....
└── metrics
    ├── interrupted.csv
    ├── report.csv
    └── metrics.csv
```

### Reports

Given the generated image you can evaluate the results using the provided metrics. For example, to evaluate the generated images using the `CLIP-Text` metric, you can use the following command :

```bash
python examples/report/metrics.py --metrics ClipText --input-path /path/to/dataset --output-path /path/to/output
```

You can run multiple metrics at once by providing a list of metrics to the `--metrics` argument, ie : 

```bash
python examples/report/metrics.py --metrics "[ClipText, ClipImage, Dinov2, ImageReward]" --input-path /path/to/dataset --output-path /path/to/output
```

It will output the results in the `/path/to/output/metrics.csv` file and the mean for each metric in the `/path/to/output/report.csv` file.

If you cancel the process, it will automatically save the results in the `/path/to/output/interrupted.csv` file.


## Results

Running the evaluation on the provided `stylebench_papers.tar` dataset, we get the following results :

| **Model**    | **ImageReward** | **Clip-Text** | **Clip-Image** | **Dinov2** |
| ------------ | --------------- | ------------- | -------------- | ---------- |
| StyleAligned | -1.26           | 19.26         | 68.72          | 36.29      |
| VisualStyle  | -0.72           | 22.12         | 66.68          | 20.80      |
| InstantStyle | -0.13           | 22.78         | 66.43          | 18.48      |
| IP-Adapter   | -2.03           | 15.01         | 83.66          | 40.50      |
  
## Tests

To run the tests to make sure the models and metrics are working as expected, you need to install pytest and run the tests using the following command :

```bash
pip install pytest
````

```bash
pytest tests/
```
