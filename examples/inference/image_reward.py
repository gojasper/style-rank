import os
from pathlib import Path

from stylebench.metrics import ImageRewardMetric, ImageRewardMetricConfig

metric_config = ImageRewardMetricConfig()
metric = ImageRewardMetric(metric_config)
