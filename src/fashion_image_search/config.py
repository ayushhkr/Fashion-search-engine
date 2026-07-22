from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class AppConfig:
    dataset_root: Path = field(default_factory=lambda: Path("dataset"))
    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))
    model_name: str = "efficientnet_b0"
    batch_size: int = 32
    num_workers: int = 0
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

