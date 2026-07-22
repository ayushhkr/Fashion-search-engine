from dataclasses import dataclass, field
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class AppConfig:
    dataset_root: Path = field(default_factory=lambda: PROJECT_ROOT / "dataset")
    artifacts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts")
    model_name: str = "efficientnet_b0"
    batch_size: int = 32
    num_workers: int = 0
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
