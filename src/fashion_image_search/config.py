from dataclasses import dataclass, field
from pathlib import Path

import torch


def resolve_project_root() -> Path:
    """Resolve the project root in both local and deployed environments.

    On Render, the package can live inside the virtualenv site-packages directory,
    so relying on ``__file__`` alone points to the wrong base path. We prefer the
    current working directory when it contains the app entrypoint and project files.
    """
    cwd = Path.cwd().resolve()
    candidate_roots = [cwd, *cwd.parents]

    for candidate in candidate_roots:
        if (candidate / 'app.py').exists() and (candidate / 'dataset').exists():
            return candidate

    source_path = Path(__file__).resolve()
    for candidate in source_path.parents:
        if (candidate / 'app.py').exists() and (candidate / 'dataset').exists():
            return candidate

    return cwd


PROJECT_ROOT = resolve_project_root()


@dataclass
class AppConfig:
    dataset_root: Path = field(default_factory=lambda: PROJECT_ROOT / 'dataset')
    artifacts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / 'artifacts')
    model_name: str = 'efficientnet_b0'
    batch_size: int = 32
    num_workers: int = 0
    device: str = field(
        default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu'
    )
