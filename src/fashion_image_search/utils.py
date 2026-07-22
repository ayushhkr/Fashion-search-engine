from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import BinaryIO

from PIL import Image


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_open_image(source: str | Path | BinaryIO) -> Image.Image:
    if hasattr(source, "read"):
        payload = source.read()
        if hasattr(source, "seek"):
            source.seek(0)
        image = Image.open(io.BytesIO(payload))
    else:
        image = Image.open(source)
    return image.convert("RGB")


def log_skipped_image(image_path: Path, reason: str) -> None:
    LOGGER.warning("Skipping image %s: %s", image_path, reason)
