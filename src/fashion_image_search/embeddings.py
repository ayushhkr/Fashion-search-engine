from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import UnidentifiedImageError
from tqdm import tqdm

from .feature_extractor import FeatureExtractor
from .utils import LOGGER, configure_logging, ensure_directory

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
LOG_EVERY_N_IMAGES = 100
CHECKPOINT_EVERY_N_IMAGES = 500
EMBEDDING_FILENAME = "embeddings.npy"
FILENAMES_FILENAME = "filenames.pkl"
EMBEDDING_DIM = 1280


def list_image_files(images_dir: Path) -> List[Path]:
    """Return supported image files from a directory in deterministic order."""
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not images_dir.is_dir():
        raise ValueError(f"Expected a directory, received: {images_dir}")

    image_paths = sorted(
        path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not image_paths:
        raise RuntimeError(f"No supported images found in: {images_dir}")
    return image_paths


def save_progress(artifacts_dir: Path, embeddings: Sequence[np.ndarray], filenames: Sequence[str]) -> None:
    """Persist the current embeddings and filenames checkpoint to disk."""
    ensure_directory(artifacts_dir)

    embedding_array = (
        np.vstack(embeddings).astype(np.float32)
        if embeddings
        else np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    )
    embeddings_path = artifacts_dir / EMBEDDING_FILENAME
    filenames_path = artifacts_dir / FILENAMES_FILENAME

    np.save(embeddings_path, embedding_array)

    with filenames_path.open("wb") as filenames_file:
        pickle.dump(list(filenames), filenames_file)

    LOGGER.info("Saved progress: %s embeddings written to %s", len(filenames), artifacts_dir)


def generate_embeddings(images_dir: Path, artifacts_dir: Path, device: str | None = None) -> Tuple[np.ndarray, List[str]]:
    """Generate normalized embeddings for all readable images in a directory.

    Args:
        images_dir: Directory containing input images.
        artifacts_dir: Directory where checkpoint and final files are saved.
        device: Optional inference device override.

    Returns:
        A tuple containing:
        - A NumPy array with shape ``(n_images, 1280)`` and dtype ``float32``.
        - A list of filenames in the same order as the embeddings.

    Raises:
        FileNotFoundError: If the input directory does not exist.
        RuntimeError: If no valid embeddings can be generated.
        ValueError: If the images path is not a directory.
    """
    configure_logging()
    ensure_directory(artifacts_dir)

    image_paths = list_image_files(images_dir)
    extractor = FeatureExtractor(device=device)

    embeddings: List[np.ndarray] = []
    filenames: List[str] = []
    skipped_images = 0

    LOGGER.info("Starting embedding generation for %s images from %s", len(image_paths), images_dir)

    for index, image_path in enumerate(tqdm(image_paths, desc="Generating embeddings"), start=1):
        try:
            embedding = extractor.extract_embedding(image_path)
        except (FileNotFoundError, ValueError, RuntimeError, UnidentifiedImageError, OSError) as exc:
            skipped_images += 1
            LOGGER.warning("Skipping %s: %s", image_path.name, exc)
            continue

        embeddings.append(embedding)
        filenames.append(image_path.name)

        if index % LOG_EVERY_N_IMAGES == 0:
            LOGGER.info(
                "Processed %s/%s images. Valid=%s Skipped=%s",
                index,
                len(image_paths),
                len(filenames),
                skipped_images,
            )

        if index % CHECKPOINT_EVERY_N_IMAGES == 0:
            save_progress(artifacts_dir=artifacts_dir, embeddings=embeddings, filenames=filenames)

    if not embeddings:
        raise RuntimeError("No valid embeddings were generated. Check the input images.")

    save_progress(artifacts_dir=artifacts_dir, embeddings=embeddings, filenames=filenames)

    final_embeddings = np.vstack(embeddings).astype(np.float32)
    LOGGER.info(
        "Embedding generation complete. Valid=%s Skipped=%s Saved to %s",
        len(filenames),
        skipped_images,
        artifacts_dir,
    )
    return final_embeddings, filenames


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the embedding generation script."""
    parser = argparse.ArgumentParser(description="Generate image embeddings from dataset/images_sample.")
    parser.add_argument("--images-dir", type=Path, default=Path("dataset/images_sample"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """Run embedding generation from the command line."""
    args = parse_args()
    generate_embeddings(
        images_dir=args.images_dir,
        artifacts_dir=args.artifacts_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
