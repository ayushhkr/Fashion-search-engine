from __future__ import annotations

import logging
from pathlib import Path

from fashion_image_search.feature_extractor import FeatureExtractor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def main() -> None:
    sample_dir = Path("dataset/images_sample")
    if not sample_dir.exists():
        raise FileNotFoundError(
            "Sample directory not found. Create `dataset/images_sample` and add one image."
        )

    supported_suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = sorted(
        path for path in sample_dir.iterdir() if path.is_file() and path.suffix.lower() in supported_suffixes
    )
    if not image_paths:
        raise FileNotFoundError(
            "No sample image found in `dataset/images_sample`."
        )

    extractor = FeatureExtractor()
    embedding = extractor.extract_embedding(image_paths[0])

    print(f"Image: {image_paths[0]}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding dtype: {embedding.dtype}")
    print(f"L2 norm: {float((embedding ** 2).sum() ** 0.5):.6f}")
    print(f"First 10 values: {embedding[:10]}")


if __name__ == "__main__":
    main()
