from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import AppConfig
from .data import load_catalog
from .feature_extractor import FeatureExtractor
from .utils import LOGGER, configure_logging, ensure_directory, log_skipped_image, safe_open_image


class CatalogImageDataset(Dataset):
    def __init__(self, catalog: pd.DataFrame, extractor: FeatureExtractor) -> None:
        self.catalog = catalog.reset_index(drop=True)
        self.extractor = extractor

    def __len__(self) -> int:
        return len(self.catalog)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.catalog.iloc[index]
        image_path = Path(row["image_path"])
        image = safe_open_image(image_path)
        tensor = self.extractor.preprocess_image(image)
        return tensor, index


def collate_valid_samples(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, List[int]]:
    tensors = [item[0] for item in batch]
    indices = [item[1] for item in batch]
    return torch.stack(tensors), indices


def validate_catalog_images(catalog: pd.DataFrame) -> pd.DataFrame:
    valid_indices: List[int] = []

    for index, row in tqdm(catalog.iterrows(), total=len(catalog), desc="Validating images"):
        image_path = Path(row["image_path"])
        try:
            with safe_open_image(image_path) as image:
                image.verify()
            valid_indices.append(index)
        except (FileNotFoundError, OSError, UnidentifiedImageError) as exc:
            log_skipped_image(image_path, str(exc))

    return catalog.iloc[valid_indices].reset_index(drop=True)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def generate_embeddings(
    dataset_root: Path,
    artifacts_dir: Path,
    model_name: str,
    batch_size: int,
    num_workers: int,
    device: str,
) -> None:
    configure_logging()
    ensure_directory(artifacts_dir)

    LOGGER.info("Loading catalog from %s", dataset_root)
    catalog = load_catalog(dataset_root)
    if catalog.empty:
        raise RuntimeError("No valid image-metadata pairs were found in the dataset.")

    LOGGER.info("Validating image files")
    catalog = validate_catalog_images(catalog)
    if catalog.empty:
        raise RuntimeError("All images were filtered out during validation.")

    extractor = FeatureExtractor(model_name=model_name, device=device)
    dataset = CatalogImageDataset(catalog=catalog, extractor=extractor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_valid_samples,
    )

    embedding_batches: List[np.ndarray] = []
    kept_rows: List[pd.DataFrame] = []

    LOGGER.info("Computing embeddings with model=%s on device=%s", model_name, device)
    for tensors, indices in tqdm(dataloader, desc="Extracting embeddings"):
        embeddings = extractor.extract_tensor_embeddings(tensors)
        embedding_batches.append(embeddings)
        kept_rows.append(catalog.iloc[indices])

    stacked_embeddings = np.vstack(embedding_batches).astype("float32")
    final_catalog = pd.concat(kept_rows, ignore_index=True)

    embeddings_path = artifacts_dir / "embeddings.npy"
    catalog_path = artifacts_dir / "catalog.csv"
    index_path = artifacts_dir / "faiss.index"

    LOGGER.info("Saving embeddings to %s", embeddings_path)
    np.save(embeddings_path, stacked_embeddings)

    LOGGER.info("Saving catalog to %s", catalog_path)
    final_catalog.to_csv(catalog_path, index=False)

    LOGGER.info("Building FAISS index")
    index = build_faiss_index(stacked_embeddings)
    faiss.write_index(index, str(index_path))
    LOGGER.info("Saved FAISS index to %s", index_path)
    LOGGER.info("Indexed %s products", len(final_catalog))


def parse_args() -> argparse.Namespace:
    config = AppConfig()
    parser = argparse.ArgumentParser(description="Generate embeddings and a FAISS index.")
    parser.add_argument("--dataset-root", type=Path, default=config.dataset_root)
    parser.add_argument("--artifacts-dir", type=Path, default=config.artifacts_dir)
    parser.add_argument(
        "--model",
        type=str,
        default=config.model_name,
        choices=["efficientnet_b0", "resnet50"],
    )
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--num-workers", type=int, default=config.num_workers)
    parser.add_argument("--device", type=str, default=config.device)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_embeddings(
        dataset_root=args.dataset_root,
        artifacts_dir=args.artifacts_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
