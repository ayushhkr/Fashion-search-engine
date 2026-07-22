from __future__ import annotations

from pathlib import Path
from typing import List

import faiss
import pandas as pd
from PIL import Image

from .feature_extractor import FeatureExtractor


class FashionSearchEngine:
    def __init__(
        self,
        extractor: FeatureExtractor,
        artifacts_dir: Path,
        dataset_root: Path,
    ) -> None:
        self.extractor = extractor
        self.artifacts_dir = Path(artifacts_dir)
        self.dataset_root = Path(dataset_root)

        self.catalog_path = self.artifacts_dir / "catalog.csv"
        self.index_path = self.artifacts_dir / "faiss.index"

        if not self.catalog_path.exists() or not self.index_path.exists():
            raise FileNotFoundError(
                "Search artifacts are missing. Generate embeddings and the FAISS "
                "index before running the app."
            )

        self.catalog = pd.read_csv(self.catalog_path)
        self.index = faiss.read_index(str(self.index_path))

        if self.index.ntotal != len(self.catalog):
            raise RuntimeError(
                "FAISS index size does not match the catalog size. Rebuild the artifacts."
            )

    def search(self, query_image: Image.Image, top_k: int = 10) -> List[dict]:
        query_embedding = self.extractor.extract_image_embedding(query_image).reshape(1, -1)
        scores, indices = self.index.search(query_embedding, top_k)

        results: List[dict] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0 or index >= len(self.catalog):
                continue

            row = self.catalog.iloc[int(index)].to_dict()
            results.append(
                {
                    "score": float(score),
                    "image_path": row["image_path"],
                    "metadata": row,
                }
            )

        return results
