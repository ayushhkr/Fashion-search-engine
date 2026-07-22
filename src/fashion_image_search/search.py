from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Sequence

import faiss
import numpy as np

from .utils import LOGGER, configure_logging, ensure_directory

INDEX_FILENAME = "faiss_index.bin"
EMBEDDINGS_FILENAME = "embeddings.npy"
FILENAMES_FILENAME = "filenames.pkl"


class FaissImageIndex:
    """FAISS-backed image retrieval index for cosine-similarity search."""

    def __init__(self, index: faiss.Index, filenames: Sequence[str]) -> None:
        """Initialize the FAISS index wrapper.

        Args:
            index: A FAISS index storing normalized image embeddings.
            filenames: Filenames aligned with the vectors stored in the index.
        """
        self.index = index
        self.filenames = list(filenames)

        if self.index.ntotal != len(self.filenames):
            raise RuntimeError(
                "FAISS index size does not match the number of filenames."
            )

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[dict[str, float | str]]:
        """Search the index with one query embedding.

        Args:
            query_embedding: A single embedding vector with shape ``(1280,)`` or ``(1, 1280)``.
            top_k: Number of nearest neighbors to return.

        Returns:
            A list of dictionaries containing ``filename`` and ``similarity_score``.
        """
        normalized_query = normalize_embeddings(np.asarray(query_embedding, dtype=np.float32))
        scores, indices = self.index.search(normalized_query, top_k)

        results: List[dict[str, float | str]] = []
        for score, index_value in zip(scores[0], indices[0]):
            if index_value < 0 or index_value >= len(self.filenames):
                continue
            results.append(
                {
                    "filename": self.filenames[int(index_value)],
                    "similarity_score": float(score),
                }
            )
        return results


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings for cosine-similarity search."""
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 1D or 2D NumPy array.")

    normalized = embeddings.astype(np.float32, copy=True)
    faiss.normalize_L2(normalized)
    return normalized


def load_artifacts(artifacts_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Load saved embeddings and filenames artifacts from disk."""
    embeddings_path = Path(artifacts_dir) / EMBEDDINGS_FILENAME
    filenames_path = Path(artifacts_dir) / FILENAMES_FILENAME

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not filenames_path.exists():
        raise FileNotFoundError(f"Filenames file not found: {filenames_path}")

    embeddings = np.load(embeddings_path)
    with filenames_path.open("rb") as filenames_file:
        filenames = pickle.load(filenames_file)

    if not isinstance(filenames, list):
        raise RuntimeError("Filenames artifact is not a list.")
    if embeddings.ndim != 2:
        raise RuntimeError("Embeddings array must be 2-dimensional.")
    if embeddings.shape[0] != len(filenames):
        raise RuntimeError(
            "The number of embeddings does not match the number of filenames."
        )

    return embeddings.astype(np.float32), filenames


def build_index(embeddings: np.ndarray, filenames: Sequence[str], artifacts_dir: Path) -> FaissImageIndex:
    """Build and save a FAISS IndexFlatIP from normalized embeddings."""
    configure_logging()
    ensure_directory(Path(artifacts_dir))

    if len(filenames) == 0:
        raise ValueError("Cannot build an index with zero filenames.")
    if embeddings.shape[0] != len(filenames):
        raise ValueError("Embeddings and filenames must have the same length.")

    normalized_embeddings = normalize_embeddings(embeddings)
    dimension = normalized_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(normalized_embeddings)

    index_path = Path(artifacts_dir) / INDEX_FILENAME
    faiss.write_index(index, str(index_path))
    LOGGER.info("Saved FAISS index with %s vectors to %s", index.ntotal, index_path)

    return FaissImageIndex(index=index, filenames=filenames)


def load_index(artifacts_dir: Path) -> FaissImageIndex:
    """Load a saved FAISS index and aligned filenames from disk."""
    configure_logging()
    artifacts_dir = Path(artifacts_dir)
    index_path = artifacts_dir / INDEX_FILENAME
    filenames_path = artifacts_dir / FILENAMES_FILENAME

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index file not found: {index_path}")
    if not filenames_path.exists():
        raise FileNotFoundError(f"Filenames file not found: {filenames_path}")

    index = faiss.read_index(str(index_path))
    with filenames_path.open("rb") as filenames_file:
        filenames = pickle.load(filenames_file)

    if not isinstance(filenames, list):
        raise RuntimeError("Filenames artifact is not a list.")

    LOGGER.info("Loaded FAISS index with %s vectors from %s", index.ntotal, index_path)
    return FaissImageIndex(index=index, filenames=filenames)


def search(query_embedding: np.ndarray, artifacts_dir: Path, top_k: int = 5) -> List[dict[str, float | str]]:
    """Load the index from disk and search for nearest filenames."""
    faiss_index = load_index(artifacts_dir)
    return faiss_index.search(query_embedding=query_embedding, top_k=top_k)
