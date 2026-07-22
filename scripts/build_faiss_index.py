from __future__ import annotations

from pathlib import Path

from fashion_image_search.search import build_index, load_artifacts


ARTIFACTS_DIR = Path("artifacts")
TOP_K = 5


def main() -> None:
    embeddings, filenames = load_artifacts(ARTIFACTS_DIR)
    faiss_index = build_index(
        embeddings=embeddings,
        filenames=filenames,
        artifacts_dir=ARTIFACTS_DIR,
    )

    query_embedding = embeddings[0]
    results = faiss_index.search(query_embedding=query_embedding, top_k=TOP_K)

    print("Top 5 similar images for sample query:")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['filename']} | similarity={result['similarity_score']:.6f}"
        )


if __name__ == "__main__":
    main()
