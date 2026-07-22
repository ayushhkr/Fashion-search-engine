# Fashion Image Search Engine

A production-style image retrieval project that recommends visually similar fashion products using deep learning embeddings, FAISS, and a Streamlit interface.

## Overview

This project uses a pretrained CNN backbone as a feature extractor, precomputes embeddings for the Kaggle **Fashion Product Images Dataset** by Paramaggarwal, and indexes those embeddings with FAISS for fast nearest-neighbor search.

Given an uploaded fashion image, the app:

1. Extracts a normalized embedding
2. Searches the FAISS index with cosine similarity
3. Returns the top 10 visually similar products
4. Displays similarity scores and metadata such as category, brand, and color

## Features

- PyTorch-based feature extraction with `EfficientNetB0` by default
- Optional `ResNet50` backbone support
- Offline embedding generation for the full dataset
- FAISS index with normalized vectors for cosine-similarity search
- Streamlit UI for image upload and recommendations
- Graceful handling for missing metadata, missing images, and corrupted files
- Modular code structure suitable for an internship portfolio

## Project Structure

```text
fashion-image-search-engine/
├── app.py
├── README.md
├── requirements.txt
├── dataset/
│   ├── images/
│   └── styles.csv
├── artifacts/
│   ├── embeddings.npy
│   ├── catalog.csv
│   └── faiss.index
└── src/
    └── fashion_image_search/
        ├── __init__.py
        ├── config.py
        ├── data.py
        ├── embeddings.py
        ├── feature_extractor.py
        ├── search.py
        └── utils.py
```

## Dataset Setup

Download the **Fashion Product Images Dataset** from Kaggle:

- Dataset name: `paramaggarwal/fashion-product-images-dataset`

Place the files like this:

```text
dataset/
├── images/
│   ├── 10000.jpg
│   ├── 10001.jpg
│   └── ...
└── styles.csv
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Build the Embedding Index

Run the offline indexing pipeline after the dataset is available:

```bash
python -m src.fashion_image_search.embeddings --dataset-root dataset --artifacts-dir artifacts --model efficientnet_b0
```

Optional backbone:

```bash
python -m src.fashion_image_search.embeddings --dataset-root dataset --artifacts-dir artifacts --model resnet50
```

This command will:

- Read `styles.csv`
- Match metadata to image files
- Skip invalid or missing images gracefully
- Compute normalized embeddings
- Save `embeddings.npy`
- Save a cleaned `catalog.csv`
- Build and save `faiss.index`

## Run the Streamlit App

```bash
streamlit run app.py
```

## How It Works

### 1. Feature extraction

The model removes the final classification layer from a pretrained CNN and uses the penultimate representation as an embedding vector.

### 2. Embedding normalization

Embeddings are L2-normalized before indexing. This lets FAISS inner-product search behave like cosine similarity.

### 3. Similarity search

The FAISS index returns the nearest vectors from the dataset. The app maps each result back to metadata and image paths.

## Notes

- `brand` is shown when a brand-related column exists in `styles.csv`. If the dataset version does not include it, the app shows `Unknown`.
- Missing or corrupted images are skipped during preprocessing instead of crashing the pipeline.
- The first model run may download pretrained weights, so internet access may be required the first time.

## Example Portfolio Talking Points

- Built an end-to-end visual retrieval system using transfer learning and vector search
- Optimized similarity search with FAISS over precomputed embeddings
- Designed a modular inference and indexing pipeline for production-style use
- Added robust error handling for real-world dataset issues

## Future Improvements

- Add approximate FAISS indexes such as IVF or HNSW for larger catalogs
- Store embeddings in a vector database for deployment
- Add filters for gender, category, season, and usage
- Support batch query evaluation and retrieval metrics
