# Fashion Image Search Engine

A polished deep learning image retrieval project that finds visually similar fashion products using EfficientNet-B0 embeddings, FAISS vector search, and a Streamlit web app.

## Overview

This project is an image search engine for fashion products, not a classifier. Instead of predicting a label, it maps every product image into a dense embedding space and retrieves the most visually similar items with nearest-neighbor search.

The system is built on top of the Kaggle **Fashion Product Images Dataset** by Paramaggarwal and is designed to be portfolio-ready for ML and software internship applications.

## Highlights

- Pretrained `EfficientNet-B0` used as a feature extractor
- Normalized `1280`-dimensional image embeddings
- FAISS `IndexFlatIP` index for cosine-similarity search
- Streamlit interface for interactive fashion retrieval
- Clean `src/` package layout with reusable modules
- Checkpointed embedding generation pipeline
- Friendly error handling for missing, invalid, or corrupted images
- Deployment-ready setup for Render

## Demo Flow

1. A user uploads a fashion image.
2. The app extracts an EfficientNet-B0 embedding.
3. The embedding is searched against a prebuilt FAISS index.
4. The app returns the most visually similar products.
5. Each result is displayed with similarity score and product metadata.

## Tech Stack

- Python
- PyTorch
- torchvision
- FAISS
- NumPy
- pandas
- Pillow
- Streamlit
- setuptools / `pyproject.toml`

## Project Structure

```text
Fashion-search/
|-- app.py
|-- README.md
|-- Procfile
|-- pyproject.toml
|-- requirements.txt
|-- .streamlit/
|   |-- config.toml
|-- artifacts/
|   |-- embeddings.npy
|   |-- filenames.pkl
|   |-- faiss_index.bin
|-- dataset/
|   |-- images/
|   |-- images_sample/
|   |-- styles.csv
|-- scripts/
|   |-- build_faiss_index.py
|   |-- extract_sample_embedding.py
|   |-- generate_embeddings.py
|-- src/
|   |-- fashion_image_search/
|   |   |-- __init__.py
|   |   |-- config.py
|   |   |-- embeddings.py
|   |   |-- feature_extractor.py
|   |   |-- search.py
|   |   |-- utils.py
```

## Dataset

Dataset used:

- Kaggle: `paramaggarwal/fashion-product-images-dataset`

Expected structure:

```text
dataset/
|-- images/
|   |-- 10000.jpg
|   |-- 10001.jpg
|   |-- ...
|-- images_sample/
|-- styles.csv
```

`styles.csv` provides metadata such as:

- `id`
- `gender`
- `articleType`
- `baseColour`
- `season`
- `productDisplayName`

Image files are matched to metadata by filename, for example `10000.jpg -> id = 10000`.

## Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate it

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install the project

Editable install:

```bash
pip install -e .
```

This installs the `fashion_image_search` package from the `src/` layout and makes local code changes immediately available without reinstalling.

## Running the Pipeline

### 1. Extract one sample embedding

```bash
python scripts/extract_sample_embedding.py
```

### 2. Generate embeddings from `dataset/images_sample`

```bash
python scripts/generate_embeddings.py
```

Outputs:

- `artifacts/embeddings.npy`
- `artifacts/filenames.pkl`

### 3. Build the FAISS index

```bash
python scripts/build_faiss_index.py
```

Outputs:

- `artifacts/faiss_index.bin`

## Run the Streamlit App

Local:

```bash
streamlit run app.py
```

The app loads:

- `dataset/styles.csv`
- `artifacts/embeddings.npy`
- `artifacts/filenames.pkl`
- `artifacts/faiss_index.bin`

without rebuilding embeddings or the FAISS index at runtime.

## How the Retrieval System Works

### Feature Extraction

A pretrained EfficientNet-B0 model is loaded with ImageNet weights and its final classification layer is removed. The network output is used as a compact visual descriptor for each image.

### Embedding Generation

Every valid image is passed through the feature extractor and converted into a normalized `1280`-dimensional vector.

### Similarity Search

All embeddings are L2-normalized and indexed with FAISS using inner-product similarity, which corresponds to cosine similarity for normalized vectors.

### Retrieval Output

For a query image, the app returns the nearest product images along with metadata such as:

- Product name
- Article type
- Gender
- Base colour
- Season
- Product ID

## Current Features

- Query image preview with polished UI
- Sidebar controls for Top K and similarity threshold
- Responsive result cards
- Similarity progress bars
- Metadata-backed recommendation display
- Graceful fallback for missing dataset images
- Cached model, index, and metadata loading for faster app performance

## Deployment on Render

This repository is prepared for Render deployment.

### Important files

- `Procfile`
- `.streamlit/config.toml`
- `pyproject.toml`
- `requirements.txt`

### Start command

```bash
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

### Render notes

Make sure the deployed project includes:

- `dataset/styles.csv`
- `artifacts/embeddings.npy`
- `artifacts/filenames.pkl`
- `artifacts/faiss_index.bin`

## Error Handling

The project is built to fail gracefully when possible:

- Invalid uploads are rejected with user-friendly messages
- Missing dataset images are handled without crashing the UI
- Corrupted image files are skipped during embedding generation
- Missing artifacts raise clear deployment/runtime errors

## Portfolio Talking Points

- Built an end-to-end visual search engine using transfer learning and vector retrieval
- Replaced classification with embedding-based similarity search for real retrieval use cases
- Implemented FAISS indexing for scalable nearest-neighbor lookup
- Structured the project with production-style packaging and deployment configuration
- Designed a clean interactive UI with cached inference components

## Future Improvements

- Add category and gender filters on top of similarity search
- Support larger datasets with IVF / HNSW FAISS indexes
- Add batch search evaluation and retrieval metrics
- Deploy embeddings to a managed vector database
- Add model comparison between EfficientNet and CLIP-style encoders

## License / Usage

This project is intended for educational, portfolio, and experimentation purposes. Please review Kaggle dataset licensing and usage terms before public redistribution.
