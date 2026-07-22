from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import UnidentifiedImageError

from fashion_image_search.config import AppConfig
from fashion_image_search.feature_extractor import FeatureExtractor
from fashion_image_search.search import FaissImageIndex, load_index
from fashion_image_search.utils import safe_open_image

RESULTS_PER_ROW = 3


st.set_page_config(
    page_title="Fashion Image Search Engine",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_feature_extractor() -> FeatureExtractor:
    """Load the feature extractor once per Streamlit session."""
    config = AppConfig()
    return FeatureExtractor(device=config.device)


@st.cache_resource(show_spinner=False)
def load_faiss_index() -> FaissImageIndex:
    """Load the FAISS index once per Streamlit session."""
    config = AppConfig()
    return load_index(config.artifacts_dir)


@st.cache_data(show_spinner=False)
def load_styles_metadata() -> pd.DataFrame:
    """Load and index product metadata once for fast lookups."""
    config = AppConfig()
    styles_path = config.dataset_root / "styles.csv"
    if not styles_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {styles_path}")

    dataframe = pd.read_csv(styles_path, on_bad_lines="skip")
    if "id" not in dataframe.columns:
        raise ValueError("The styles.csv file must contain an 'id' column.")

    dataframe["id"] = dataframe["id"].astype(str)
    return dataframe.set_index("id", drop=False)


def build_image_path(filename: str) -> Path:
    """Resolve a dataset image path from a stored filename."""
    config = AppConfig()
    return config.dataset_root / "images" / filename


def get_result_metadata(filename: str, metadata_df: pd.DataFrame) -> dict[str, str]:
    """Return display metadata for one filename, falling back gracefully."""
    product_id = Path(filename).stem
    if product_id not in metadata_df.index:
        return {
            "productDisplayName": "Unknown product",
            "articleType": "Unknown",
            "gender": "Unknown",
            "baseColour": "Unknown",
            "season": "Unknown",
        }

    row = metadata_df.loc[product_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    def read_value(column: str, fallback: str = "Unknown") -> str:
        value = row.get(column, fallback)
        if pd.isna(value):
            return fallback
        text = str(value).strip()
        return text or fallback

    return {
        "productDisplayName": read_value("productDisplayName", "Unknown product"),
        "articleType": read_value("articleType"),
        "gender": read_value("gender"),
        "baseColour": read_value("baseColour"),
        "season": read_value("season"),
    }


def render_result_card(result: dict[str, Any], metadata_df: pd.DataFrame) -> None:
    """Render one similar-product result card."""
    filename = str(result["filename"])
    score = float(result["similarity_score"])
    image_path = build_image_path(filename)
    metadata = get_result_metadata(filename, metadata_df)

    with st.container(border=True):
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
        else:
            st.warning("Product image is missing from dataset/images.")

        st.metric("Similarity", f"{score * 100:.1f}%")
        st.markdown(f"**{metadata['productDisplayName']}**")
        st.caption(filename)

        with st.expander("Product details", expanded=True):
            st.write(f"**Article Type:** {metadata['articleType']}")
            st.write(f"**Gender:** {metadata['gender']}")
            st.write(f"**Base Colour:** {metadata['baseColour']}")
            st.write(f"**Season:** {metadata['season']}")


def render_sidebar() -> tuple[int, float]:
    """Render the application sidebar controls."""
    st.sidebar.header("⚙️ Search Controls")
    top_k = st.sidebar.slider("Top K", min_value=1, max_value=10, value=5)
    similarity_threshold = st.sidebar.slider(
        "Minimum similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.01,
        format="%.2f",
    )

    with st.sidebar.expander("ℹ️ About", expanded=True):
        st.write(
            "This demo uses EfficientNet-B0 embeddings and a FAISS inner-product "
            "index over L2-normalized vectors to find visually similar fashion items."
        )
        st.write(
            "Upload a product image to retrieve the closest matches from the indexed catalog."
        )

    return top_k, similarity_threshold


def main() -> None:
    """Run the Streamlit fashion image retrieval app."""
    st.title("🛍️ Fashion Image Search Engine")
    st.caption(
        "Upload a fashion image to discover the most visually similar products in the catalog."
    )

    top_k, similarity_threshold = render_sidebar()

    try:
        metadata_df = load_styles_metadata()
        faiss_index = load_faiss_index()
        extractor = load_feature_extractor()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        st.error(str(exc))
        st.info(
            "Make sure `dataset/styles.csv`, `artifacts/embeddings.npy`, "
            "`artifacts/filenames.pkl`, and `artifacts/faiss_index.bin` are present."
        )
        return

    uploaded_file = st.file_uploader(
        "Upload a fashion image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, JPEG, PNG, and WEBP.",
    )

    if uploaded_file is None:
        st.info("Choose an image to start a similarity search.")
        return

    try:
        query_image = safe_open_image(uploaded_file)
    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image.")
        return
    except OSError:
        st.error("The uploaded image could not be read.")
        return

    preview_col, info_col = st.columns([1, 1])
    with preview_col:
        st.subheader("Query Image")
        st.image(query_image, use_container_width=True)

    with info_col:
        st.subheader("Search Settings")
        st.write(f"Top K requested: **{top_k}**")
        st.write(f"Minimum similarity: **{similarity_threshold * 100:.0f}%**")
        with st.expander("Search notes", expanded=True):
            st.write(
                "Similarity is computed with cosine similarity using normalized image embeddings."
            )

    with st.spinner("Searching the catalog for similar products..."):
        query_embedding = extractor.extract_embedding(query_image)
        raw_results = faiss_index.search(query_embedding=query_embedding, top_k=top_k)

    filtered_results = [
        result for result in raw_results if float(result["similarity_score"]) >= similarity_threshold
    ]

    st.subheader("Top Matches")
    if not filtered_results:
        st.warning(
            "No results met the current similarity threshold. Try lowering the threshold in the sidebar."
        )
        return

    columns = st.columns(RESULTS_PER_ROW)
    for index, result in enumerate(filtered_results):
        with columns[index % RESULTS_PER_ROW]:
            render_result_card(result, metadata_df)


if __name__ == "__main__":
    main()
