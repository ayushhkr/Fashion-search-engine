from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from fashion_image_search.config import AppConfig
from fashion_image_search.feature_extractor import FeatureExtractor
from fashion_image_search.search import FaissImageIndex, load_index
from fashion_image_search.utils import safe_open_image

DEFAULT_TOP_K = 5
RESULTS_PER_ROW_WIDE = 3
RESULTS_PER_ROW_NARROW = 2
QUERY_IMAGE_WIDTH_PX = 330
RESULT_IMAGE_HEIGHT_PX = 176


st.set_page_config(
    page_title="Fashion Image Search Engine",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_custom_styles() -> None:
    """Inject custom CSS for a cleaner, premium interface."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
            padding-left: 2.2rem;
            padding-right: 2.2rem;
            max-width: 1280px;
        }
        .hero-title {
            font-size: 2.35rem;
            font-weight: 800;
            line-height: 1.15;
            color: #0f172a;
            margin-bottom: 0.35rem;
            letter-spacing: -0.03em;
        }
        .hero-subtitle {
            font-size: 1rem;
            color: #475569;
            max-width: 880px;
            margin-bottom: 1.4rem;
        }
        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.8rem;
        }
        .query-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
            padding: 1rem;
            text-align: center;
        }
        .query-card img {
            width: 100%;
            max-width: 330px;
            height: auto;
            border-radius: 16px;
            display: block;
            margin: 0 auto;
            object-fit: contain;
        }
        .result-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            box-shadow: 0 14px 36px rgba(15, 23, 42, 0.07);
            padding: 1rem;
            margin-bottom: 1rem;
            min-height: 100%;
            display: flex;
            flex-direction: column;
            gap: 0.15rem;
        }
        .result-image {
            width: 100%;
            height: 176px;
            border-radius: 16px;
            overflow: hidden;
            background: #f1f5f9;
            border: 1px solid #edf2f7;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1rem;
            padding: 0.65rem;
        }
        .result-image img {
            max-width: 100%;
            max-height: 100%;
            width: auto;
            height: auto;
            object-fit: contain;
            display: block;
            margin: 0 auto;
            image-rendering: auto;
        }
        .missing-image {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            height: 100%;
            color: #64748b;
            font-size: 0.95rem;
            font-weight: 600;
            text-align: center;
            padding: 1rem;
        }
        .product-title {
            color: #0f172a;
            font-size: 1rem;
            font-weight: 700;
            line-height: 1.4;
            margin-bottom: 0.35rem;
            min-height: 3rem;
        }
        .product-id {
            color: #64748b;
            font-size: 0.85rem;
            margin-bottom: 0.85rem;
        }
        .meta-line {
            color: #334155;
            font-size: 0.93rem;
            margin-bottom: 0.28rem;
        }
        .meta-label {
            font-weight: 700;
            color: #0f172a;
        }
        .similarity-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            margin-bottom: 0.45rem;
        }
        .similarity-label {
            font-size: 0.92rem;
            font-weight: 700;
            color: #0f172a;
        }
        .similarity-value {
            font-size: 0.92rem;
            font-weight: 700;
            color: #0f766e;
        }
        .progress-shell {
            width: 100%;
            height: 10px;
            border-radius: 999px;
            background: #e2e8f0;
            overflow: hidden;
            margin-bottom: 0.85rem;
        }
        .progress-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #14b8a6 0%, #0f766e 100%);
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid #e2e8f0;
        }
        </style>
        """,
        unsafe_allow_html=True,
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
    """Resolve the dataset image path for one indexed filename."""
    config = AppConfig()
    return config.dataset_root / "images" / filename


def get_result_metadata(filename: str, metadata_df: pd.DataFrame) -> dict[str, str]:
    """Return product metadata for one indexed result with safe fallbacks."""
    product_id = Path(filename).stem
    default_metadata = {
        "productDisplayName": "Unknown product",
        "articleType": "Unknown",
        "gender": "Unknown",
        "baseColour": "Unknown",
        "season": "Unknown",
        "product_id": product_id,
    }

    if product_id not in metadata_df.index:
        return default_metadata

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
        "product_id": product_id,
    }


def image_to_data_uri(image: Image.Image, output_format: str = "PNG") -> str:
    """Convert a PIL image to an inline data URI for HTML rendering."""
    buffer = io.BytesIO()
    image.save(buffer, format=output_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{output_format.lower()};base64,{encoded}"


def image_path_to_data_uri(image_path: Path) -> str | None:
    """Convert a local image file to a data URI, returning None when unavailable."""
    if not image_path.exists():
        return None

    try:
        with Image.open(image_path) as image:
            return image_to_data_uri(image.convert("RGB"))
    except OSError:
        return None


def render_header() -> None:
    """Render the main page heading and subtitle."""
    st.markdown('<div class="hero-title">👕 Fashion Image Search Engine</div>', unsafe_allow_html=True)
    st.markdown(
        (
            '<div class="hero-subtitle">Upload a fashion image to discover visually '
            'similar products using Deep Learning (EfficientNet-B0) and FAISS.</div>'
        ),
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[int, float]:
    """Render search controls and the project description in the sidebar."""
    st.sidebar.header("Search Controls")
    top_k = st.sidebar.slider("Top K", min_value=1, max_value=10, value=DEFAULT_TOP_K)
    similarity_threshold = st.sidebar.slider(
        "Minimum similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01,
        format="%.2f",
    )

    with st.sidebar.expander("About Project", expanded=True):
        st.write(
            "This project uses EfficientNet-B0 as a visual feature extractor and FAISS "
            "to retrieve the most similar fashion products from a pre-indexed catalog."
        )
        st.write(
            "The recommendations are based on cosine similarity over normalized image embeddings."
        )

    return top_k, similarity_threshold


def render_query_section(query_image: Image.Image) -> None:
    """Render the uploaded query image in a centered premium card."""
    st.markdown('<div class="section-title">Query Image</div>', unsafe_allow_html=True)
    query_uri = image_to_data_uri(query_image)

    left_spacer, center_col, right_spacer = st.columns([1.1, 1.6, 1.1])
    with center_col:
        st.markdown(
            f'''
            <div class="query-card">
                <img src="{query_uri}" alt="Uploaded query image" />
            </div>
            ''',
            unsafe_allow_html=True,
        )


def render_result_card(result: dict[str, Any], metadata_df: pd.DataFrame) -> None:
    """Render one recommendation card with consistent image sizing and metadata."""
    filename = str(result["filename"])
    score = max(0.0, min(1.0, float(result["similarity_score"])))
    metadata = get_result_metadata(filename, metadata_df)
    image_uri = image_path_to_data_uri(build_image_path(filename))
    progress_width = f"{score * 100:.1f}%"

    image_markup = (
        f'<img src="{image_uri}" alt="{filename}" />'
        if image_uri
        else '<div class="missing-image">Image unavailable</div>'
    )

    st.markdown(
        f'''
        <div class="result-card">
            <div class="result-image">{image_markup}</div>
            <div class="similarity-row">
                <span class="similarity-label">Similarity</span>
                <span class="similarity-value">{score * 100:.1f}%</span>
            </div>
            <div class="progress-shell">
                <div class="progress-fill" style="width: {progress_width};"></div>
            </div>
            <div class="product-title">{metadata['productDisplayName']}</div>
            <div class="product-id">Product ID: {metadata['product_id']}</div>
            <div class="meta-line"><span class="meta-label">Article Type:</span> {metadata['articleType']}</div>
            <div class="meta-line"><span class="meta-label">Gender:</span> {metadata['gender']}</div>
            <div class="meta-line"><span class="meta-label">Base Colour:</span> {metadata['baseColour']}</div>
            <div class="meta-line"><span class="meta-label">Season:</span> {metadata['season']}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def render_results_grid(results: list[dict[str, Any]], metadata_df: pd.DataFrame) -> None:
    """Render search results in a balanced responsive grid."""
    columns_per_row = RESULTS_PER_ROW_WIDE if len(results) >= 5 else RESULTS_PER_ROW_NARROW
    columns = st.columns(columns_per_row)
    for index, result in enumerate(results):
        with columns[index % columns_per_row]:
            render_result_card(result, metadata_df)


def load_app_dependencies() -> tuple[pd.DataFrame, FaissImageIndex, FeatureExtractor]:
    """Load cached app dependencies and surface user-friendly failures."""
    metadata_df = load_styles_metadata()
    faiss_index = load_faiss_index()
    extractor = load_feature_extractor()
    return metadata_df, faiss_index, extractor


def main() -> None:
    """Run the improved Streamlit UI for fashion image retrieval."""
    apply_custom_styles()
    render_header()
    top_k, similarity_threshold = render_sidebar()

    try:
        metadata_df, faiss_index, extractor = load_app_dependencies()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        st.error(str(exc))
        st.info(
            "Please make sure `dataset/styles.csv`, `artifacts/embeddings.npy`, "
            "`artifacts/filenames.pkl`, and `artifacts/faiss_index.bin` are available."
        )
        return

    uploaded_file = st.file_uploader(
        "Upload a fashion image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, JPEG, PNG, and WEBP.",
    )

    if uploaded_file is None:
        st.info("Upload a fashion image to begin exploring visually similar products.")
        return

    try:
        query_image = safe_open_image(uploaded_file)
    except UnidentifiedImageError:
        st.error("That file does not look like a valid image. Please upload a JPG, PNG, or WEBP image.")
        return
    except OSError:
        st.error("The uploaded image could not be processed. Please try another file.")
        return

    render_query_section(query_image)

    with st.spinner("Searching similar products..."):
        query_embedding = extractor.extract_embedding(query_image)
        raw_results = faiss_index.search(query_embedding=query_embedding, top_k=top_k)

    filtered_results = [
        result for result in raw_results if float(result["similarity_score"]) >= similarity_threshold
    ]

    if not filtered_results:
        st.warning(
            "No similar products matched the current similarity threshold. Try lowering the threshold in the sidebar."
        )
        return

    st.success(f"Found {len(filtered_results)} similar products.")
    st.markdown('<div class="section-title">Recommended Products</div>', unsafe_allow_html=True)
    render_results_grid(filtered_results, metadata_df)


if __name__ == "__main__":
    main()


