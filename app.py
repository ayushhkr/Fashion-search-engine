from pathlib import Path

import streamlit as st
from PIL import UnidentifiedImageError

from src.fashion_image_search.config import AppConfig
from src.fashion_image_search.feature_extractor import FeatureExtractor
from src.fashion_image_search.search import FashionSearchEngine
from src.fashion_image_search.utils import safe_open_image


st.set_page_config(
    page_title="Fashion Image Search Engine",
    page_icon="👗",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_search_engine() -> FashionSearchEngine:
    config = AppConfig()
    extractor = FeatureExtractor(model_name=config.model_name, device=config.device)
    return FashionSearchEngine(
        extractor=extractor,
        artifacts_dir=config.artifacts_dir,
        dataset_root=config.dataset_root,
    )


def render_result_card(result: dict) -> None:
    image_path = Path(result["image_path"])
    metadata = result["metadata"]

    with st.container(border=True):
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
        else:
            st.warning("Image file is missing on disk.")

        st.caption(f"Similarity score: {result['score']:.4f}")
        st.write(f"**Product**: {metadata.get('productDisplayName', 'Unknown')}")
        st.write(f"**Category**: {metadata.get('display_category', 'Unknown')}")
        st.write(f"**Brand**: {metadata.get('brand', 'Unknown')}")
        st.write(f"**Color**: {metadata.get('baseColour', 'Unknown')}")


def main() -> None:
    st.title("Fashion Image Search Engine")
    st.write(
        "Upload a fashion product image to find the most visually similar items "
        "from the indexed dataset."
    )

    try:
        search_engine = load_search_engine()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info(
            "Place the dataset under `dataset/` and run the embedding generation "
            "command from the README before launching the app."
        )
        return
    except RuntimeError as exc:
        st.error(str(exc))
        return

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, JPEG, PNG, and WEBP.",
    )

    if uploaded_file is None:
        st.info("Upload an image to see similar fashion products.")
        return

    try:
        query_image = safe_open_image(uploaded_file)
    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image.")
        return
    except OSError:
        st.error("The uploaded image could not be read.")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Query Image")
        st.image(query_image, use_container_width=True)

    with col2:
        with st.spinner("Searching for similar products..."):
            results = search_engine.search(query_image=query_image, top_k=10)

        st.subheader("Top Matches")
        if not results:
            st.warning("No similar items were found in the current index.")
            return

        result_columns = st.columns(2)
        for index, result in enumerate(results):
            with result_columns[index % 2]:
                render_result_card(result)


if __name__ == "__main__":
    main()
