from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


BRAND_CANDIDATE_COLUMNS = ["brandName", "brand", "brand_name"]


@dataclass
class CatalogItem:
    product_id: int
    image_path: str
    productDisplayName: str
    masterCategory: str
    subCategory: str
    articleType: str
    baseColour: str
    gender: str
    season: str
    usage: str
    brand: str
    display_category: str


def _clean_value(value: object, default: str = "Unknown") -> str:
    if pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def _resolve_brand_column(dataframe: pd.DataFrame) -> str | None:
    for column in BRAND_CANDIDATE_COLUMNS:
        if column in dataframe.columns:
            return column
    return None


def load_catalog(dataset_root: Path) -> pd.DataFrame:
    styles_path = dataset_root / "styles.csv"
    images_dir = dataset_root / "images"

    if not styles_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {styles_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    dataframe = pd.read_csv(styles_path, on_bad_lines="skip")
    if "id" not in dataframe.columns:
        raise ValueError("The styles.csv file must contain an 'id' column.")

    brand_column = _resolve_brand_column(dataframe)
    records: List[CatalogItem] = []

    for row in dataframe.itertuples(index=False):
        product_id = getattr(row, "id", None)
        if pd.isna(product_id):
            continue

        try:
            numeric_id = int(product_id)
        except (TypeError, ValueError):
            continue

        image_path = images_dir / f"{numeric_id}.jpg"
        if not image_path.exists():
            continue

        master_category = _clean_value(getattr(row, "masterCategory", None))
        sub_category = _clean_value(getattr(row, "subCategory", None))
        article_type = _clean_value(getattr(row, "articleType", None))

        if article_type != "Unknown":
            display_category = article_type
        elif sub_category != "Unknown":
            display_category = sub_category
        else:
            display_category = master_category

        brand_value = "Unknown"
        if brand_column:
            brand_value = _clean_value(getattr(row, brand_column, None))

        records.append(
            CatalogItem(
                product_id=numeric_id,
                image_path=str(image_path),
                productDisplayName=_clean_value(
                    getattr(row, "productDisplayName", None)
                ),
                masterCategory=master_category,
                subCategory=sub_category,
                articleType=article_type,
                baseColour=_clean_value(getattr(row, "baseColour", None)),
                gender=_clean_value(getattr(row, "gender", None)),
                season=_clean_value(getattr(row, "season", None)),
                usage=_clean_value(getattr(row, "usage", None)),
                brand=brand_value,
                display_category=display_category,
            )
        )

    return pd.DataFrame([item.__dict__ for item in records])
