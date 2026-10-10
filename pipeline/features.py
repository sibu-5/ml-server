import pandas as pd

from app.features.url import FEATURE_COLUMNS, extract_url_features


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        extract_url_features(url)
        for url in df["normalized_url"]
    ]

    return pd.DataFrame(
        rows,
        columns=FEATURE_COLUMNS,
    )


def save_features(features: pd.DataFrame):
    output_path = "data/processed/features.parquet"
    features.to_parquet(output_path, index=False)
    return output_path
