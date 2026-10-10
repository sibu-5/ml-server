import re
from urllib.parse import urlparse

import pandas as pd


LABEL_MAP = {
    "0": 0,
    "1": 1,
    "benign": 0,
    "legitimate": 0,
    "legit": 0,
    "safe": 0,
    "good": 0,
    "phishing": 1,
    "malicious": 1,
    "malware": 1,
    "bad": 1,
}


def normalize_label(value):
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in LABEL_MAP:
        return LABEL_MAP[value]

    return None


def canonicalize_url(value):
    raw = str(value).strip()

    if not raw:
        return None

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw):
        raw = "http://" + raw

    try:
        parsed = urlparse(raw)

        if parsed.scheme.lower() not in {"http", "https"}:
            return None

        if not parsed.hostname:
            return None

        hostname = parsed.hostname.encode("idna").decode("ascii").lower()

        port = parsed.port
        default_port = (
            (parsed.scheme.lower() == "http" and port == 80)
            or (parsed.scheme.lower() == "https" and port == 443)
        )

        host = hostname if not port or default_port else f"{hostname}:{port}"

        result = f"{parsed.scheme.lower()}://{host}"

        if parsed.path:
            result += parsed.path

        if parsed.query:
            result += f"?{parsed.query}"

        if parsed.fragment:
            result += f"#{parsed.fragment}"

        return result

    except Exception:
        return None


def clean(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    output["original_url"] = (
        output["original_url"]
        .astype("string")
        .str.strip()
    )

    output = output[
        output["original_url"].notna()
        & (output["original_url"] != "")
        & (output["original_url"].str.len() <= 4096)
    ]

    output["label"] = output["raw_label"].map(normalize_label)
    output["normalized_url"] = output["original_url"].map(canonicalize_url)

    output = output[
        output["normalized_url"].notna()
        & output["label"].isin([0, 1])
    ]

    # Exact normalized duplicates are removed.
    # Conflicting-label handling is done separately.
    grouped = output.groupby("normalized_url")["label"].nunique()
    conflicting_urls = set(grouped[grouped > 1].index)

    output = output[
        ~output["normalized_url"].isin(conflicting_urls)
    ]

    output = output.drop_duplicates(
        subset=["normalized_url"],
        keep="first",
    )

    return output.reset_index(drop=True)


def save_cleaned(df: pd.DataFrame):
    output_path = "data/processed/cleaned.parquet"
    df.to_parquet(output_path, index=False)
    return output_path
