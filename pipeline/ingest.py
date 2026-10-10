from pathlib import Path
import json
import pandas as pd

RAW_DIR = Path("raw_data")
REPORT_DIR = Path("reports")

URL_COLUMNS = ["url", "URL", "link", "uri", "domain", "website"]
LABEL_COLUMNS = ["label", "Label", "class", "target", "status", "type", "category"]


def find_column(columns, candidates):
    exact = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in exact:
            return exact[candidate.lower()]

    for column in columns:
        name = str(column).lower()
        if any(candidate.lower() in name for candidate in candidates):
            return column
    return None


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".txt":
        return pd.read_csv(path, header=None, names=["url"])

    raise ValueError(f"Unsupported file type: {path.suffix}")


def discover_files():
    allowed = {".csv", ".parquet", ".json", ".jsonl", ".txt"}
    return [
        path
        for path in RAW_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed
    ]


def ingest() -> pd.DataFrame:
    records = []
    diagnostics = []

    for path in discover_files():
        try:
            frame = read_table(path)

            url_column = find_column(frame.columns, URL_COLUMNS)
            label_column = find_column(frame.columns, LABEL_COLUMNS)

            if url_column is None:
                diagnostics.append({
                    "file": str(path),
                    "status": "skipped",
                    "reason": "No URL column detected",
                    "columns": [str(c) for c in frame.columns],
                })
                continue

            output = pd.DataFrame({
                "original_url": frame[url_column].astype("string"),
                "raw_label": (
                    frame[label_column].astype("string")
                    if label_column is not None
                    else pd.Series(pd.NA, index=frame.index, dtype="string")
                ),
                "source": path.parent.name,
                "source_file": str(path),
            })

            records.append(output)

            diagnostics.append({
                "file": str(path),
                "status": "accepted",
                "rows": int(len(output)),
                "url_column": str(url_column),
                "label_column": str(label_column) if label_column else None,
            })

        except Exception as exc:
            diagnostics.append({
                "file": str(path),
                "status": "error",
                "reason": str(exc),
            })

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "ingestion_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )

    if not records:
        raise RuntimeError(
            "No compatible dataset was found. "
            "Put datasets under raw_data/ and inspect reports/ingestion_diagnostics.json."
        )

    return pd.concat(records, ignore_index=True)
