import json
from pathlib import Path

import pandas as pd


def validate_dataset(X: pd.DataFrame, y: pd.Series):
    report = {
        "rows": int(len(X)),
        "features": list(X.columns),
        "missing_values": int(X.isna().sum().sum()),
        "positive_samples": int((y == 1).sum()),
        "negative_samples": int((y == 0).sum()),
        "class_balance": {
            "0": float((y == 0).mean()),
            "1": float((y == 1).mean()),
        },
    }

    Path("reports").mkdir(parents=True, exist_ok=True)

    Path("reports/validation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    if len(X) == 0:
        raise ValueError("Dataset contains zero rows.")

    if X.isna().sum().sum() > 0:
        raise ValueError("Feature matrix contains missing values.")

    if y.nunique() < 2:
        raise ValueError("Training data needs both classes: 0 and 1.")

    return report
