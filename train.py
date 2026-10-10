from pathlib import Path
import json

import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from pipeline.ingest import ingest
from pipeline.clean import clean, save_cleaned
from pipeline.features import build_features, save_features
from pipeline.validate import validate_dataset
from pipeline.split import split_data


MODEL_DIR = Path("models/ml-phishing")
DATA_DIR = Path("data/final")
REPORT_DIR = Path("reports")


def main():
    print("1. Ingesting raw datasets...")
    raw = ingest()
    print(f"   Ingested rows: {len(raw)}")

    print("2. Cleaning and normalizing...")
    cleaned = clean(raw)

    if len(cleaned) < 20:
        raise RuntimeError(
            "Fewer than 20 valid rows remain after cleaning. "
            "Check the dataset adapters and diagnostics."
        )

    save_cleaned(cleaned)

    print(f"   Clean rows: {len(cleaned)}")

    print("3. Building features...")
    X = build_features(cleaned)
    save_features(X)

    y = cleaned["label"].astype(int)

    print("4. Validating dataset...")
    validate_dataset(X, y)

    print("5. Splitting dataset...")
    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_data(X, y)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_parquet(DATA_DIR / "X_train.parquet", index=False)
    X_validation.to_parquet(
        DATA_DIR / "X_validation.parquet",
        index=False,
    )
    X_test.to_parquet(DATA_DIR / "X_test.parquet", index=False)

    pd.DataFrame({"label": y_train}).to_parquet(
        DATA_DIR / "y_train.parquet",
        index=False,
    )
    pd.DataFrame({"label": y_validation}).to_parquet(
        DATA_DIR / "y_validation.parquet",
        index=False,
    )
    pd.DataFrame({"label": y_test}).to_parquet(
        DATA_DIR / "y_test.parquet",
        index=False,
    )

    print("6. Training XGBoost...")

    # Use RTX 3050 GPU if available, fall back to CPU
    try:
        import xgboost as xgb
        xgb.train({"device": "cuda"}, xgb.DMatrix([[0]], label=[0]), num_boost_round=1)
        xgb_device = "cuda"
        print("   GPU detected — training on RTX 3050 (cuda)")
    except Exception:
        xgb_device = "cpu"
        print("   GPU not available — training on CPU")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.90,
        colsample_bytree=0.90,
        eval_metric="logloss",
        random_state=42,
        device=xgb_device,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=False,
    )

    print("7. Evaluating...")
    import numpy as np
    if xgb_device == "cuda":
        import xgboost as _xgb
        dtest = _xgb.DMatrix(X_test, device="cuda")
        probabilities = model.get_booster().predict(dtest)
    else:
        probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    metrics = {
        "model": "phishing-xgboost",
        "version": "v1",
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "feature_columns": list(X.columns),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("8. Saving model...")

    joblib.dump(
        model,
        MODEL_DIR / "model.joblib",
    )

    (MODEL_DIR / "feature_schema.json").write_text(
        json.dumps(list(X.columns), indent=2),
        encoding="utf-8",
    )

    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "model": "phishing-xgboost",
                "version": "v1",
                "algorithm": "XGBoost",
                "input_type": "URL-derived numerical features",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    (REPORT_DIR / "training_report.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nTraining complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
