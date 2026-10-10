from pathlib import Path

import joblib

from app.features.url import vectorize


class PhishingModel:
    name = "phishing"
    version = "v1"

    def __init__(
        self,
        artifact_path="models/phishing/model.joblib",
    ):
        self.artifact_path = Path(artifact_path)
        self.model = None

    @property
    def loaded(self):
        return self.model is not None

    def load(self):
        if not self.artifact_path.exists():
            return False

        self.model = joblib.load(
            self.artifact_path
        )
        if hasattr(self.model, "set_params"):
            try:
                self.model.set_params(device="cpu")
            except Exception:
                pass
        return True

    def predict_probability(self, features):
        if self.model is None:
            raise RuntimeError(
                "Phishing model is not loaded."
            )

        vector = vectorize(features)

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(
                [vector]
            )
            return float(probabilities[0][-1])

        raise TypeError(
            "Loaded phishing model does not support predict_proba()."
        )
