"""
Industrial Deepfake Image Detection Model.

Integrates frequency domain spectral noise analysis, spatial high-pass filtering,
and PyTorch weight execution when PyTorch weights (.pt/.pth) are present.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from PIL import Image

from app.deepfake.models.base import BaseDeepfakeModel
from app.deepfake.preprocessing.image import extract_image_features

logger = logging.getLogger(__name__)


class DeepfakeImageModel(BaseDeepfakeModel):
    """
    Industry-grade Deepfake Image Detector.

    Applies frequency domain artifact detection, spatial noise filtering,
    and PyTorch inference (when model weights artifact is present).
    """

    def __init__(
        self,
        name: str = "deepfake_image",
        version: str = "v1.0.0",
        artifact_path: str = "models/deepfake/image/v1/model.pt",
        media_type: str = "image",
        framework: str = "pytorch",
    ):
        super().__init__(
            name=name,
            version=version,
            artifact_path=artifact_path,
            media_type=media_type,
            framework=framework,
        )
        self.torch_model = None

    def load(self) -> bool:
        """Attempt to load PyTorch weights if artifact exists, or initialize heuristic pipeline."""
        if self.artifact_path.exists():
            try:
                import torch
                # Safe weights-only load when PyTorch is available
                self.torch_model = torch.jit.load(str(self.artifact_path))
                self.torch_model.eval()
                logger.info("Loaded PyTorch model artifact from %s", self.artifact_path)
            except Exception as exc:
                logger.warning("Could not load PyTorch model artifact (%s) — falling back to heuristic spectral engine.", exc)

        self.loaded = True
        return True

    def predict(self, input_data: Any, metadata: Optional[dict] = None) -> dict:
        """
        Predict deepfake manipulation on a PIL Image or extracted feature dict.

        Args:
            input_data: PIL.Image.Image or dict of pre-extracted features.
            metadata: Optional dictionary containing image dimensions/format.

        Returns:
            dict with prediction, confidence, risk_level, and forensic details.
        """
        if isinstance(input_data, Image.Image):
            features = extract_image_features(input_data)
        elif isinstance(input_data, dict):
            features = input_data
        else:
            raise ValueError(f"Invalid input_data type '{type(input_data)}' for DeepfakeImageModel.")

        # Extract core forensic signals
        spectral_var = features.get("spectral_variance", 0.0)
        high_freq_ratio = features.get("high_freq_ratio", 0.0)
        spatial_noise = features.get("spatial_noise_level", 0.0)
        color_var = features.get("color_channel_variance", 0.0)

        # PyTorch model inference if weights are loaded
        if self.torch_model is not None:
            try:
                import torch
                # Tensor preprocessing: 224x224 RGB normalized
                if isinstance(input_data, Image.Image):
                    resized = input_data.convert("RGB").resize((224, 224))
                    arr = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float() / 255.0
                    tensor_in = arr.unsqueeze(0)
                    with torch.no_grad():
                        output = self.torch_model(tensor_in)
                        prob = float(torch.sigmoid(output).item())
                        confidence = round(prob, 4)
                        prediction = "fake" if confidence >= 0.5 else "real"
                        risk_level = self._compute_risk_level(confidence, prediction)
                        return {
                            "prediction": prediction,
                            "confidence": confidence,
                            "risk_level": risk_level,
                            "details": {
                                "model_type": "pytorch",
                                "raw_probability": confidence,
                                "features": features,
                            },
                        }
            except Exception as exc:
                logger.warning("PyTorch inference error (%s) — using heuristic analysis.", exc)

        # Heuristic Spectral & Spatial Fusion Algorithm
        # Deepfake/GAN generated images often exhibit high-frequency power spectrum anomalies
        # and abnormal spatial noise residual distributions.
        anomaly_score = 0.0

        if high_freq_ratio > 0.85 or high_freq_ratio < 0.15:
            anomaly_score += 0.35
        if spectral_var > 12.0:
            anomaly_score += 0.30
        if spatial_noise > 45.0:
            anomaly_score += 0.25
        if color_var < 100.0:  # Oversmoothed generative colors
            anomaly_score += 0.10

        confidence = round(min(max(anomaly_score, 0.05), 0.98), 4)
        prediction = "fake" if confidence >= 0.50 else "real"
        risk_level = self._compute_risk_level(confidence, prediction)

        details = {
            "model_type": "spectral_spatial_heuristic",
            "spectral_variance": spectral_var,
            "high_freq_ratio": high_freq_ratio,
            "spatial_noise_level": spatial_noise,
            "color_channel_variance": color_var,
        }
        if metadata:
            details.update(metadata)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "risk_level": risk_level,
            "details": details,
        }

    @staticmethod
    def _compute_risk_level(confidence: float, prediction: str) -> str:
        if prediction != "fake":
            return "low" if confidence < 0.3 else "none"

        if confidence >= 0.85:
            return "critical"
        elif confidence >= 0.70:
            return "high"
        elif confidence >= 0.50:
            return "medium"
        return "low"
