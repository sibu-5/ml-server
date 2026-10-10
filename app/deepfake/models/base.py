"""
Base interfaces and concurrency controls for Deepfake Detection Models.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Global semaphore for GPU/CPU inference concurrency bounding
_INFERENCE_SEMAPHORE: Optional[asyncio.Semaphore] = None


def init_inference_semaphore(max_concurrent: int = 2) -> None:
    """Initialize the global concurrency semaphore for model inference."""
    global _INFERENCE_SEMAPHORE
    _INFERENCE_SEMAPHORE = asyncio.Semaphore(max_concurrent)
    logger.info("Initialized deepfake inference semaphore (max_concurrent=%d)", max_concurrent)


def get_inference_semaphore() -> Optional[asyncio.Semaphore]:
    """Return the global inference semaphore."""
    return _INFERENCE_SEMAPHORE


class BaseDeepfakeModel(ABC):
    """Abstract base class for deepfake detection models."""

    def __init__(
        self,
        name: str,
        version: str,
        artifact_path: str,
        media_type: str = "image",
        framework: str = "custom",
    ):
        self.name = name
        self.version = version
        self.artifact_path = Path(artifact_path)
        self.media_type = media_type
        self.framework = framework
        self.loaded = False

    @abstractmethod
    def load(self) -> bool:
        """Load model weights or setup inference pipeline. Return True on success."""
        pass

    @abstractmethod
    def predict(self, input_data: Any, metadata: Optional[dict] = None) -> dict:
        """
        Execute prediction on input data.

        Returns dict:
            {
                "prediction": "fake" | "real" | "unavailable",
                "confidence": float [0, 1],
                "risk_level": "critical" | "high" | "medium" | "low" | "none",
                "details": dict,
            }
        """
        pass


class DeepfakeStubModel(BaseDeepfakeModel):
    """
    Fallback stub model returned when a model artifact has not yet been trained.
    Returns structured 'unavailable' response without crashing the API.
    """

    def load(self) -> bool:
        self.loaded = True
        return True

    def predict(self, input_data: Any, metadata: Optional[dict] = None) -> dict:
        return {
            "prediction": "unavailable",
            "confidence": 0.0,
            "risk_level": "none",
            "details": {
                "reason": f"Model artifact '{self.name}' not found at {self.artifact_path}. Train and deploy artifact to activate live predictions.",
            },
        }


def create_deepfake_model(item: dict) -> BaseDeepfakeModel:
    """Factory function for instantiating deepfake models from registry config."""
    model_type = item.get("type", "heuristic").lower()
    name = item.get("name", "deepfake_image")
    version = item.get("version", "v1")
    artifact_path = item.get("artifact", "")
    media_type = item.get("media_type", "image")

    if model_type in ("heuristic", "pytorch", "custom"):
        from app.deepfake.models.image_model import DeepfakeImageModel
        return DeepfakeImageModel(
            name=name,
            version=version,
            artifact_path=artifact_path,
            media_type=media_type,
            framework=model_type,
        )

    return DeepfakeStubModel(
        name=name,
        version=version,
        artifact_path=artifact_path,
        media_type=media_type,
        framework="stub",
    )
