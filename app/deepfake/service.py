"""
Deepfake Detection Service — Coordination and Execution Layer.

Coordinates PixelGuard validation, concurrency limiting, model inference,
and structured response formatting for Deepfake Image Detection.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from app.config import settings
from app.deepfake.models.base import get_inference_semaphore, DeepfakeStubModel
from app.deepfake.preprocessing.pixel_guard import validate_raw_image_bytes, PixelGuardError
from app.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class DeepfakeService:
    """Service layer managing deepfake detection requests."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    async def analyze_image(
        self,
        raw_bytes: bytes,
        filename: Optional[str] = None,
    ) -> dict:
        """
        Analyze an uploaded image payload for deepfake manipulation.

        Returns:
            dict containing prediction, confidence, risk_level, details, and error if any.
        """
        t0 = time.perf_counter()

        # 1. Validation and Safety Guard
        try:
            img, metadata = validate_raw_image_bytes(
                raw_bytes=raw_bytes,
                max_bytes=settings.deepfake_max_image_bytes,
                max_pixels=settings.deepfake_max_image_pixels,
            )
        except PixelGuardError as pge:
            logger.warning("PixelGuard rejected image (%s): %s", filename, pge)
            return {
                "error": str(pge),
                "prediction": "unavailable",
                "confidence": 0.0,
                "risk_level": "none",
            }
        except Exception as exc:
            logger.error("Unexpected image preprocessing failure (%s): %s", filename, exc)
            return {
                "error": f"Failed to process image payload: {str(exc)}",
                "prediction": "unavailable",
                "confidence": 0.0,
                "risk_level": "none",
            }

        if filename:
            metadata["filename"] = filename

        # 2. Model Lookup
        model = self.registry.get("deepfake_image")
        if model is None:
            logger.warning("No 'deepfake_image' model registered.")
            # Fallback stub model
            model = DeepfakeStubModel(
                name="deepfake_image",
                version="v1.0.0",
                artifact_path="models/deepfake/image/v1/model.pt",
            )

        # 3. Concurrency-Bounded Inference Execution
        sema = get_inference_semaphore()
        try:
            if sema is not None:
                async with sema:
                    result = await asyncio.to_thread(model.predict, img, metadata)
            else:
                result = await asyncio.to_thread(model.predict, img, metadata)
        except Exception as exc:
            logger.error("Deepfake image inference error (%s): %s", filename, exc)
            return {
                "error": f"Inference execution failed: {str(exc)}",
                "prediction": "unavailable",
                "confidence": 0.0,
                "risk_level": "none",
            }

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        result["processing_ms"] = elapsed_ms
        result["model"] = model.name
        result["model_version"] = model.version

        return result
