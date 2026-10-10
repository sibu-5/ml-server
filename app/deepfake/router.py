"""
Deepfake Detection API Router — /api/v1/deepfake/

Endpoints:
  POST /api/v1/deepfake/image — Single image deepfake analysis.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.config import settings
from app.security import require_api_key
from llm.provider import llm_provider

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/deepfake",
    tags=["Deepfake Detection"],
    dependencies=[Depends(require_api_key)],
)


def _get_deepfake_service(request: Request):
    """Retrieve DeepfakeService instance from app.state."""
    svc = getattr(request.app.state, "deepfake_service", None)
    if svc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Deepfake detection service is not initialized.",
        )
    return svc


async def _read_upload(file: UploadFile, max_bytes: int) -> bytes:
    """Read upload bytes and enforce max payload size limit."""
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        status_code = getattr(status, "HTTP_413_CONTENT_TOO_LARGE", status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
        raise HTTPException(
            status_code=status_code,
            detail=f"Uploaded file exceeds maximum allowed size of {max_bytes // (1024 * 1024)} MB.",
        )
    return data


def _handle_service_error(result: dict) -> None:
    """Convert service-level error strings into standard HTTP status codes."""
    if "error" not in result:
        return

    err = result["error"]
    status_422 = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", status.HTTP_422_UNPROCESSABLE_ENTITY)

    validation_keywords = ("Empty", "exceeds", "Unsupported", "Corrupted", "dimensions", "pixels", "header")
    if any(kw.lower() in err.lower() for kw in validation_keywords):
        raise HTTPException(status_code=status_422, detail=err)

    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=err)


@router.post(
    "/image",
    summary="Detect deepfake manipulation in an image",
    description="Submit an image file (JPEG, PNG, WEBP, BMP, TIFF) for deepfake manipulation analysis.",
)
async def analyze_image(
    request: Request,
    file: Annotated[UploadFile, File(description="Image file to analyze")],
):
    """Endpoint for deepfake image analysis."""
    rid = str(uuid.uuid4())
    t0 = time.perf_counter()
    logger.info("deepfake/image request_id=%s filename=%s", rid, file.filename)

    svc = _get_deepfake_service(request)
    raw_bytes = await _read_upload(file, settings.deepfake_max_image_bytes)

    result = await svc.analyze_image(raw_bytes, filename=file.filename)
    _handle_service_error(result)

    # Attach correlation metadata
    result["request_id"] = rid
    result["feature"] = "deepfake"
    result["media_type"] = "image"

    # Explanation Layer Integration
    if hasattr(llm_provider, "enhance_deepfake_explanation"):
        explanation_res = await llm_provider.enhance_deepfake_explanation(
            prediction=result.get("prediction", "unavailable"),
            confidence=result.get("confidence", 0.0),
            risk_level=result.get("risk_level", "none"),
            details=result.get("details", {}),
            filename=file.filename,
        )
        result["explanation"] = explanation_res
    else:
        result["explanation"] = {
            "summary": "Deepfake analysis complete.",
            "status": "disabled" if not settings.llm_enabled else "unavailable",
        }

    return result
