from fastapi import Header, HTTPException

from app.config import settings


def require_api_key(
    x_api_key: str | None = Header(default=None),
):
    if not settings.ml_api_key:
        raise HTTPException(
            status_code=503,
            detail="ML_API_KEY is not configured.",
        )

    if x_api_key != settings.ml_api_key:
        raise HTTPException(
            status_code=403,
            detail="Forbidden",
        )
