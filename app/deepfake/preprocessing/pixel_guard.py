"""
PixelGuard — Decompression bomb protection & Image Safety Validation.

Protects against:
- Empty files
- Oversized byte payloads
- Decompression bomb image payloads
- Corrupt image streams
- Unsupported media types
"""

import io
import logging
from typing import Tuple
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}
MAGIC_HEADERS = (
    b"\xff\xd8\xff",       # JPEG
    b"\x89PNG\r\n\x1a\n", # PNG
    b"RIFF",              # WEBP / AVI
    b"BM",                # BMP
    b"II*\x00",           # TIFF (Intel)
    b"MM\x00*",           # TIFF (Motorola)
)


class PixelGuardError(ValueError):
    """Raised when image validation or security checks fail."""
    pass


def validate_raw_image_bytes(
    raw_bytes: bytes,
    max_bytes: int = 20 * 1024 * 1024,
    max_pixels: int = 25_000_000,
) -> Tuple[Image.Image, dict]:
    """
    Safely validate and decode raw image bytes.

    Returns:
        (PIL.Image.Image, metadata_dict)
    Raises:
        PixelGuardError if validation fails.
    """
    if not raw_bytes or len(raw_bytes) == 0:
        raise PixelGuardError("Empty file payload provided.")

    if len(raw_bytes) > max_bytes:
        raise PixelGuardError(
            f"File size ({len(raw_bytes)} bytes) exceeds maximum limit of {max_bytes} bytes."
        )

    # Magic header verification
    if not any(raw_bytes.startswith(hdr) for hdr in MAGIC_HEADERS):
        raise PixelGuardError("Unsupported or corrupted image file header.")

    # Configure PIL safety limits
    Image.MAX_IMAGE_PIXELS = max_pixels

    try:
        buf = io.BytesIO(raw_bytes)
        img = Image.open(buf)
        img.verify()  # Fast structural verification

        # Re-open for actual processing (verify closes/invalidates image)
        buf.seek(0)
        img = Image.open(buf)
        img.load()  # Decode image pixels into memory under MAX_IMAGE_PIXELS guard
    except Image.DecompressionBombError:
        raise PixelGuardError(
            f"Decompression bomb detected! Image exceeds maximum pixel limit of {max_pixels} pixels."
        )
    except Exception as exc:
        raise PixelGuardError(f"Corrupted image payload: {str(exc)}")

    fmt = (img.format or "UNKNOWN").upper()
    if fmt not in SUPPORTED_FORMATS:
        raise PixelGuardError(f"Unsupported image format '{fmt}'. Allowed: {', '.join(SUPPORTED_FORMATS)}")

    width, height = img.size
    total_pixels = width * height

    if width < 16 or height < 16:
        raise PixelGuardError(f"Image dimensions ({width}x{height}) are too small (minimum 16x16 required).")

    if total_pixels > max_pixels:
        raise PixelGuardError(f"Image total pixels ({total_pixels}) exceed max limit of {max_pixels}.")

    metadata = {
        "format": fmt,
        "width": width,
        "height": height,
        "total_pixels": total_pixels,
        "mode": img.mode,
        "size_bytes": len(raw_bytes),
    }

    return img, metadata
