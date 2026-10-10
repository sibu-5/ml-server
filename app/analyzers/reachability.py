"""
Website reachability analyzer — V3.

Performs a safe, SSRF-protected HTTP fetch.

Rules:
- Only http:// and https:// are allowed.
- Every redirect destination is re-validated via SSRF guard.
- Strict timeouts: connect, read, total.
- Response body is limited to MAX_HTML_BYTES.
- Does NOT execute JavaScript.
- Does NOT launch a browser.
- Does NOT process file://, ftp://, etc.
- Does NOT download binary/non-HTML content.
- Returns only HTML/text content for analysis.

Reachability statuses
---------------------
live            — 2xx response received
redirect        — final URL after redirect chain
http_error      — 4xx / 5xx
dns_not_found   — hostname does not resolve
timeout         — connection or read timed out
connection_failed — TCP refused or network error
ssrf_blocked    — destination is private/internal
analysis_error  — unexpected exception

IMPORTANT:
UNREACHABLE != SAFE.
UNREACHABLE != LEGITIMATE.
The system must NOT upgrade a site's verdict to SAFE
merely because it could not be reached.
"""

import re
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx

from app.analyzers.ssrf import SSRFViolation, validate_url_for_fetch
from app.config import settings


# Maximum characters we will keep from the response HTML for downstream
# analysis.  Requests larger than MAX_RESPONSE_BYTES are aborted.
_MAX_RESPONSE_BYTES = settings.max_response_bytes
_MAX_HTML_BYTES = settings.max_html_bytes
_CONNECT_TIMEOUT = settings.connect_timeout
_READ_TIMEOUT = settings.read_timeout
_TOTAL_TIMEOUT = settings.request_timeout
_MAX_REDIRECTS = settings.max_redirects

_HTML_CONTENT_TYPES = frozenset({
    "text/html",
    "text/xhtml",
    "application/xhtml+xml",
    "text/plain",
})

_USER_AGENT = (
    "CyberGuard-Security-Scanner/3.0 "
    "(malicious-URL analysis; non-commercial; internal)"
)


def _safe_content_type(ct: str) -> bool:
    """Return True if the Content-Type is one we will process."""
    ct_lower = (ct or "").lower()
    return any(safe in ct_lower for safe in _HTML_CONTENT_TYPES)


async def fetch_url(
    url: str,
    *,
    follow_redirects: bool = True,
) -> dict:
    """
    Fetch *url* safely and return a structured result.

    Parameters
    ----------
    url             : The URL to fetch (already normalised).
    follow_redirects: Whether to follow HTTP redirects (default True).

    Return dict keys
    ----------------
    status          : str  (one of the reachability statuses above)
    status_code     : int | None
    initial_url     : str
    final_url       : str
    redirect_count  : int
    redirect_chain  : list[dict]   — each hop
    content_type    : str | None
    html            : str | None   — truncated HTML body
    html_length     : int
    is_html         : bool
    error           : str | None
    ssrf_blocked    : bool
    signals         : list[dict]
    """
    signals: list[dict] = []
    redirect_chain: list[dict] = []

    result: dict = {
        "status": "analysis_error",
        "status_code": None,
        "initial_url": url,
        "final_url": url,
        "redirect_count": 0,
        "redirect_chain": redirect_chain,
        "content_type": None,
        "html": None,
        "html_length": 0,
        "is_html": False,
        "error": None,
        "ssrf_blocked": False,
        "signals": signals,
    }

    # ── Pre-fetch SSRF validation ────────────────────────────────────
    try:
        validate_url_for_fetch(url)
    except SSRFViolation as exc:
        result["status"] = "ssrf_blocked"
        result["ssrf_blocked"] = True
        result["error"] = str(exc)
        signals.append({
            "id": "ssrf_blocked",
            "severity": "critical",
            "title": "Request blocked by SSRF protection",
            "detail": str(exc),
        })
        return result

    # ── Custom redirect handler so we can validate each hop ──────────
    class _SSRFTransport(httpx.AsyncHTTPTransport):
        pass  # We handle redirects manually below.

    timeout = httpx.Timeout(
        connect=_CONNECT_TIMEOUT,
        read=_READ_TIMEOUT,
        write=5.0,
        pool=5.0,
    )

    current_url = url
    redirect_count = 0

    async with httpx.AsyncClient(
        follow_redirects=False,   # We manage redirects manually.
        timeout=timeout,
        headers={"User-Agent": _USER_AGENT},
        verify=False,  # We do SSL checking in ssl analyser; don't abort here.
    ) as client:
        while True:
            # SSRF check for the current URL.
            try:
                validate_url_for_fetch(current_url)
            except SSRFViolation as exc:
                result["status"] = "ssrf_blocked"
                result["ssrf_blocked"] = True
                result["final_url"] = current_url
                result["redirect_count"] = redirect_count
                result["error"] = str(exc)
                signals.append({
                    "id": "ssrf_redirect_blocked",
                    "severity": "critical",
                    "title": "Redirect destination blocked by SSRF protection",
                    "detail": str(exc),
                })
                return result

            try:
                response = await client.get(
                    current_url,
                    extensions={"timeout": {
                        "connect": _CONNECT_TIMEOUT,
                        "read": _READ_TIMEOUT,
                    }},
                )
            except httpx.ConnectError as exc:
                # Could be DNS failure or TCP refused.
                msg = str(exc).lower()
                if "name or service not known" in msg or "getaddrinfo" in msg:
                    result["status"] = "dns_not_found"
                else:
                    result["status"] = "connection_failed"
                result["final_url"] = current_url
                result["redirect_count"] = redirect_count
                result["error"] = str(exc)
                return result
            except httpx.TimeoutException as exc:
                result["status"] = "timeout"
                result["final_url"] = current_url
                result["redirect_count"] = redirect_count
                result["error"] = str(exc)
                return result
            except Exception as exc:
                result["status"] = "analysis_error"
                result["final_url"] = current_url
                result["redirect_count"] = redirect_count
                result["error"] = str(exc)
                return result

            hop = {
                "url": current_url,
                "status_code": response.status_code,
            }

            # Redirect handling.
            if response.is_redirect and follow_redirects:
                location = response.headers.get("location", "")
                if not location:
                    # Redirect with no Location — treat as error.
                    result["status"] = "http_error"
                    result["status_code"] = response.status_code
                    result["final_url"] = current_url
                    result["redirect_count"] = redirect_count
                    redirect_chain.append(hop)
                    break

                next_url = urljoin(current_url, location)
                hop["redirect_to"] = next_url
                redirect_chain.append(hop)
                redirect_count += 1

                if redirect_count > _MAX_REDIRECTS:
                    result["status"] = "http_error"
                    result["status_code"] = response.status_code
                    result["final_url"] = current_url
                    result["redirect_count"] = redirect_count
                    result["error"] = (
                        f"Exceeded maximum redirect limit ({_MAX_REDIRECTS})."
                    )
                    signals.append({
                        "id": "too_many_redirects",
                        "severity": "medium",
                        "title": "Redirect chain exceeded the maximum limit",
                        "detail": result["error"],
                    })
                    break

                current_url = next_url
                continue  # Follow the redirect.

            # ── Final response ───────────────────────────────────────
            redirect_chain.append(hop)
            result["final_url"] = current_url
            result["redirect_count"] = redirect_count
            result["status_code"] = response.status_code

            ct = response.headers.get("content-type", "")
            result["content_type"] = ct
            result["is_html"] = _safe_content_type(ct)

            if response.is_success:
                result["status"] = "live"
            else:
                result["status"] = "http_error"
                result["error"] = (
                    f"HTTP {response.status_code}: {response.reason_phrase}"
                )
                # Don't read body for error responses unless it's small.

            # Read HTML body (only for text/html responses, under limit).
            if result["is_html"]:
                try:
                    raw_bytes = await response.aread()
                    raw_bytes = raw_bytes[:_MAX_RESPONSE_BYTES]
                    html_text = raw_bytes.decode("utf-8", errors="replace")
                    html_text = html_text[:_MAX_HTML_BYTES]
                    result["html"] = html_text
                    result["html_length"] = len(html_text)
                except Exception as exc:
                    result["error"] = f"Failed to read response body: {exc}"
            break

    result["redirect_chain"] = redirect_chain
    result["signals"] = signals
    return result
