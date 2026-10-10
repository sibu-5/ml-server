"""
SSL/TLS analyser wrapper — V3.

Thin wrapper around features.ssl.analyse_ssl() that adapts
the result for use by the V3 pipeline.

Exported function: analyse_ssl_for_url(url, timeout)
"""

from urllib.parse import urlparse

from app.features.ssl import analyse_ssl as _analyse_ssl


def analyse_ssl_for_url(url: str, timeout: float = 5.0) -> dict:
    """
    Run SSL/TLS analysis for *url*.

    Only runs for https:// URLs.
    For http:// returns explicit not_checked status.
    """
    try:
        parsed = urlparse(url.strip())
    except Exception:
        return {
            "status": "not_checked",
            "https_used": False,
            "error": "Could not parse URL.",
            "signals": [],
        }

    scheme = parsed.scheme.lower()
    if scheme != "https":
        return {
            "status": "not_checked",
            "https_used": False,
            "tls_connected": False,
            "cert_present": False,
            "cert_valid": False,
            "cert_expired": False,
            "cert_not_yet_valid": False,
            "hostname_match": None,
            "subject_cn": None,
            "issuer_o": None,
            "valid_from": None,
            "valid_until": None,
            "days_until_expiry": None,
            "error": "URL uses HTTP — no SSL to check.",
            "signals": [{
                "id": "no_https",
                "severity": "medium",
                "title": "URL does not use HTTPS",
                "detail": (
                    "The URL uses plain HTTP. "
                    "Credentials sent over HTTP are transmitted in cleartext."
                ),
            }],
        }

    hostname = (parsed.hostname or "").lower()
    port = parsed.port or 443

    return _analyse_ssl(hostname, port=port, timeout=timeout)
