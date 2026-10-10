"""
Domain spoofing / look-alike detection analyser — V3.

Combines:
- Brand impersonation (from intelligence/brands.py)
- Character substitution detection
- Homoglyph detection
- Punycode analysis

Produces evidence signals for the risk engine.
Does NOT independently classify a URL as malicious.
"""

from urllib.parse import urlparse

import tldextract

from app.intelligence.brands import detect_brand_impersonation


def analyse_spoofing(url: str) -> dict:
    """
    Return a dict of spoofing/look-alike analysis results.

    Return dict keys
    ----------------
    registered_domain   : str
    subdomain           : str
    path                : str
    signals             : list[dict]
    """
    try:
        parsed = urlparse(url.strip())
        hostname = (parsed.hostname or "").lower()
        path = parsed.path or ""
    except Exception:
        return {"registered_domain": "", "subdomain": "", "path": "", "signals": []}

    extracted = tldextract.extract(hostname)
    registered_domain = extracted.top_domain_under_public_suffix or hostname
    subdomain = extracted.subdomain or ""

    brand_signals = detect_brand_impersonation(
        registered_domain=registered_domain,
        subdomain=subdomain,
        path=path,
    )

    return {
        "registered_domain": registered_domain,
        "subdomain": subdomain,
        "path": path,
        "signals": brand_signals,
    }
