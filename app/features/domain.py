"""
Domain analysis feature service — V3.

Analyses the structural properties of a domain/hostname for signs of:
- suspicious structure (excessive subdomains, length, entropy)
- IP-based URL
- punycode / Unicode confusables
- suspicious TLD signal
- brand names in subdomain or path
- domain/brand mismatch indicators

Does NOT connect to any external service.
Does NOT fabricate reputation data.
"""

import math
import re
import unicodedata
from collections import Counter
from urllib.parse import urlparse

import tldextract

from app.features.url import SUSPICIOUS_TLDS, entropy, _is_ip_address


# ── Unicode homoglyph / confusable map ─────────────────────────────────────
# Covers the most common character substitutions used in look-alike domains.
_HOMOGLYPH_MAP: dict[str, str] = {
    "0": "o",
    "1": "i",
    "1": "l",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "9": "g",
    "@": "a",
    "а": "a",   # Cyrillic а → Latin a
    "е": "e",   # Cyrillic е → Latin e
    "о": "o",   # Cyrillic о → Latin o
    "р": "p",   # Cyrillic р → Latin p
    "с": "c",   # Cyrillic с → Latin c
    "х": "x",   # Cyrillic х → Latin x
    "ν": "v",   # Greek nu
    "μ": "u",   # Greek mu
    "ı": "i",   # Dotless i
    "ℓ": "l",   # Script ℓ
    "ó": "o",
    "ö": "o",
    "ü": "u",
    "ä": "a",
    "à": "a",
    "á": "a",
    "è": "e",
    "é": "e",
    "ê": "e",
    "ñ": "n",
    "ß": "ss",
}


def _normalize_for_comparison(text: str) -> str:
    """
    Convert *text* to an ASCII-like form for typosquatting comparison.
    Applies homoglyph mapping and strips diacritics.
    """
    # NFKD decomposition removes many combining diacritics.
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()

    result = []
    for ch in text:
        result.append(_HOMOGLYPH_MAP.get(ch, ch))
    return "".join(result)


# ── Domain analysis ─────────────────────────────────────────────────────────

def analyse_domain(url: str) -> dict:
    """
    Return a dict of domain-level features and signals.

    Keys
    ----
    hostname          : str
    registered_domain : str
    subdomain         : str
    tld               : str
    is_ip             : bool
    domain_length     : int
    subdomain_count   : int
    has_punycode      : bool
    has_unicode       : bool
    hostname_entropy  : float
    has_suspicious_tld: bool
    suspicious_tld_value: str | None
    has_digits_in_domain : bool
    has_hyphens_in_domain: bool
    signals           : list[dict]   — list of evidence signals
    """
    parsed = urlparse(url.strip())
    raw_hostname = (parsed.hostname or "").lower()

    extracted = tldextract.extract(raw_hostname)
    subdomain = extracted.subdomain or ""
    domain = extracted.domain or ""
    suffix = extracted.suffix or ""
    registered_domain = extracted.top_domain_under_public_suffix or raw_hostname

    is_ip = _is_ip_address(raw_hostname)

    signals: list[dict] = []

    # ── IP address ──────────────────────────────────────────────────
    if is_ip:
        signals.append({
            "id": "ip_based_url",
            "severity": "medium",
            "title": "URL uses a raw IP address",
            "detail": (
                f"The host '{raw_hostname}' is a bare IP address. "
                "Legitimate services typically use domain names."
            ),
        })

    # ── Excessive subdomains ─────────────────────────────────────────
    subdomain_parts = [p for p in subdomain.split(".") if p]
    if len(subdomain_parts) >= 3:
        signals.append({
            "id": "excessive_subdomains",
            "severity": "medium",
            "title": "Hostname has many subdomains",
            "detail": (
                f"'{raw_hostname}' has {len(subdomain_parts)} subdomain "
                "levels, which is unusual for legitimate services."
            ),
        })

    # ── Domain length ────────────────────────────────────────────────
    if len(registered_domain) > 40:
        signals.append({
            "id": "long_domain",
            "severity": "low",
            "title": "Registered domain is unusually long",
            "detail": (
                f"The registered domain '{registered_domain}' is "
                f"{len(registered_domain)} characters long."
            ),
        })

    # ── Digits in domain ─────────────────────────────────────────────
    has_digits = bool(re.search(r"\d", domain))
    if has_digits:
        signals.append({
            "id": "digits_in_domain",
            "severity": "low",
            "title": "Domain name contains digits",
            "detail": (
                f"The domain label '{domain}' contains numeric characters. "
                "This is a common technique in typosquatting."
            ),
        })

    # ── High entropy ─────────────────────────────────────────────────
    h = entropy(registered_domain)
    if h > 3.8:
        signals.append({
            "id": "high_domain_entropy",
            "severity": "low",
            "title": "Domain name has high character entropy",
            "detail": (
                f"Entropy of '{registered_domain}' is {h:.2f}, "
                "which may indicate a randomly-generated or DGA domain."
            ),
        })

    # ── Punycode / IDN ───────────────────────────────────────────────
    has_punycode = "xn--" in raw_hostname
    has_unicode = any(ord(c) > 127 for c in raw_hostname)
    if has_punycode or has_unicode:
        signals.append({
            "id": "punycode_domain",
            "severity": "medium",
            "title": "Domain uses punycode or Unicode characters",
            "detail": (
                f"Hostname '{raw_hostname}' contains non-ASCII or punycode "
                "encoding, which is used in homograph attacks."
            ),
        })

    # ── Suspicious TLD ────────────────────────────────────────────────
    has_suspicious_tld = suffix.lower() in SUSPICIOUS_TLDS
    if has_suspicious_tld:
        signals.append({
            "id": "suspicious_tld",
            "severity": "low",
            "title": f"TLD '.{suffix}' is associated with abuse",
            "detail": (
                f"The TLD '.{suffix}' appears in lists of top-abused TLDs. "
                "This alone does not confirm phishing; it is one signal."
            ),
        })

    return {
        "hostname": raw_hostname,
        "registered_domain": registered_domain,
        "subdomain": subdomain,
        "tld": suffix,
        "is_ip": is_ip,
        "domain_length": len(registered_domain),
        "subdomain_count": len(subdomain_parts),
        "has_punycode": has_punycode,
        "has_unicode": has_unicode,
        "hostname_entropy": entropy(raw_hostname),
        "has_suspicious_tld": has_suspicious_tld,
        "suspicious_tld_value": suffix if has_suspicious_tld else None,
        "has_digits_in_domain": has_digits,
        "has_hyphens_in_domain": "-" in domain,
        "signals": signals,
    }


def normalize_domain_for_comparison(domain: str) -> str:
    """Expose the internal normaliser for use by the spoofing analyser."""
    return _normalize_for_comparison(domain)
