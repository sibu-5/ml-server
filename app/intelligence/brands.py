"""
Brand intelligence — V3.

Provides a configurable brand registry and look-alike domain detection.

The registry maps brand names → list of legitimate domains.

Detection strategy
------------------
1. Exact registered-domain match     → legitimate brand domain
2. Brand name present in subdomain   → possible impersonation signal
3. Brand name present in path        → possible impersonation signal
4. Typo-distance (Levenshtein) to the brand label → possible look-alike
5. Homoglyph normalisation match     → possible homograph attack

IMPORTANT:
  Resembling a brand name is NOT proof of phishing.
  The system raises evidence signal "possible_brand_impersonation"
  and lets the risk engine combine it with other evidence.

The registry is intentionally small and configurable.
Add entries to BRAND_REGISTRY below or load them from a JSON file.
"""

import json
import re
from pathlib import Path
from typing import Optional

import tldextract

from app.features.domain import normalize_domain_for_comparison


# ── Brand registry ──────────────────────────────────────────────────────────
# Map: brand_key → {legitimate_domains, display_name}

BRAND_REGISTRY: dict[str, dict] = {
    "paypal": {
        "display_name": "PayPal",
        "legitimate_domains": {"paypal.com", "paypal.me"},
    },
    "microsoft": {
        "display_name": "Microsoft",
        "legitimate_domains": {
            "microsoft.com",
            "live.com",
            "office.com",
            "outlook.com",
            "hotmail.com",
            "onedrive.com",
            "azure.com",
            "microsoftonline.com",
        },
    },
    "google": {
        "display_name": "Google",
        "legitimate_domains": {
            "google.com",
            "gmail.com",
            "youtube.com",
            "googleusercontent.com",
            "googleapis.com",
            "goog",
        },
    },
    "apple": {
        "display_name": "Apple",
        "legitimate_domains": {"apple.com", "icloud.com", "itunes.com"},
    },
    "amazon": {
        "display_name": "Amazon",
        "legitimate_domains": {
            "amazon.com",
            "amazon.co.uk",
            "amazon.de",
            "aws.amazon.com",
            "amazonaws.com",
        },
    },
    "facebook": {
        "display_name": "Facebook / Meta",
        "legitimate_domains": {
            "facebook.com",
            "fb.com",
            "instagram.com",
            "whatsapp.com",
            "meta.com",
        },
    },
    "netflix": {
        "display_name": "Netflix",
        "legitimate_domains": {"netflix.com"},
    },
    "chase": {
        "display_name": "Chase Bank",
        "legitimate_domains": {"chase.com", "jpmorgan.com"},
    },
    "wellsfargo": {
        "display_name": "Wells Fargo",
        "legitimate_domains": {"wellsfargo.com"},
    },
    "bankofamerica": {
        "display_name": "Bank of America",
        "legitimate_domains": {"bankofamerica.com"},
    },
    "steam": {
        "display_name": "Steam / Valve",
        "legitimate_domains": {"steampowered.com", "steamcommunity.com"},
    },
    "dropbox": {
        "display_name": "Dropbox",
        "legitimate_domains": {"dropbox.com"},
    },
    "linkedin": {
        "display_name": "LinkedIn",
        "legitimate_domains": {"linkedin.com"},
    },
    "twitter": {
        "display_name": "Twitter / X",
        "legitimate_domains": {"twitter.com", "x.com"},
    },
    "dhl": {
        "display_name": "DHL",
        "legitimate_domains": {"dhl.com", "dhl.de"},
    },
    "fedex": {
        "display_name": "FedEx",
        "legitimate_domains": {"fedex.com"},
    },
    "ups": {
        "display_name": "UPS",
        "legitimate_domains": {"ups.com"},
    },
}


# ── Levenshtein distance ────────────────────────────────────────────────────

def _levenshtein(a: str, b: str) -> int:
    """Classic Levenshtein edit distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(
                prev[j + 1] + 1,
                curr[j] + 1,
                prev[j] + (ca != cb),
            ))
        prev = curr
    return prev[-1]


# ── Brand detection ─────────────────────────────────────────────────────────

def detect_brand_impersonation(
    registered_domain: str,
    subdomain: str,
    path: str,
) -> list[dict]:
    """
    Compare *registered_domain* against the brand registry.

    Returns a list of evidence signal dicts.
    An empty list means no brand-impersonation signals were found.

    Each signal has keys: id, severity, title, detail, brand.
    """
    signals: list[dict] = []

    # Extract the domain label (without TLD) for comparison.
    extracted = tldextract.extract(registered_domain)
    domain_label = extracted.domain or registered_domain

    norm_label = normalize_domain_for_comparison(domain_label)
    norm_subdomain = normalize_domain_for_comparison(subdomain)
    norm_path = path.lower()

    for brand_key, brand_info in BRAND_REGISTRY.items():
        display = brand_info["display_name"]
        legit_domains = brand_info["legitimate_domains"]

        # ── Exact registered domain match → legitimate, skip ─────────
        if registered_domain in legit_domains:
            continue
        # Also check base labels.
        legit_labels = {
            tldextract.extract(d).domain
            for d in legit_domains
        }

        # ── Homoglyph normalised exact match ─────────────────────────
        norm_brand = normalize_domain_for_comparison(brand_key)
        norm_legit = {normalize_domain_for_comparison(lbl) for lbl in legit_labels}

        if norm_label == norm_brand or norm_label in norm_legit:
            # The normalised label matches the brand exactly — strong signal.
            signals.append({
                "id": "possible_brand_impersonation",
                "severity": "high",
                "title": f"Domain closely resembles '{display}'",
                "detail": (
                    f"'{domain_label}' normalises to '{norm_label}', "
                    f"which matches the brand '{brand_key}'. "
                    "The domain is not an official {display} domain."
                ),
                "brand": brand_key,
            })
            continue

        # ── Levenshtein typosquatting check ───────────────────────────
        # Only compare when lengths are similar to avoid false positives
        # on short brand names.
        if len(brand_key) >= 4:
            dist = _levenshtein(norm_label, norm_brand)
            # Allow at most 1 edit per 6 characters of the brand name.
            threshold = max(1, len(brand_key) // 6)
            if 0 < dist <= threshold:
                signals.append({
                    "id": "possible_brand_impersonation",
                    "severity": "medium",
                    "title": (
                        f"Domain is a possible typosquatting variant "
                        f"of '{display}'"
                    ),
                    "detail": (
                        f"'{domain_label}' has an edit distance of {dist} "
                        f"from brand label '{brand_key}'. "
                        "This may be a look-alike domain."
                    ),
                    "brand": brand_key,
                })
                continue

        # ── Brand label in subdomain ──────────────────────────────────
        if brand_key in norm_subdomain and norm_subdomain:
            # Ensure it's not one of the actual legitimate domains
            # (e.g. accounts.google.com has 'google' in subdomain but is legit).
            full = f"{subdomain}.{registered_domain}" if subdomain else registered_domain
            if not any(
                full == ld or full.endswith("." + ld)
                for ld in legit_domains
            ):
                signals.append({
                    "id": "brand_in_subdomain",
                    "severity": "medium",
                    "title": (
                        f"Brand name '{display}' appears in the subdomain"
                    ),
                    "detail": (
                        f"The subdomain '{subdomain}' contains the brand "
                        f"label '{brand_key}', but the registered domain "
                        f"'{registered_domain}' is not an official {display} domain."
                    ),
                    "brand": brand_key,
                })
                continue

        # ── Brand label in path ───────────────────────────────────────
        if brand_key in norm_path and registered_domain not in legit_domains:
            signals.append({
                "id": "brand_in_path",
                "severity": "low",
                "title": (
                    f"Brand name '{display}' appears in the URL path"
                ),
                "detail": (
                    f"The URL path contains '{brand_key}', suggesting the "
                    "page may be impersonating that brand."
                ),
                "brand": brand_key,
            })

    return signals
