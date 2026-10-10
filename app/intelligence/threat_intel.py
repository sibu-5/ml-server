"""
Threat intelligence provider — V3.

Design: provider interface pattern.
The application talks only to the ThreatIntelProvider abstraction.
Concrete providers implement the interface.

For V3 the default provider is NullProvider (no external service).
When VIRUSTOTAL_API_KEY is set, VirusTotalProvider is used.

IMPORTANT:
  Do NOT fabricate reputation results.
  If a provider is not configured, return status "not_configured".
  If a lookup fails, return status "error" — never "safe" or "clean".

Future providers can be added by implementing ThreatIntelProvider.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


# ── Abstract interface ───────────────────────────────────────────────────────

class ThreatIntelProvider(ABC):
    """Interface for all threat-intelligence providers."""

    @abstractmethod
    async def lookup_url(self, url: str) -> dict:
        """
        Look up *url* in the threat intelligence database.

        Must return a dict with at least:
            status   : "clean" | "malicious" | "suspicious" | "unknown" |
                       "not_configured" | "error"
            source   : str
            detail   : str
        """

    @abstractmethod
    async def lookup_domain(self, domain: str) -> dict:
        """
        Look up *domain* in the threat intelligence database.

        Same return contract as lookup_url().
        """


# ── Null provider (default) ──────────────────────────────────────────────────

class NullProvider(ThreatIntelProvider):
    """
    Placeholder used when no threat-intel provider is configured.

    Always returns not_configured — never fabricates results.
    """

    async def lookup_url(self, url: str) -> dict:
        return {
            "status": "not_configured",
            "source": "none",
            "detail": (
                "No threat intelligence provider is configured. "
                "Set THREAT_INTEL_PROVIDER and the appropriate API key in .env."
            ),
        }

    async def lookup_domain(self, domain: str) -> dict:
        return {
            "status": "not_configured",
            "source": "none",
            "detail": (
                "No threat intelligence provider is configured."
            ),
        }


# ── VirusTotal provider (optional) ───────────────────────────────────────────

class VirusTotalProvider(ThreatIntelProvider):
    """
    VirusTotal v3 API provider.

    Requires VIRUSTOTAL_API_KEY to be set.
    Uses the public API — rate limited to 4 requests/minute on free tier.

    IMPORTANT: Never treat an error response as "clean".
    """

    _BASE = "https://www.virustotal.com/api/v3"

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._headers = {"x-apikey": api_key}

    async def lookup_url(self, url: str) -> dict:
        import base64
        url_id = base64.urlsafe_b64encode(url.encode()).rstrip(b"=").decode()
        endpoint = f"{self._BASE}/urls/{url_id}"
        return await self._get(endpoint, "url", url)

    async def lookup_domain(self, domain: str) -> dict:
        endpoint = f"{self._BASE}/domains/{domain}"
        return await self._get(endpoint, "domain", domain)

    async def _get(self, endpoint: str, kind: str, query: str) -> dict:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(endpoint, headers=self._headers)

            if resp.status_code == 404:
                return {
                    "status": "unknown",
                    "source": "virustotal",
                    "detail": f"{kind.capitalize()} not found in VirusTotal database.",
                }

            if resp.status_code == 429:
                logger.warning("VirusTotal rate limit hit.")
                return {
                    "status": "error",
                    "source": "virustotal",
                    "detail": "VirusTotal API rate limit exceeded.",
                }

            if resp.status_code != 200:
                return {
                    "status": "error",
                    "source": "virustotal",
                    "detail": f"VirusTotal returned HTTP {resp.status_code}.",
                }

            data = resp.json()
            stats = (
                data.get("data", {})
                .get("attributes", {})
                .get("last_analysis_stats", {})
            )
            malicious = stats.get("malicious", 0)
            suspicious = stats.get("suspicious", 0)
            total = sum(stats.values()) or 1

            if malicious >= 3:
                status = "malicious"
            elif malicious >= 1 or suspicious >= 3:
                status = "suspicious"
            else:
                status = "clean"

            return {
                "status": status,
                "source": "virustotal",
                "malicious_engines": malicious,
                "suspicious_engines": suspicious,
                "total_engines": total,
                "detail": (
                    f"VirusTotal: {malicious}/{total} engines flagged as malicious."
                ),
            }

        except Exception as exc:
            logger.error("VirusTotal lookup error: %s", exc)
            return {
                "status": "error",
                "source": "virustotal",
                "detail": f"Lookup failed: {exc}",
            }


# ── Provider factory ─────────────────────────────────────────────────────────

def get_threat_intel_provider() -> ThreatIntelProvider:
    """
    Return the configured threat-intel provider.

    Selection order:
    1. THREAT_INTEL_PROVIDER=virustotal and VIRUSTOTAL_API_KEY set → VirusTotal
    2. Anything else → NullProvider (not_configured)
    """
    provider = settings.threat_intel_provider.lower()

    if provider == "virustotal" and settings.virustotal_api_key:
        return VirusTotalProvider(settings.virustotal_api_key)

    return NullProvider()


# Singleton — created once at import time.
threat_intel: ThreatIntelProvider = get_threat_intel_provider()
