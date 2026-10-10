"""
Redirect analysis — V3.

Analyses the redirect chain produced by reachability.fetch_url().

Produces:
- redirect_count
- initial_domain
- final_domain
- domain_changed
- tld_changed
- cross_scheme  (http → https is OK; https → http is suspicious)
- evidence signals
"""

from urllib.parse import urlparse

import tldextract


def analyse_redirects(
    initial_url: str,
    final_url: str,
    redirect_chain: list[dict],
) -> dict:
    """
    Examine the redirect chain and return structured analysis.

    Parameters
    ----------
    initial_url     : The URL originally submitted.
    final_url       : The URL where the chain ended.
    redirect_chain  : List of hop dicts from reachability.fetch_url().

    Return dict keys
    ----------------
    redirect_count  : int
    initial_domain  : str
    final_domain    : str
    initial_tld     : str
    final_tld       : str
    domain_changed  : bool
    tld_changed     : bool
    cross_scheme    : bool   — scheme changed across any hop
    https_to_http   : bool   — security downgrade detected
    chain_summary   : list[str]  — human-readable hop descriptions
    signals         : list[dict]
    """
    signals: list[dict] = []

    def _registered(url: str) -> tuple[str, str]:
        """Return (registered_domain, tld) for *url*."""
        try:
            parsed = urlparse(url)
            hostname = (parsed.hostname or "").lower()
            ext = tldextract.extract(hostname)
            return ext.top_domain_under_public_suffix or hostname, ext.suffix or ""
        except Exception:
            return url, ""

    def _scheme(url: str) -> str:
        try:
            return urlparse(url).scheme.lower()
        except Exception:
            return ""

    init_domain, init_tld = _registered(initial_url)
    final_domain, final_tld = _registered(final_url)

    redirect_count = len(redirect_chain) - 1  # hops minus the final destination
    if redirect_count < 0:
        redirect_count = 0

    domain_changed = init_domain != final_domain
    tld_changed = init_tld != final_tld

    # Detect scheme changes across the chain.
    cross_scheme = False
    https_to_http = False
    prev_scheme = _scheme(initial_url)
    chain_summary: list[str] = []

    for i, hop in enumerate(redirect_chain):
        hop_url = hop.get("url", "")
        hop_scheme = _scheme(hop_url)
        status = hop.get("status_code", "?")

        chain_summary.append(
            f"Hop {i}: [{status}] {hop_url[:120]}"
        )

        if prev_scheme and hop_scheme and hop_scheme != prev_scheme:
            cross_scheme = True
            if prev_scheme == "https" and hop_scheme == "http":
                https_to_http = True
        prev_scheme = hop_scheme

    # ── Evidence signals ─────────────────────────────────────────────

    if domain_changed:
        signals.append({
            "id": "redirect_domain_changed",
            "severity": "high",
            "title": "Redirect crosses to a different domain",
            "detail": (
                f"The URL started at '{init_domain}' but the final "
                f"destination is '{final_domain}'. Cross-domain redirects "
                "are a common phishing technique."
            ),
        })

    if tld_changed:
        signals.append({
            "id": "redirect_tld_changed",
            "severity": "medium",
            "title": "Redirect crosses to a different TLD",
            "detail": (
                f"TLD changed from '.{init_tld}' to '.{final_tld}' "
                "during the redirect chain."
            ),
        })

    if https_to_http:
        signals.append({
            "id": "redirect_https_to_http",
            "severity": "high",
            "title": "Redirect downgrades from HTTPS to HTTP",
            "detail": (
                "One or more redirects changed the scheme from HTTPS to HTTP, "
                "removing transport encryption."
            ),
        })

    if redirect_count >= 3:
        signals.append({
            "id": "many_redirects",
            "severity": "medium",
            "title": f"Redirect chain has {redirect_count} hops",
            "detail": (
                f"The URL required {redirect_count} redirect(s) to reach the "
                "final destination. Long chains are sometimes used to obscure "
                "the true destination."
            ),
        })

    return {
        "redirect_count": redirect_count,
        "initial_domain": init_domain,
        "final_domain": final_domain,
        "initial_tld": init_tld,
        "final_tld": final_tld,
        "domain_changed": domain_changed,
        "tld_changed": tld_changed,
        "cross_scheme": cross_scheme,
        "https_to_http": https_to_http,
        "chain_summary": chain_summary,
        "signals": signals,
    }
