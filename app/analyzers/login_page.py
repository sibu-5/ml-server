"""
Fake login page detector — V3.

Combines signals from:
- HTML analysis (login form, password field, email field, etc.)
- Domain/spoofing analysis (brand impersonation)
- Redirect analysis (cross-domain redirects)
- URL analysis (suspicious keywords, IP address, etc.)

A page is considered a CANDIDATE fake login page when it has
MULTIPLE corroborating signals.

IMPORTANT:
  A login form ALONE is not evidence of phishing.
  Legitimate sites (banks, email services, etc.) have login forms.
  This module only raises signals when COMBINATIONS are suspicious.
"""


def analyse_login_page(
    html_analysis: dict,
    domain_analysis: dict,
    spoofing_analysis: dict,
    url_features: dict,
) -> dict:
    """
    Combine analysis results to detect fake login page patterns.

    Parameters
    ----------
    html_analysis    : Output of analyzers.website.analyse_website_html()
    domain_analysis  : Output of features.domain.analyse_domain()
    spoofing_analysis: Output of analyzers.spoofing.analyse_spoofing()
    url_features     : Output of features.url.extract_url_features()

    Return dict keys
    ----------------
    is_fake_login_candidate : bool
    confidence              : "high" | "medium" | "low" | "none"
    signals                 : list[dict]
    """
    signals: list[dict] = []

    has_login_form = html_analysis.get("has_login_form", False)
    has_password = html_analysis.get("has_password_field", False)
    has_email = html_analysis.get("has_email_field", False)
    has_external_action = html_analysis.get("has_external_form_action", False)

    brand_impersonation = any(
        s.get("id") in ("possible_brand_impersonation", "brand_in_subdomain")
        for s in spoofing_analysis.get("signals", [])
    )

    suspicious_domain = any(
        s.get("id") in ("suspicious_tld", "excessive_subdomains", "ip_based_url")
        for s in domain_analysis.get("signals", [])
    )

    has_ip = bool(url_features.get("has_ip"))
    has_at = bool(url_features.get("has_at_symbol"))
    keyword_dense = url_features.get("suspicious_keyword_count", 0) >= 2

    # ── Score and classify ───────────────────────────────────────────
    score = 0

    if has_login_form and has_password:
        score += 2
    if has_external_action:
        score += 4
        signals.append({
            "id": "fake_login_external_action",
            "severity": "critical",
            "title": "Credential form posts to a different domain",
            "detail": (
                "A login form submits credentials to an external domain. "
                "This is a hallmark of credential-harvesting phishing pages."
            ),
        })
    if brand_impersonation:
        score += 3
        signals.append({
            "id": "fake_login_brand_impersonation",
            "severity": "high",
            "title": "Login page may be impersonating a known brand",
            "detail": (
                "The page has a login form AND the domain resembles a known brand."
            ),
        })
    if has_login_form and suspicious_domain:
        score += 2
        signals.append({
            "id": "fake_login_suspicious_domain",
            "severity": "high",
            "title": "Login form on a suspicious domain",
            "detail": (
                "A login form was found on a domain with suspicious characteristics."
            ),
        })
    if has_ip and has_login_form:
        score += 3
        signals.append({
            "id": "fake_login_ip_host",
            "severity": "high",
            "title": "Login form hosted on an IP address",
            "detail": (
                "A page with a login form is served from a bare IP address "
                "rather than a domain name, which is highly unusual for "
                "legitimate services."
            ),
        })
    if has_at and has_login_form:
        score += 2
    if keyword_dense and has_login_form:
        score += 1

    if score >= 6:
        confidence = "high"
    elif score >= 3:
        confidence = "medium"
    elif score >= 1 and has_login_form:
        confidence = "low"
    else:
        confidence = "none"

    is_candidate = score >= 3

    return {
        "is_fake_login_candidate": is_candidate,
        "confidence": confidence,
        "signals": signals,
    }
