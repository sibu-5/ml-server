"""
Deterministic explanation engine — V3.

Converts the risk engine output + evidence items into a human-readable
explanation WITHOUT calling any external service.

This is the fallback explanation layer.  When the LLM is available it
can augment these explanations.  When the LLM is unavailable this
module provides the full explanation on its own.

The LLM MUST NOT override or contradict these deterministic outputs.
"""

from __future__ import annotations

from app.evidence import EvidenceItem


def build_explanation(
    *,
    verdict: str,
    risk_score: float,
    ml_probability: float,
    evidence_items: list[EvidenceItem],
    website_status: str,
    normalized_url: str,
) -> dict:
    """
    Build a deterministic explanation for the given analysis.

    Returns
    -------
    {
        "summary"      : str  — one-paragraph explanation
        "reasons"      : list[str]  — bullet-point evidence descriptions
        "recommendation": list[str] — actionable advice
        "certainty"    : str  — "high" | "medium" | "low" | "unknown"
    }
    """
    reasons: list[str] = []
    recommendations: list[str] = []

    # ── Collect reason bullets from evidence ──────────────────────────
    severity_order = ["critical", "high", "medium", "low", "info"]
    sorted_evidence = sorted(
        evidence_items,
        key=lambda e: severity_order.index(e.severity)
        if e.severity in severity_order else 99,
    )

    for item in sorted_evidence:
        if item.severity == "info":
            continue  # Info signals don't add to user-visible reasons.
        reasons.append(f"[{item.severity.upper()}] {item.title}: {item.details}")

    # ── Verdict-specific summary ──────────────────────────────────────
    verdict_upper = verdict.upper()

    if verdict_upper == "INVALID_URL":
        summary = (
            "The submitted URL is malformed or uses an unsupported scheme. "
            "It cannot be analysed. Do not visit this URL."
        )
        recommendations = [
            "Double-check the URL for typos.",
            "Only submit http:// or https:// URLs.",
        ]
        certainty = "high"

    elif verdict_upper == "UNREACHABLE":
        summary = (
            f"The website at '{normalized_url}' could not be reached "
            f"(status: {website_status}). "
            "Website content could not be analysed. "
            "This does NOT mean the URL is safe — phishing URLs are "
            "often unreachable because they are taken down quickly."
        )
        if evidence_items:
            summary += (
                f" The URL itself exhibited {len(evidence_items)} "
                "suspicious signal(s) based on static analysis."
            )
        recommendations = [
            "Do not assume an unreachable URL is safe.",
            "If you received this URL unexpectedly, treat it as suspicious.",
        ]
        certainty = "low"

    elif verdict_upper == "PHISHING":
        num_high = sum(
            1 for e in evidence_items if e.severity in ("critical", "high")
        )
        summary = (
            f"This URL exhibits strong indicators of a phishing attack "
            f"(risk score: {risk_score:.0f}/100, ML probability: "
            f"{ml_probability:.0%}). "
            f"{num_high} high-severity signal(s) were detected."
        )
        if reasons:
            top = reasons[0]
            summary += f" The primary indicator is: {top}"
        recommendations = [
            "Do NOT enter any personal information on this website.",
            "Do NOT click any links within this page.",
            "Report this URL to your security team.",
            "If you already submitted credentials, change your passwords immediately.",
        ]
        certainty = "high" if risk_score >= 80 else "medium"

    elif verdict_upper == "SUSPICIOUS":
        summary = (
            f"This URL shows moderate phishing indicators "
            f"(risk score: {risk_score:.0f}/100, ML probability: "
            f"{ml_probability:.0%}). "
            "Not enough evidence to definitively classify as phishing, "
            "but exercise caution."
        )
        recommendations = [
            "Verify the URL's legitimacy before submitting any personal information.",
            "Check the domain carefully for typos or character substitutions.",
            "Contact the supposed organisation through official channels to verify.",
        ]
        certainty = "medium"

    elif verdict_upper == "SAFE":
        summary = (
            f"No significant phishing indicators were detected "
            f"(risk score: {risk_score:.0f}/100, ML probability: "
            f"{ml_probability:.0%}). "
            "The domain, redirects, and page structure appear consistent "
            "with a normal website."
        )
        recommendations = [
            "Always verify you are on the correct domain before logging in.",
            "Use a password manager to prevent credential submission on wrong sites.",
        ]
        certainty = "medium"  # We cannot guarantee safety.

    elif verdict_upper == "ANALYSIS_ERROR":
        summary = (
            "An error occurred during analysis. "
            "The result is inconclusive. Do not treat this as confirmation of safety."
        )
        recommendations = [
            "Try the request again.",
            "If the error persists, contact the security team.",
        ]
        certainty = "unknown"

    else:
        summary = (
            f"Analysis returned verdict: {verdict}. "
            "Risk score: {risk_score:.0f}/100."
        )
        recommendations = []
        certainty = "unknown"

    return {
        "summary": summary,
        "reasons": reasons,
        "recommendation": recommendations,
        "certainty": certainty,
    }
