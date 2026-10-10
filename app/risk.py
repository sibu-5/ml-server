"""
Risk engine — V3.

Combines ALL evidence signals into a single risk_score and verdict.

Design principles
-----------------
- Every risk contribution is traceable to an evidence item.
- Score is deterministic given the same inputs.
- The XGBoost probability is ONE signal among many.
- A site being UNREACHABLE is NOT the same as being SAFE.
- DNS failure / timeout / SSL error does NOT upgrade verdict to SAFE.
- Errors return explicit verdicts, never "legitimate".

Verdict values (stable — Laptop 1 must be able to parse these)
---------------------------------------------------------------
INVALID_URL       — URL failed validation
UNREACHABLE       — site could not be reached
SAFE              — no significant signals
SUSPICIOUS        — moderate evidence of risk
PHISHING          — strong evidence of phishing
ANALYSIS_ERROR    — unexpected exception during analysis

Backward compatibility
----------------------
The legacy calculate_risk(probability, detections) function is
preserved for the /predict and /api/v1/models/{name}/predict endpoints
so Laptop 1 is not broken.
"""

from __future__ import annotations

from app.evidence import EvidenceItem, SEVERITY_WEIGHTS, total_evidence_weight


# ── Legacy risk function — DO NOT REMOVE (backward compatibility) ──────────

def calculate_risk(
    probability: float,
    detections: list[str],
) -> tuple[float, str]:
    """
    Original V1/V2 risk function — preserved for backward compatibility.

    Used by:
    - POST /predict
    - POST /api/v1/models/{model_name}/predict

    DO NOT change the signature or return type.
    """
    rule_bonus = min(len(detections) * 4.0, 20.0)
    score = min(100.0, probability * 80.0 + rule_bonus)

    if score >= 75:
        verdict = "phishing"
    elif score >= 45:
        verdict = "suspicious"
    else:
        verdict = "legitimate"

    return round(score, 2), verdict


# ── V3 risk engine ──────────────────────────────────────────────────────────

# Contribution of the XGBoost probability to the total risk score.
# This is scaled to at most 40 points so that evidence can override it.
_ML_MAX_CONTRIBUTION = 40.0

# Maximum contribution from evidence signals (before cap).
# Critical+high signals add up quickly; cap ensures score stays ≤ 100.
_EVIDENCE_SCALE = 1.0

# Score thresholds for verdicts.
_PHISHING_THRESHOLD = 65.0
_SUSPICIOUS_THRESHOLD = 35.0


def calculate_risk_v3(
    *,
    ml_probability: float,
    evidence_items: list[EvidenceItem],
    website_status: str,
    url_valid: bool,
) -> dict:
    """
    V3 risk engine — combines ML probability with evidence signals.

    Parameters
    ----------
    ml_probability : float [0,1] from XGBoost. Use 0.0 if model unavailable.
    evidence_items : Collected EvidenceItem list from evidence.collect().
    website_status : Reachability status string from reachability.fetch_url().
    url_valid      : Whether the URL passed validation.

    Returns
    -------
    {
        "risk_score"  : float [0, 100]
        "verdict"     : str
        "severity"    : str  — "critical" | "high" | "medium" | "low" | "none"
        "evidence_ids": list[str]  — IDs that contributed to score
        "score_breakdown": dict   — how each component contributed
    }
    """

    # ── Structural overrides (before scoring) ────────────────────────
    if not url_valid:
        return {
            "risk_score": 0.0,
            "verdict": "INVALID_URL",
            "severity": "none",
            "evidence_ids": [],
            "score_breakdown": {"reason": "URL failed validation"},
        }

    if website_status in ("ssrf_blocked",):
        # SSRF attempts get maximum score.
        return {
            "risk_score": 100.0,
            "verdict": "PHISHING",
            "severity": "critical",
            "evidence_ids": ["ssrf_blocked"],
            "score_breakdown": {"ssrf_blocked": 100.0},
        }

    # ── Score computation ─────────────────────────────────────────────

    # ML contribution (max 40 points).
    ml_contribution = min(ml_probability * _ML_MAX_CONTRIBUTION * 2, _ML_MAX_CONTRIBUTION)

    # Evidence contribution.
    evidence_weight = total_evidence_weight(evidence_items)
    evidence_contribution = min(evidence_weight * _EVIDENCE_SCALE, 60.0)

    raw_score = ml_contribution + evidence_contribution
    score = min(100.0, round(raw_score, 2))

    evidence_ids = [item.id for item in evidence_items if item.weight > 0]

    breakdown = {
        "ml_probability": round(ml_probability, 4),
        "ml_contribution": round(ml_contribution, 2),
        "evidence_weight": round(evidence_weight, 2),
        "evidence_contribution": round(evidence_contribution, 2),
        "raw_score": round(raw_score, 2),
    }

    # ── Verdict ────────────────────────────────────────────────────────
    # If site was unreachable, report that explicitly regardless of score.
    if website_status in (
        "dns_not_found", "timeout", "connection_failed",
        "http_error", "unreachable",
    ):
        # Still compute and return score (URL may already look bad).
        verdict = "UNREACHABLE"
        if score >= _PHISHING_THRESHOLD:
            # High ML/evidence score + unreachable = suspicious at minimum.
            severity = "high"
        elif score >= _SUSPICIOUS_THRESHOLD:
            severity = "medium"
        else:
            severity = "low"

        return {
            "risk_score": score,
            "verdict": verdict,
            "severity": severity,
            "evidence_ids": evidence_ids,
            "score_breakdown": breakdown,
        }

    if score >= _PHISHING_THRESHOLD:
        verdict = "PHISHING"
        severity = "critical" if score >= 85 else "high"
    elif score >= _SUSPICIOUS_THRESHOLD:
        verdict = "SUSPICIOUS"
        severity = "medium"
    else:
        verdict = "SAFE"
        severity = "none" if score < 10 else "low"

    # Upgrade to at least SUSPICIOUS if a critical evidence item exists
    # regardless of numeric score.
    has_critical = any(item.severity == "critical" for item in evidence_items)
    if has_critical and verdict == "SAFE":
        verdict = "SUSPICIOUS"
        severity = "high"
        score = max(score, _SUSPICIOUS_THRESHOLD)

    return {
        "risk_score": score,
        "verdict": verdict,
        "severity": severity,
        "evidence_ids": evidence_ids,
        "score_breakdown": breakdown,
    }
