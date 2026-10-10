"""
Risk engine tests — V3.
"""

import pytest
from app.evidence import EvidenceItem
from app.risk import calculate_risk, calculate_risk_v3


# ── Legacy risk engine tests (backward compatibility) ────────────────────────

class TestLegacyRisk:

    def test_high_prob_phishing(self):
        score, verdict = calculate_risk(0.95, [])
        assert verdict == "phishing"
        assert score >= 75

    def test_low_prob_legitimate(self):
        score, verdict = calculate_risk(0.05, [])
        assert verdict == "legitimate"

    def test_detections_increase_score(self):
        score1, _ = calculate_risk(0.3, [])
        score2, _ = calculate_risk(0.3, ["signal1", "signal2", "signal3"])
        assert score2 > score1

    def test_score_capped_at_100(self):
        score, _ = calculate_risk(1.0, ["a"] * 20)
        assert score <= 100.0


# ── V3 risk engine tests ─────────────────────────────────────────────────────

def _make_item(severity: str) -> EvidenceItem:
    return EvidenceItem(
        id=f"test_{severity}",
        severity=severity,
        title="Test signal",
        details="Test",
        source="test",
    )


class TestV3Risk:

    def test_invalid_url(self):
        result = calculate_risk_v3(
            ml_probability=0.5,
            evidence_items=[],
            website_status="live",
            url_valid=False,
        )
        assert result["verdict"] == "INVALID_URL"
        assert result["risk_score"] == 0.0

    def test_unreachable_does_not_become_safe(self):
        result = calculate_risk_v3(
            ml_probability=0.1,
            evidence_items=[],
            website_status="dns_not_found",
            url_valid=True,
        )
        # Must NOT be SAFE
        assert result["verdict"] == "UNREACHABLE"

    def test_unreachable_with_high_ml_score(self):
        result = calculate_risk_v3(
            ml_probability=0.9,
            evidence_items=[_make_item("high"), _make_item("critical")],
            website_status="timeout",
            url_valid=True,
        )
        assert result["verdict"] == "UNREACHABLE"
        assert result["risk_score"] > 30  # Still has meaningful score.

    def test_ssrf_blocked_is_phishing(self):
        result = calculate_risk_v3(
            ml_probability=0.0,
            evidence_items=[],
            website_status="ssrf_blocked",
            url_valid=True,
        )
        assert result["verdict"] == "PHISHING"
        assert result["risk_score"] == 100.0

    def test_high_evidence_phishing(self):
        items = [
            _make_item("critical"),
            _make_item("high"),
            _make_item("high"),
        ]
        result = calculate_risk_v3(
            ml_probability=0.8,
            evidence_items=items,
            website_status="live",
            url_valid=True,
        )
        assert result["verdict"] == "PHISHING"

    def test_low_evidence_safe(self):
        result = calculate_risk_v3(
            ml_probability=0.05,
            evidence_items=[],
            website_status="live",
            url_valid=True,
        )
        assert result["verdict"] == "SAFE"

    def test_critical_signal_not_safe(self):
        """A single critical signal must never produce a SAFE verdict."""
        result = calculate_risk_v3(
            ml_probability=0.0,
            evidence_items=[_make_item("critical")],
            website_status="live",
            url_valid=True,
        )
        assert result["verdict"] != "SAFE"

    def test_moderate_evidence_suspicious(self):
        items = [_make_item("medium"), _make_item("low")]
        result = calculate_risk_v3(
            ml_probability=0.3,
            evidence_items=items,
            website_status="live",
            url_valid=True,
        )
        assert result["verdict"] in ("SUSPICIOUS", "PHISHING")
