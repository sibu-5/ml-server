"""
Domain spoofing / brand look-alike detection tests — V3.
"""

import pytest
from app.intelligence.brands import detect_brand_impersonation, _levenshtein
from app.analyzers.spoofing import analyse_spoofing


class TestLevenshtein:

    def test_identical(self):
        assert _levenshtein("paypal", "paypal") == 0

    def test_one_substitution(self):
        assert _levenshtein("paypa1", "paypal") == 1

    def test_two_substitutions(self):
        assert _levenshtein("micr0soft", "microsoft") == 1  # '0' → 'o'

    def test_completely_different(self):
        d = _levenshtein("google", "amazon")
        assert d > 3


class TestBrandImpersonation:

    def _signal_ids(self, signals):
        return [s["id"] for s in signals]

    def test_paypal_typosquatting(self):
        """paypa1.com should be detected as a PayPal look-alike."""
        signals = detect_brand_impersonation("paypa1.com", "", "/")
        assert any(
            "brand_impersonation" in s["id"] or "possible_brand_impersonation" in s["id"]
            for s in signals
        )

    def test_microsoft_typosquatting(self):
        """micr0soft.com — '0' is a numeric substitution for 'o'."""
        signals = detect_brand_impersonation("micr0soft.com", "", "/")
        # normalises to 'microsoft' which matches brand key
        ids = self._signal_ids(signals)
        assert any("impersonation" in i for i in ids)

    def test_google_typosquatting(self):
        """goog1e.com — '1' replaces 'l'."""
        signals = detect_brand_impersonation("goog1e.com", "", "/")
        ids = self._signal_ids(signals)
        assert any("impersonation" in i for i in ids)

    def test_legit_google_no_signal(self):
        """google.com itself should not be flagged."""
        signals = detect_brand_impersonation("google.com", "", "/")
        # Should be empty — it's the legit domain.
        assert len(signals) == 0

    def test_legit_microsoft_no_signal(self):
        signals = detect_brand_impersonation("microsoft.com", "", "/")
        assert len(signals) == 0

    def test_brand_in_subdomain(self):
        """paypal.phishing-site.com — brand in subdomain."""
        signals = detect_brand_impersonation(
            "phishing-site.com", "paypal", "/account"
        )
        ids = self._signal_ids(signals)
        assert "brand_in_subdomain" in ids

    def test_brand_in_path(self):
        """random.com/paypal/login — brand keyword in path."""
        signals = detect_brand_impersonation("random.com", "", "/paypal/login")
        ids = self._signal_ids(signals)
        assert "brand_in_path" in ids

    def test_unrelated_domain_clean(self):
        signals = detect_brand_impersonation("unrelatedcorp.com", "", "/")
        assert len(signals) == 0


class TestSpoofingAnalyser:

    def test_analyse_spoofing_paypa1(self):
        result = analyse_spoofing("https://paypa1.com/login")
        assert result["registered_domain"] == "paypa1.com"
        assert len(result["signals"]) > 0

    def test_analyse_spoofing_clean(self):
        result = analyse_spoofing("https://example.com/home")
        assert len(result["signals"]) == 0
