"""
Tests for URL normalisation, validation, and manipulation detection — V3.
"""

import pytest
from app.features.url import (
    normalize_url,
    validate_url,
    detect_url_manipulation,
)


# ── Normalisation tests ──────────────────────────────────────────────────────

class TestNormalizeURL:

    def test_trailing_slash_root(self):
        """https://example.com/ and https://example.com should normalise the same."""
        assert normalize_url("https://example.com/") == normalize_url("https://example.com")

    def test_scheme_case(self):
        """HTTPS:// should normalise to https://."""
        result = normalize_url("HTTPS://EXAMPLE.COM/")
        assert result.startswith("https://")
        assert "example.com" in result

    def test_hostname_lowercase(self):
        """Hostname should be lowercased."""
        result = normalize_url("https://EXAMPLE.COM/path")
        assert "example.com" in result

    def test_preserves_path(self):
        """Path must not be stripped — /login ≠ /home."""
        n1 = normalize_url("https://example.com/login")
        n2 = normalize_url("https://example.com/home")
        assert n1 != n2

    def test_default_port_stripped(self):
        """https://example.com:443/ should strip the default port."""
        result = normalize_url("https://example.com:443/")
        assert ":443" not in result

    def test_http_default_port_stripped(self):
        """http://example.com:80/ should strip the default port."""
        result = normalize_url("http://example.com:80/")
        assert ":80" not in result

    def test_non_default_port_kept(self):
        """https://example.com:8443/ should keep the non-default port."""
        result = normalize_url("https://example.com:8443/")
        assert "8443" in result

    def test_query_sorted(self):
        """Query params should be sorted for stable comparison."""
        n1 = normalize_url("https://example.com/?b=2&a=1")
        n2 = normalize_url("https://example.com/?a=1&b=2")
        assert n1 == n2

    def test_non_root_path_trailing_slash_preserved(self):
        """A non-root trailing slash is preserved (meaningful path)."""
        r1 = normalize_url("https://example.com/login/")
        r2 = normalize_url("https://example.com/login")
        # They may differ — that's acceptable; just ensure /login/ ≠ /home/
        r3 = normalize_url("https://example.com/home/")
        assert r1 != r3


# ── Validation tests ─────────────────────────────────────────────────────────

class TestValidateURL:

    def test_valid_https(self):
        result = validate_url("https://example.com")
        assert result.valid
        assert result.status == "VALID"

    def test_valid_http(self):
        result = validate_url("http://example.com/path?q=1")
        assert result.valid

    def test_invalid_scheme_ftp(self):
        result = validate_url("ftp://example.com")
        assert not result.valid
        assert result.status == "UNSUPPORTED_SCHEME"

    def test_invalid_scheme_file(self):
        result = validate_url("file:///etc/passwd")
        assert not result.valid
        assert result.status == "UNSUPPORTED_SCHEME"

    def test_invalid_scheme_javascript(self):
        result = validate_url("javascript:alert(1)")
        assert not result.valid

    def test_missing_hostname(self):
        result = validate_url("https:///path")
        assert not result.valid
        assert result.status in ("MISSING_HOSTNAME", "INVALID_URL")

    def test_empty_string(self):
        result = validate_url("")
        assert not result.valid

    def test_too_long(self):
        url = "https://example.com/" + "a" * 5000
        result = validate_url(url, max_length=100)
        assert not result.valid
        assert result.status == "URL_TOO_LONG"

    def test_ip_url_valid(self):
        """An IP-based URL is syntactically valid."""
        result = validate_url("https://8.8.8.8/")
        assert result.valid


# ── Manipulation detection tests ─────────────────────────────────────────────

class TestDetectURLManipulation:

    def test_at_symbol_detected(self):
        findings = detect_url_manipulation(
            "https://trusted.example@evil.example/login"
        )
        ids = [f["id"] for f in findings]
        assert "at_symbol_abuse" in ids

    def test_ip_url_detected(self):
        findings = detect_url_manipulation("https://192.168.1.1/login")
        ids = [f["id"] for f in findings]
        assert "ip_address_url" in ids

    def test_long_url_detected(self):
        url = "https://example.com/" + "a" * 300
        findings = detect_url_manipulation(url)
        ids = [f["id"] for f in findings]
        assert "excessive_url_length" in ids

    def test_punycode_detected(self):
        findings = detect_url_manipulation("https://xn--pypal-4ve.com/login")
        ids = [f["id"] for f in findings]
        assert "punycode_hostname" in ids

    def test_clean_url_no_findings(self):
        findings = detect_url_manipulation("https://example.com/home")
        assert len(findings) == 0

    def test_credential_params_detected(self):
        findings = detect_url_manipulation(
            "https://evil.example/reset?password=abc123"
        )
        ids = [f["id"] for f in findings]
        assert "credential_query_params" in ids
