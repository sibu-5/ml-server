"""
URL feature extraction — V3.

Keeps the exact same FEATURE_COLUMNS list and vectorize() contract
so the existing XGBoost model continues to work unchanged.

New in V3:
  - normalize_url()        : canonical form comparison
  - validate_url()         : structured validation result
  - detect_url_manipulation(): @ abuse, credential params, IP URLs, …
"""

import math
import re
import ipaddress
from collections import Counter
from urllib.parse import (
    parse_qsl,
    quote,
    unquote,
    urlparse,
    urlunparse,
)

import tldextract


# ── Feature columns (unchanged — XGBoost contract) ─────────────────────────
FEATURE_COLUMNS = [
    "url_length",
    "domain_length",
    "subdomain_count",
    "path_length",
    "query_length",
    "digit_count",
    "special_char_count",
    "dot_count",
    "hyphen_count",
    "slash_count",
    "has_https",
    "has_ip",
    "has_at_symbol",
    "has_punycode",
    "has_encoded_chars",
    "has_fragment",
    "suspicious_keyword_count",
    "hostname_entropy",
    "query_parameter_count",
    "path_entropy",
    "subdomain_entropy",
    "longest_token_length",
    "digit_letter_ratio",
    "has_suspicious_tld",
]

SUSPICIOUS_KEYWORDS = {
    "login",
    "signin",
    "verify",
    "verification",
    "secure",
    "account",
    "update",
    "confirm",
    "password",
    "wallet",
    "bank",
    "billing",
    "invoice",
    "payment",
    "auth",
    "authenticate",
    "credential",
}

SUSPICIOUS_TLDS = {
    "zip",
    "mov",
    "click",
    "top",
    "work",
    "gq",
    "tk",
    "ml",
    "cf",
    "ga",
    "icu",
    "buzz",
    "rest",
    "cam",
    "fit",
    "xyz",
    "pw",
    "cc",
    "su",
    "ws",
    "biz",
}

# Credential-related query parameter names that may indicate phishing.
CREDENTIAL_PARAMS = {
    "password",
    "passwd",
    "pass",
    "pwd",
    "token",
    "secret",
    "key",
    "apikey",
    "api_key",
    "auth",
    "otp",
    "pin",
    "credential",
}

# Default ports for schemes — used to strip redundant port.
_DEFAULT_PORTS = {"http": 80, "https": 443}


# ── Helpers ─────────────────────────────────────────────────────────────────

def entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    total = len(value)
    return round(
        -sum(
            (c / total) * math.log2(c / total)
            for c in counts.values()
        ),
        6,
    )


def _is_ip_address(hostname: str) -> bool:
    """Return True if *hostname* is a bare IP (v4 or v6)."""
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def is_ip(hostname: str) -> int:
    return 1 if _is_ip_address(hostname) else 0


# ── URL normalisation ────────────────────────────────────────────────────────

def normalize_url(raw_url: str) -> str:
    """
    Return a canonical form of *raw_url*.

    Normalisation steps
    -------------------
    1. Strip leading/trailing whitespace.
    2. Lowercase the scheme and hostname.
    3. Remove the default port (80 for http, 443 for https).
    4. Percent-encode only characters that must be encoded; decode
       over-encoded but safe characters so that trivially equivalent
       URLs compare equal.
    5. Normalise the path:
       - Collapse double slashes (except after the authority).
       - Remove a *trailing* slash only when the path == "/" (i.e. the
         root).  A non-root trailing slash is preserved so that
         /login/ stays distinct from /login.
    6. Sort query parameters for stable comparison.
    7. Preserve the fragment (it may carry payload in phishing URLs).
    8. Handle IDNA / punycode hostnames without decoding them (they
       are already canonical for security purposes).

    The function never discards meaningful path / query information.
    """
    url = raw_url.strip()
    try:
        parsed = urlparse(url)
    except Exception:
        return url  # Return as-is; validate_url() will reject it.

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc

    # Lowercase hostname, preserve port and user-info.
    hostname = (parsed.hostname or "").lower()
    port = parsed.port

    # Reconstruct netloc: userinfo@host:port
    userinfo = ""
    if parsed.username:
        userinfo = parsed.username
        if parsed.password:
            userinfo += f":{parsed.password}"
        userinfo += "@"

    if port and _DEFAULT_PORTS.get(scheme) == port:
        port = None  # Strip redundant default port.

    if port:
        netloc = f"{userinfo}{hostname}:{port}"
    else:
        netloc = f"{userinfo}{hostname}"

    # Path normalisation.
    path = parsed.path
    # Collapse consecutive slashes in the path (but keep leading /).
    path = re.sub(r"/{2,}", "/", path)
    # Remove trailing slash only from the bare root "/".
    if path == "/":
        path = ""

    # Query: sort parameters for stable comparison.
    query = parsed.query
    if query:
        params = parse_qsl(query, keep_blank_values=True)
        params.sort()
        query = "&".join(
            f"{quote(k, safe='~-._')}={quote(v, safe='~-._')}"
            for k, v in params
        )

    normalized = urlunparse(
        (scheme, netloc, path, parsed.params, query, parsed.fragment)
    )
    return normalized


# ── URL validation ───────────────────────────────────────────────────────────

class URLValidationResult:
    """Structured result of URL validation."""

    __slots__ = ("valid", "status", "detail", "parsed", "hostname")

    def __init__(
        self,
        valid: bool,
        status: str,
        detail: str = "",
        parsed=None,
        hostname: str = "",
    ):
        self.valid = valid
        self.status = status
        self.detail = detail
        self.parsed = parsed
        self.hostname = hostname

    def as_dict(self) -> dict:
        return {
            "valid": self.valid,
            "status": self.status,
            "detail": self.detail,
            "hostname": self.hostname,
        }


def validate_url(raw_url: str, max_length: int = 4096) -> URLValidationResult:
    """
    Validate *raw_url* and return a structured result.

    Possible status values
    ----------------------
    VALID
    INVALID_URL            — could not parse
    UNSUPPORTED_SCHEME     — not http or https
    MISSING_HOSTNAME       — no host part
    INVALID_HOSTNAME       — hostname contains illegal characters
    INVALID_PORT           — port out of range
    URL_TOO_LONG           — exceeds max_length
    MALFORMED_ENCODING     — bad percent-encoding
    """
    if not raw_url or not isinstance(raw_url, str):
        return URLValidationResult(
            False,
            "INVALID_URL",
            "URL must be a non-empty string.",
        )

    if len(raw_url) > max_length:
        return URLValidationResult(
            False,
            "URL_TOO_LONG",
            f"URL exceeds maximum length of {max_length} characters.",
        )

    url = raw_url.strip()

    try:
        parsed = urlparse(url)
    except Exception as exc:
        return URLValidationResult(
            False,
            "INVALID_URL",
            f"Could not parse URL: {exc}",
        )

    # Scheme check.
    if parsed.scheme.lower() not in ("http", "https"):
        return URLValidationResult(
            False,
            "UNSUPPORTED_SCHEME",
            (
                f"Scheme '{parsed.scheme}' is not supported. "
                "Only http and https are allowed."
            ),
        )

    # Hostname check.
    hostname = parsed.hostname or ""
    if not hostname:
        return URLValidationResult(
            False,
            "MISSING_HOSTNAME",
            "URL has no hostname.",
        )

    # Port range check.
    try:
        port = parsed.port  # raises ValueError if out of range
        if port is not None and not (1 <= port <= 65535):
            raise ValueError("port out of range")
    except ValueError:
        return URLValidationResult(
            False,
            "INVALID_PORT",
            f"Port number is invalid.",
        )

    # Percent-encoding sanity check.
    try:
        unquote(url, errors="strict")
    except Exception:
        return URLValidationResult(
            False,
            "MALFORMED_ENCODING",
            "URL contains invalid percent-encoding sequences.",
        )

    # Hostname character check — allow labels, hyphens, dots, brackets for IPv6.
    # IP addresses are legitimate (xn--… for IDN, digits, colons for v6).
    hostname_clean = hostname.strip("[]")  # strip IPv6 brackets
    if not re.match(
        r"^(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)*"
        r"(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?)$",
        hostname_clean,
        re.IGNORECASE,
    ):
        # Accept numeric IPv4 / IPv6 addresses too.
        if not _is_ip_address(hostname_clean):
            return URLValidationResult(
                False,
                "INVALID_HOSTNAME",
                f"Hostname '{hostname}' contains illegal characters.",
            )

    return URLValidationResult(
        True,
        "VALID",
        parsed=parsed,
        hostname=hostname.lower(),
    )


# ── URL manipulation detection ───────────────────────────────────────────────

def detect_url_manipulation(raw_url: str) -> list[dict]:
    """
    Return a list of manipulation evidence dicts found in *raw_url*.

    Each dict has keys: id, severity, title, detail.
    """
    findings: list[dict] = []
    url = raw_url.strip()

    try:
        parsed = urlparse(url)
    except Exception:
        return findings

    # ── @ symbol abuse ──────────────────────────────────────────────
    if "@" in (parsed.netloc or ""):
        # Real hostname is *after* the @ sign.
        real_host = parsed.hostname or ""
        visual_host = parsed.netloc.split("@", 1)[0]
        findings.append({
            "id": "at_symbol_abuse",
            "severity": "high",
            "title": "URL contains @ symbol (authority abuse)",
            "detail": (
                f"The @ symbol causes browsers to use '{real_host}' "
                f"as the real host while displaying '{visual_host}' to users."
            ),
        })

    # ── IP address instead of domain ────────────────────────────────
    hostname = (parsed.hostname or "").lower()
    if _is_ip_address(hostname):
        findings.append({
            "id": "ip_address_url",
            "severity": "medium",
            "title": "URL uses a bare IP address instead of a domain name",
            "detail": (
                f"The URL resolves directly to IP address '{hostname}', "
                "which is unusual for legitimate websites and may hide the real destination."
            ),
        })

    # ── Excessive length ─────────────────────────────────────────────
    if len(url) > 200:
        findings.append({
            "id": "excessive_url_length",
            "severity": "low",
            "title": "Unusually long URL",
            "detail": (
                f"The URL is {len(url)} characters long. "
                "Excessively long URLs are a common obfuscation tactic."
            ),
        })

    # ── Excessive subdomains ─────────────────────────────────────────
    if not _is_ip_address(hostname):
        parts = hostname.split(".")
        if len(parts) >= 5:
            findings.append({
                "id": "excessive_subdomains",
                "severity": "medium",
                "title": "URL has an unusually large number of subdomains",
                "detail": (
                    f"The hostname '{hostname}' has {len(parts) - 2} subdomains, "
                    "which may be used to make the URL look legitimate."
                ),
            })

    # ── Encoded characters ───────────────────────────────────────────
    encoded_count = len(re.findall(r"%[0-9a-fA-F]{2}", url))
    if encoded_count >= 3:
        findings.append({
            "id": "excessive_encoding",
            "severity": "medium",
            "title": "URL contains multiple percent-encoded characters",
            "detail": (
                f"Found {encoded_count} percent-encoded sequences. "
                "Excessive encoding is used to hide the actual URL content."
            ),
        })

    # ── Credential query parameters ──────────────────────────────────
    query = parsed.query or ""
    param_names = {k.lower() for k, _ in parse_qsl(query, keep_blank_values=True)}
    cred_params = param_names & CREDENTIAL_PARAMS
    if cred_params:
        findings.append({
            "id": "credential_query_params",
            "severity": "high",
            "title": "URL query string contains credential-related parameters",
            "detail": (
                f"Sensitive parameter names detected in query: "
                f"{', '.join(sorted(cred_params))}."
            ),
        })

    # ── Multiple scheme-like patterns ────────────────────────────────
    extra_schemes = re.findall(r"(?:https?|ftp)://", url[8:])  # skip the real scheme
    if extra_schemes:
        findings.append({
            "id": "embedded_url",
            "severity": "high",
            "title": "URL contains an embedded URL",
            "detail": (
                "A secondary URL pattern was found inside the URL, "
                "which may be used to redirect users to a malicious destination."
            ),
        })

    # ── Punycode / IDN ───────────────────────────────────────────────
    if "xn--" in hostname:
        findings.append({
            "id": "punycode_hostname",
            "severity": "medium",
            "title": "Hostname uses punycode / internationalized domain name",
            "detail": (
                f"The punycode hostname '{hostname}' may visually resemble "
                "a trusted brand using Unicode characters."
            ),
        })

    return findings


# ── Core feature extraction (unchanged contract) ─────────────────────────────

def extract_url_features(url: str) -> dict:
    """
    Extract numerical features used by the XGBoost phishing model.

    The returned dict keys and types are UNCHANGED from V1/V2 so
    that existing model inference continues to work.
    """
    parsed = urlparse(url.strip())
    hostname = (parsed.hostname or "").lower()

    extracted = tldextract.extract(hostname)

    subdomain = extracted.subdomain or ""
    suffix = extracted.suffix or ""

    raw = url.strip()
    path = parsed.path or ""
    query = parsed.query or ""

    tokens = [
        token
        for token in re.split(r"[^A-Za-z0-9]+", raw)
        if token
    ]

    longest_token = max(
        (len(token) for token in tokens),
        default=0,
    )

    digits = sum(character.isdigit() for character in raw)
    letters = sum(character.isalpha() for character in raw)

    keyword_count = sum(
        keyword in raw.lower()
        for keyword in SUSPICIOUS_KEYWORDS
    )

    return {
        "url_length": len(raw),
        "domain_length": len(hostname),
        "subdomain_count": (
            0 if not subdomain
            else len(subdomain.split("."))
        ),
        "path_length": len(path),
        "query_length": len(query),
        "digit_count": digits,
        "special_char_count": sum(
            not character.isalnum()
            for character in raw
        ),
        "dot_count": raw.count("."),
        "hyphen_count": raw.count("-"),
        "slash_count": raw.count("/"),
        "has_https": int(parsed.scheme.lower() == "https"),
        "has_ip": is_ip(hostname),
        "has_at_symbol": int("@" in raw),
        "has_punycode": int("xn--" in hostname),
        "has_encoded_chars": int(
            bool(re.search(r"%[0-9a-fA-F]{2}", raw))
        ),
        "has_fragment": int(bool(parsed.fragment)),
        "suspicious_keyword_count": keyword_count,
        "hostname_entropy": entropy(hostname),
        "query_parameter_count": len(
            parse_qsl(
                query,
                keep_blank_values=True,
            )
        ),
        "path_entropy": entropy(path),
        "subdomain_entropy": entropy(subdomain),
        "longest_token_length": longest_token,
        "digit_letter_ratio": round(
            digits / max(letters, 1),
            6,
        ),
        "has_suspicious_tld": int(
            suffix.lower() in SUSPICIOUS_TLDS
        ),
    }


def vectorize(features: dict) -> list[float]:
    """Convert feature dict to ordered float list for model inference."""
    return [
        float(features[column])
        for column in FEATURE_COLUMNS
    ]
