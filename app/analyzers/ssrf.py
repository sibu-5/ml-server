"""
SSRF protection layer — V3.

This module is the single authoritative SSRF guard for all outbound
HTTP requests made by the ML server.  It must be consulted:

  1. Before resolving any hostname (to catch static bad hostnames).
  2. After DNS resolution (to validate every resolved IP).
  3. Before following every redirect (hostname + resolved IPs).

Blocked address classes
-----------------------
- Loopback          127.0.0.0/8, ::1
- Private RFC1918   10/8, 172.16/12, 192.168/16
- Link-local        169.254/16, fe80::/10
- Carrier-grade NAT / Tailscale  100.64.0.0/10
- Unique-local IPv6 fc00::/7
- IPv6 multicast    ff00::/8
- Cloud metadata    169.254.169.254, 100.100.100.200 (Alibaba)
- Reserved / special-use ranges
- "localhost" hostname and any dotless hostname (single-label)

Private hostnames blocked
-------------------------
localhost, *.local, *.internal, *.localhost

All scheme validation is also performed here:
only http:// and https:// are permitted.
"""

import ipaddress
import socket
from urllib.parse import urlparse


# ── Blocked networks ────────────────────────────────────────────────────────

_BLOCKED_NETWORKS: list[ipaddress._BaseNetwork] = [
    # IPv4
    ipaddress.ip_network("0.0.0.0/8"),          # unspecified
    ipaddress.ip_network("10.0.0.0/8"),          # RFC1918 private
    ipaddress.ip_network("100.64.0.0/10"),       # carrier-grade NAT / Tailscale
    ipaddress.ip_network("127.0.0.0/8"),         # loopback
    ipaddress.ip_network("169.254.0.0/16"),      # link-local / AWS metadata
    ipaddress.ip_network("172.16.0.0/12"),       # RFC1918 private
    ipaddress.ip_network("192.0.0.0/24"),        # IETF protocol
    ipaddress.ip_network("192.0.2.0/24"),        # documentation
    ipaddress.ip_network("192.88.99.0/24"),      # 6to4 relay anycast
    ipaddress.ip_network("192.168.0.0/16"),      # RFC1918 private
    ipaddress.ip_network("198.18.0.0/15"),       # benchmarking
    ipaddress.ip_network("198.51.100.0/24"),     # documentation
    ipaddress.ip_network("203.0.113.0/24"),      # documentation
    ipaddress.ip_network("224.0.0.0/4"),         # multicast
    ipaddress.ip_network("240.0.0.0/4"),         # reserved
    ipaddress.ip_network("255.255.255.255/32"),  # broadcast
    # IPv6
    ipaddress.ip_network("::1/128"),             # loopback
    ipaddress.ip_network("::/128"),              # unspecified
    ipaddress.ip_network("::ffff:0:0/96"),       # IPv4-mapped
    ipaddress.ip_network("64:ff9b::/96"),        # IPv4-IPv6 translation
    ipaddress.ip_network("100::/64"),            # discard
    ipaddress.ip_network("2001::/23"),           # IETF protocol
    ipaddress.ip_network("2001:db8::/32"),       # documentation
    ipaddress.ip_network("2002::/16"),           # 6to4
    ipaddress.ip_network("fc00::/7"),            # unique-local
    ipaddress.ip_network("fe80::/10"),           # link-local
    ipaddress.ip_network("ff00::/8"),            # multicast
]

# Specific cloud metadata endpoints.
_BLOCKED_IPS: set[str] = {
    "169.254.169.254",    # AWS / GCP / Azure metadata
    "100.100.100.200",    # Alibaba Cloud metadata
    "192.0.0.1",          # IETF special-use
}

# Blocked hostname patterns.
_BLOCKED_HOSTNAMES: set[str] = {
    "localhost",
    "broadcasthost",
}

_BLOCKED_HOSTNAME_SUFFIXES: tuple[str, ...] = (
    ".local",
    ".internal",
    ".localhost",
    ".localdomain",
    ".lan",
    ".home",
    ".corp",
    ".test",
    ".example",
    ".invalid",
)

_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})


# ── Public API ───────────────────────────────────────────────────────────────

class SSRFViolation(Exception):
    """Raised when a URL or resolved IP is blocked by SSRF protection."""


def validate_ip(ip_str: str) -> None:
    """
    Raise *SSRFViolation* if *ip_str* is a private/reserved/special address.

    This is called after every DNS resolution.
    """
    if ip_str in _BLOCKED_IPS:
        raise SSRFViolation(
            f"IP address '{ip_str}' is a blocked cloud-metadata or "
            "special-use address."
        )
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        raise SSRFViolation(f"'{ip_str}' is not a valid IP address.")

    for net in _BLOCKED_NETWORKS:
        if addr in net:
            raise SSRFViolation(
                f"IP address '{ip_str}' belongs to blocked network "
                f"'{net}' (private/reserved/special-use)."
            )


def validate_hostname(hostname: str) -> None:
    """
    Raise *SSRFViolation* if the *hostname* itself is obviously private,
    before any DNS resolution is attempted.
    """
    if not hostname:
        raise SSRFViolation("Empty hostname is not allowed.")

    h = hostname.lower().strip()

    if h in _BLOCKED_HOSTNAMES:
        raise SSRFViolation(
            f"Hostname '{hostname}' is blocked (reserved hostname)."
        )

    for suffix in _BLOCKED_HOSTNAME_SUFFIXES:
        if h.endswith(suffix) or h == suffix.lstrip("."):
            raise SSRFViolation(
                f"Hostname '{hostname}' matches blocked suffix '{suffix}'."
            )

    # Single-label hostnames (no dot) — block unless they look like a TLD
    # stripped domain, which would be caught by domain validation anyway.
    # Exception: IPv6 literals are handled elsewhere.
    if "." not in h and ":" not in h and not h.startswith("["):
        raise SSRFViolation(
            f"Single-label hostname '{hostname}' is not allowed. "
            "Use a fully qualified domain name."
        )


def validate_scheme(scheme: str) -> None:
    """Raise *SSRFViolation* if *scheme* is not http or https."""
    if scheme.lower() not in _ALLOWED_SCHEMES:
        raise SSRFViolation(
            f"Scheme '{scheme}' is not permitted. "
            "Only http and https are allowed."
        )


def resolve_and_validate(hostname: str, timeout: float = 5.0) -> list[str]:
    """
    Resolve *hostname* via DNS and validate every resulting IP.

    Returns the list of resolved IP strings.
    Raises *SSRFViolation* if any IP is private/reserved.
    Raises *SSRFViolation* if DNS resolution fails (treat as unsafe).

    Anti-DNS-rebinding:
    The caller should re-resolve and re-validate before each actual
    TCP connection attempt for long-lived sessions.  For short-lived
    HTTP requests the single resolution here is sufficient.
    """
    validate_hostname(hostname)

    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC)
        resolved = list({info[4][0] for info in infos})
    except socket.gaierror as exc:
        raise SSRFViolation(
            f"DNS resolution failed for '{hostname}': {exc}. "
            "Cannot proceed — failing safe."
        )

    if not resolved:
        raise SSRFViolation(
            f"DNS returned no addresses for '{hostname}'."
        )

    for ip in resolved:
        validate_ip(ip)  # Raises SSRFViolation if private.

    return resolved


def validate_url_for_fetch(url: str) -> tuple[str, str, int]:
    """
    Full pre-fetch SSRF validation.

    Parses *url*, validates scheme, validates hostname, resolves DNS,
    and validates all resolved IPs.

    Returns (scheme, hostname, port) on success.
    Raises *SSRFViolation* on any violation.
    """
    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise SSRFViolation(f"Could not parse URL: {exc}")

    validate_scheme(parsed.scheme)

    hostname = (parsed.hostname or "").strip("[]")  # strip IPv6 brackets
    if not hostname:
        raise SSRFViolation("URL has no hostname.")

    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme.lower() == "https" else 80

    if not (1 <= port <= 65535):
        raise SSRFViolation(f"Port {port} is out of the valid range 1-65535.")

    # If hostname is already an IP, validate it directly.
    try:
        ipaddress.ip_address(hostname)
        validate_ip(hostname)
        return parsed.scheme.lower(), hostname, port
    except (ValueError, SSRFViolation) as exc:
        if isinstance(exc, SSRFViolation):
            raise

    # Hostname-based — resolve and validate.
    resolve_and_validate(hostname)
    return parsed.scheme.lower(), hostname, port
