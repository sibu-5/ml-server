"""
DNS resolution and analysis — V3.

Provides:
- Safe DNS resolution with SSRF IP validation
- Structured DNS result dict
- Evidence signals for missing / strange DNS behaviour

Does NOT fabricate DNS results.
If resolution fails, returns explicit status (DNS_FAILED, NXDOMAIN, etc.).
"""

import ipaddress
import socket
from typing import Optional


# ── Private / reserved IP ranges (SSRF protection mirror) ──────────────────
# Keep in sync with app/analyzers/reachability.py

_PRIVATE_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),       # loopback
    ipaddress.ip_network("10.0.0.0/8"),         # RFC1918
    ipaddress.ip_network("172.16.0.0/12"),      # RFC1918
    ipaddress.ip_network("192.168.0.0/16"),     # RFC1918
    ipaddress.ip_network("169.254.0.0/16"),     # link-local
    ipaddress.ip_network("100.64.0.0/10"),      # Carrier-grade NAT / Tailscale
    ipaddress.ip_network("192.0.0.0/24"),       # IETF protocol
    ipaddress.ip_network("198.18.0.0/15"),      # benchmark
    ipaddress.ip_network("198.51.100.0/24"),    # documentation
    ipaddress.ip_network("203.0.113.0/24"),     # documentation
    ipaddress.ip_network("240.0.0.0/4"),        # reserved
    ipaddress.ip_network("255.255.255.255/32"), # broadcast
    ipaddress.ip_network("::1/128"),            # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),           # IPv6 unique-local
    ipaddress.ip_network("fe80::/10"),          # IPv6 link-local
    ipaddress.ip_network("ff00::/8"),           # IPv6 multicast
    ipaddress.ip_network("::/128"),             # unspecified
]


def _ip_is_private(ip_str: str) -> bool:
    """Return True if *ip_str* falls within any private/reserved range."""
    try:
        addr = ipaddress.ip_address(ip_str)
        return any(addr in net for net in _PRIVATE_NETWORKS)
    except ValueError:
        return True  # Treat unparseable addresses as unsafe.


def resolve_hostname(hostname: str) -> dict:
    """
    Resolve *hostname* and return a structured result.

    Return dict keys
    ----------------
    status      : "resolved" | "nxdomain" | "timeout" | "error"
    hostname    : str
    resolved_ips: list[str]
    has_private_ip : bool  — True if any resolved IP is private/internal
    private_ips    : list[str]
    signals        : list[dict]
    """
    signals: list[dict] = []

    if not hostname:
        return {
            "status": "error",
            "hostname": hostname,
            "resolved_ips": [],
            "has_private_ip": False,
            "private_ips": [],
            "signals": [{
                "id": "empty_hostname",
                "severity": "high",
                "title": "Empty hostname",
                "detail": "No hostname was provided for DNS resolution.",
            }],
        }

    try:
        infos = socket.getaddrinfo(hostname, None)
        resolved_ips = list({info[4][0] for info in infos})
    except socket.gaierror as exc:
        code = exc.args[0] if exc.args else 0
        status = "nxdomain" if code in (
            socket.EAI_NONAME,       # -2 on Linux
            11001,                   # WSAHOST_NOT_FOUND on Windows
            11003,                   # WSANO_RECOVERY
        ) else "error"
        return {
            "status": status,
            "hostname": hostname,
            "resolved_ips": [],
            "has_private_ip": False,
            "private_ips": [],
            "signals": [{
                "id": "dns_resolution_failed",
                "severity": "info",
                "title": f"DNS resolution failed ({status.upper()})",
                "detail": str(exc),
            }],
        }
    except socket.timeout:
        return {
            "status": "timeout",
            "hostname": hostname,
            "resolved_ips": [],
            "has_private_ip": False,
            "private_ips": [],
            "signals": [{
                "id": "dns_timeout",
                "severity": "info",
                "title": "DNS resolution timed out",
                "detail": f"Hostname '{hostname}' did not respond within the timeout.",
            }],
        }
    except Exception as exc:
        return {
            "status": "error",
            "hostname": hostname,
            "resolved_ips": [],
            "has_private_ip": False,
            "private_ips": [],
            "signals": [{
                "id": "dns_error",
                "severity": "info",
                "title": "Unexpected DNS error",
                "detail": str(exc),
            }],
        }

    private_ips = [ip for ip in resolved_ips if _ip_is_private(ip)]
    has_private = bool(private_ips)

    if has_private:
        signals.append({
            "id": "dns_resolves_to_private_ip",
            "severity": "high",
            "title": "Domain resolves to a private/internal IP address",
            "detail": (
                f"Hostname '{hostname}' resolved to private addresses: "
                f"{', '.join(private_ips)}. "
                "This is a possible DNS rebinding or SSRF attempt."
            ),
        })

    return {
        "status": "resolved",
        "hostname": hostname,
        "resolved_ips": resolved_ips,
        "has_private_ip": has_private,
        "private_ips": private_ips,
        "signals": signals,
    }
