"""
SSL/TLS analysis feature service — V3.

Checks:
- Whether HTTPS is used
- TLS handshake success
- Certificate presence
- Certificate validity (date range)
- Certificate expiration
- Hostname / certificate CN match (via ssl.match_hostname fallback)

IMPORTANT:
  A valid certificate does NOT prove legitimacy.
  Phishing sites routinely obtain free HTTPS certificates.
  SSL findings are one evidence signal among many.

Does NOT fabricate SSL results.
If a check is not performed, returns status "not_checked".
"""

import socket
import ssl
import datetime
from typing import Optional


def analyse_ssl(hostname: str, port: int = 443, timeout: float = 5.0) -> dict:
    """
    Attempt a TLS connection to *hostname*:*port* and return structured results.

    Return dict keys
    ----------------
    status          : "checked" | "not_https" | "error" | "timeout" | "hostname_mismatch"
    https_used      : bool
    tls_connected   : bool
    cert_present    : bool
    cert_valid      : bool    — within validity date range
    cert_expired    : bool
    cert_not_yet_valid: bool
    hostname_match  : bool | None    — None = not checked
    subject_cn      : str | None
    issuer_o        : str | None
    valid_from      : str | None     — ISO-8601
    valid_until     : str | None     — ISO-8601
    days_until_expiry : int | None
    error           : str | None
    signals         : list[dict]
    """
    signals: list[dict] = []
    base: dict = {
        "status": "not_checked",
        "https_used": False,
        "tls_connected": False,
        "cert_present": False,
        "cert_valid": False,
        "cert_expired": False,
        "cert_not_yet_valid": False,
        "hostname_match": None,
        "subject_cn": None,
        "issuer_o": None,
        "valid_from": None,
        "valid_until": None,
        "days_until_expiry": None,
        "error": None,
        "signals": signals,
    }

    if port != 443:
        # Only check TLS for the standard HTTPS port to keep things simple.
        base["status"] = "not_checked"
        base["https_used"] = (port == 443)
        return base

    base["https_used"] = True

    ctx = ssl.create_default_context()

    try:
        with socket.create_connection(
            (hostname, port), timeout=timeout
        ) as raw_sock:
            with ctx.wrap_socket(raw_sock, server_hostname=hostname) as tls_sock:
                base["tls_connected"] = True
                cert = tls_sock.getpeercert()

                if not cert:
                    base["cert_present"] = False
                    signals.append({
                        "id": "ssl_no_cert",
                        "severity": "high",
                        "title": "No certificate returned by TLS handshake",
                        "detail": "The server did not present an X.509 certificate.",
                    })
                    base["status"] = "checked"
                    return base

                base["cert_present"] = True

                # Subject CN.
                subject = dict(
                    x[0] for x in cert.get("subject", [])
                )
                issuer = dict(
                    x[0] for x in cert.get("issuer", [])
                )
                cn = subject.get("commonName")
                issuer_o = issuer.get("organizationName")
                base["subject_cn"] = cn
                base["issuer_o"] = issuer_o

                # Validity dates.
                not_before_str = cert.get("notBefore")
                not_after_str = cert.get("notAfter")

                fmt = "%b %d %H:%M:%S %Y %Z"
                now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)

                valid_from: Optional[datetime.datetime] = None
                valid_until: Optional[datetime.datetime] = None

                if not_before_str:
                    try:
                        valid_from = datetime.datetime.strptime(
                            not_before_str, fmt
                        )
                        base["valid_from"] = valid_from.isoformat()
                    except ValueError:
                        pass

                if not_after_str:
                    try:
                        valid_until = datetime.datetime.strptime(
                            not_after_str, fmt
                        )
                        base["valid_until"] = valid_until.isoformat()
                    except ValueError:
                        pass

                # Date checks.
                if valid_from and now < valid_from:
                    base["cert_not_yet_valid"] = True
                    signals.append({
                        "id": "ssl_cert_not_yet_valid",
                        "severity": "high",
                        "title": "Certificate is not yet valid",
                        "detail": (
                            f"Certificate valid_from is {valid_from.isoformat()} "
                            f"but current time is {now.isoformat()}."
                        ),
                    })

                if valid_until:
                    days_left = (valid_until - now).days
                    base["days_until_expiry"] = days_left
                    if days_left < 0:
                        base["cert_expired"] = True
                        signals.append({
                            "id": "ssl_cert_expired",
                            "severity": "high",
                            "title": "Certificate has expired",
                            "detail": (
                                f"Certificate expired on {valid_until.isoformat()}."
                            ),
                        })
                    elif days_left <= 7:
                        signals.append({
                            "id": "ssl_cert_expiring_soon",
                            "severity": "medium",
                            "title": "Certificate expires very soon",
                            "detail": (
                                f"Certificate expires in {days_left} day(s)."
                            ),
                        })

                if not base["cert_expired"] and not base["cert_not_yet_valid"]:
                    base["cert_valid"] = True

                # Hostname match is verified by ssl.create_default_context()
                # raising ssl.CertificateError if it fails. Reaching here
                # means the hostname matched.
                base["hostname_match"] = True
                base["status"] = "checked"

    except ssl.CertificateError as exc:
        base["tls_connected"] = True
        base["hostname_match"] = False
        base["status"] = "hostname_mismatch"
        base["error"] = str(exc)
        signals.append({
            "id": "ssl_hostname_mismatch",
            "severity": "high",
            "title": "SSL certificate hostname does not match",
            "detail": str(exc),
        })

    except ssl.SSLError as exc:
        base["status"] = "error"
        base["error"] = str(exc)
        signals.append({
            "id": "ssl_error",
            "severity": "medium",
            "title": "TLS/SSL error",
            "detail": str(exc),
        })

    except socket.timeout:
        base["status"] = "timeout"
        base["error"] = f"Connection to {hostname}:{port} timed out."
        signals.append({
            "id": "ssl_timeout",
            "severity": "low",
            "title": "SSL check timed out",
            "detail": base["error"],
        })

    except ConnectionRefusedError:
        base["status"] = "error"
        base["error"] = f"Connection refused to {hostname}:{port}."

    except OSError as exc:
        base["status"] = "error"
        base["error"] = str(exc)

    # Reminder signal: valid SSL ≠ legitimate.
    if base.get("cert_valid") and base.get("hostname_match"):
        signals.append({
            "id": "ssl_valid_note",
            "severity": "info",
            "title": "Valid SSL certificate present",
            "detail": (
                "The site has a valid, matching TLS certificate. "
                "Note: phishing sites routinely obtain free HTTPS certificates. "
                "SSL validity alone does not indicate legitimacy."
            ),
        })

    return base
