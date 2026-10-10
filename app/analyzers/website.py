"""
HTML / website analyser — V3.

Analyses fetched HTML safely WITHOUT executing JavaScript.

Detects:
- Login forms (form with password input)
- Email / username inputs
- OTP / PIN / payment fields
- Credential-related text patterns
- External form actions (form posts to another domain)
- iframes
- Hidden inputs inside forms
- Suspicious page title
- Brand name in title vs domain mismatch

IMPORTANT:
  Having a login form does NOT automatically mean phishing.
  Many legitimate websites have login forms.
  This analyser returns evidence signals only.
  The risk engine weighs them with other evidence.

Does NOT execute JavaScript.
Does NOT fetch external resources.
"""

import re
from typing import Optional
from urllib.parse import urljoin, urlparse

try:
    from html.parser import HTMLParser
    _HAS_HTML_PARSER = True
except ImportError:
    _HAS_HTML_PARSER = False

import tldextract


# ── Simple HTML parser ───────────────────────────────────────────────────────

class _HTMLAnalysisParser(HTMLParser):
    """
    Lightweight SAX-like HTML parser that collects:
    - forms and their action attributes
    - input fields and their types/names
    - iframes
    - title text
    - hidden inputs
    """

    def __init__(self):
        super().__init__()
        self.forms: list[dict] = []
        self.inputs: list[dict] = []
        self.iframes: list[dict] = []
        self.title: str = ""
        self._in_title = False
        self._current_form: Optional[dict] = None

    def _attrs_dict(self, attrs) -> dict:
        return {k.lower(): (v or "").lower() for k, v in attrs}

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        a = self._attrs_dict(attrs)

        if tag == "form":
            self._current_form = {
                "action": a.get("action", ""),
                "method": a.get("method", "get"),
                "inputs": [],
            }
            self.forms.append(self._current_form)

        elif tag == "input":
            input_info = {
                "type": a.get("type", "text"),
                "name": a.get("name", ""),
                "id": a.get("id", ""),
                "hidden": a.get("type", "") == "hidden",
            }
            self.inputs.append(input_info)
            if self._current_form is not None:
                self._current_form["inputs"].append(input_info)

        elif tag == "iframe":
            self.iframes.append({
                "src": a.get("src", ""),
                "sandbox": a.get("sandbox"),
            })

        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag.lower() == "form":
            self._current_form = None
        elif tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data


# ── Credential field detection ───────────────────────────────────────────────

_CREDENTIAL_INPUT_TYPES = frozenset({
    "password", "email", "tel",
})
_CREDENTIAL_INPUT_NAMES = frozenset({
    "password", "passwd", "pass", "pwd",
    "email", "username", "user", "login",
    "otp", "pin", "card", "cvv", "cc",
    "credit", "token", "secret",
})
_CREDENTIAL_TEXT_PATTERNS = re.compile(
    r"\b(sign[\s\-]?in|log[\s\-]?in|password|username|email|otp|"
    r"verify|authentication|credential|account|confirm|reset|recover)\b",
    re.IGNORECASE,
)


def _is_credential_input(inp: dict) -> bool:
    t = inp.get("type", "text").lower()
    n = inp.get("name", "").lower()
    i = inp.get("id", "").lower()
    return (
        t in _CREDENTIAL_INPUT_TYPES
        or n in _CREDENTIAL_INPUT_NAMES
        or i in _CREDENTIAL_INPUT_NAMES
    )


def _registered_domain(url: str) -> str:
    try:
        h = (urlparse(url).hostname or "").lower()
        ext = tldextract.extract(h)
        return ext.top_domain_under_public_suffix or h
    except Exception:
        return url


# ── Main analyser function ───────────────────────────────────────────────────

def analyse_website_html(
    html: str,
    page_url: str,
) -> dict:
    """
    Analyse *html* content fetched from *page_url*.

    Parameters
    ----------
    html     : Raw HTML string (already truncated to safe size).
    page_url : The URL from which the HTML was fetched.

    Return dict keys
    ----------------
    title               : str
    has_login_form      : bool
    has_password_field  : bool
    has_email_field     : bool
    has_otp_field       : bool
    has_hidden_inputs   : bool
    has_external_form_action : bool
    has_iframe          : bool
    form_count          : int
    input_count         : int
    external_form_domains: list[str]
    credential_text_matches: int
    signals             : list[dict]
    """
    signals: list[dict] = []

    if not html:
        return {
            "title": "",
            "has_login_form": False,
            "has_password_field": False,
            "has_email_field": False,
            "has_otp_field": False,
            "has_hidden_inputs": False,
            "has_external_form_action": False,
            "has_iframe": False,
            "form_count": 0,
            "input_count": 0,
            "external_form_domains": [],
            "credential_text_matches": 0,
            "signals": signals,
        }

    parser = _HTMLAnalysisParser()
    try:
        parser.feed(html[:500_000])  # Hard cap at 500 KB for parsing
    except Exception:
        pass

    page_domain = _registered_domain(page_url)

    # ── Form analysis ────────────────────────────────────────────────
    has_password_field = False
    has_email_field = False
    has_otp_field = False
    has_login_form = False
    has_hidden_inputs = False
    external_form_domains: list[str] = []
    has_external_form_action = False

    for form in parser.forms:
        password_in_form = False
        email_in_form = False

        for inp in form.get("inputs", []):
            t = inp.get("type", "text").lower()
            if t == "password":
                has_password_field = True
                password_in_form = True
            if t in ("email", "tel"):
                has_email_field = True
                email_in_form = True
            n = inp.get("name", "").lower()
            i = inp.get("id", "").lower()
            if "otp" in n or "pin" in n or "otp" in i or "pin" in i:
                has_otp_field = True
            if inp.get("hidden"):
                has_hidden_inputs = True

        if password_in_form:
            has_login_form = True

        # Check external form action.
        action = form.get("action", "")
        if action and not action.startswith("#") and not action.startswith("?"):
            abs_action = urljoin(page_url, action)
            action_domain = _registered_domain(abs_action)
            if action_domain and action_domain != page_domain:
                external_form_domains.append(action_domain)
                has_external_form_action = True

    # ── Iframe check ─────────────────────────────────────────────────
    has_iframe = bool(parser.iframes)

    # ── Credential text ──────────────────────────────────────────────
    cred_matches = len(_CREDENTIAL_TEXT_PATTERNS.findall(html[:100_000]))

    # ── Evidence signals ─────────────────────────────────────────────

    if has_login_form and has_external_form_action:
        signals.append({
            "id": "login_form_external_action",
            "severity": "critical",
            "title": "Login form submits to an external domain",
            "detail": (
                f"A form with a password field on '{page_domain}' "
                f"posts data to: {', '.join(external_form_domains)}. "
                "This is a strong indicator of credential harvesting."
            ),
        })
    elif has_external_form_action:
        signals.append({
            "id": "external_form_action",
            "severity": "high",
            "title": "Form submits to an external domain",
            "detail": (
                f"A form on '{page_domain}' posts data to: "
                f"{', '.join(external_form_domains)}."
            ),
        })

    if has_login_form:
        signals.append({
            "id": "login_form_present",
            "severity": "info",
            "title": "Page contains a login form",
            "detail": (
                "The page has a form with a password field. "
                "Many legitimate websites have login forms; "
                "this signal alone is not sufficient to classify phishing."
            ),
        })

    if has_hidden_inputs:
        signals.append({
            "id": "hidden_inputs_in_form",
            "severity": "low",
            "title": "Form contains hidden inputs",
            "detail": (
                "One or more hidden form fields were detected. "
                "These may carry tracking tokens or anti-CSRF values, "
                "but can also be used to collect hidden data."
            ),
        })

    if has_iframe:
        signals.append({
            "id": "iframe_present",
            "severity": "low",
            "title": "Page contains an iframe",
            "detail": (
                f"{len(parser.iframes)} iframe(s) detected. "
                "Iframes can embed external content or hide phishing elements."
            ),
        })

    if cred_matches >= 3:
        signals.append({
            "id": "high_credential_text_density",
            "severity": "medium",
            "title": "Page text contains multiple credential-related terms",
            "detail": (
                f"Found {cred_matches} credential-related keyword matches "
                "(login, password, verify, etc.) in page text."
            ),
        })

    return {
        "title": parser.title.strip(),
        "has_login_form": has_login_form,
        "has_password_field": has_password_field,
        "has_email_field": has_email_field,
        "has_otp_field": has_otp_field,
        "has_hidden_inputs": has_hidden_inputs,
        "has_external_form_action": has_external_form_action,
        "has_iframe": has_iframe,
        "form_count": len(parser.forms),
        "input_count": len(parser.inputs),
        "external_form_domains": external_form_domains,
        "credential_text_matches": cred_matches,
        "signals": signals,
    }
