"""
Evidence engine — V3.

Every detection in the system produces a structured evidence dict.

Evidence schema
---------------
{
    "id"      : str   — unique snake_case identifier
    "severity": str   — "critical" | "high" | "medium" | "low" | "info"
    "title"   : str   — short human-readable title
    "details" : str   — longer explanation
    "source"  : str   — which analyser produced this
}

Rules
-----
- Evidence must be based on actual observations.
- Never invent evidence.
- Severity levels guide the risk engine — do not inflate severity.
- Evidence IDs must be stable (the risk engine keys on them).

This module provides:
- EvidenceItem dataclass
- collect() — aggregate evidence from multiple analysis results
- filter_by_severity() — filter for dashboard display
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal


Severity = Literal["critical", "high", "medium", "low", "info"]

# Numeric weights for severity levels (used by risk engine).
SEVERITY_WEIGHTS: dict[str, float] = {
    "critical": 25.0,
    "high": 15.0,
    "medium": 8.0,
    "low": 3.0,
    "info": 0.0,
}


@dataclass
class EvidenceItem:
    id: str
    severity: Severity
    title: str
    details: str
    source: str

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def weight(self) -> float:
        return SEVERITY_WEIGHTS.get(self.severity, 0.0)


def _make_item(signal: dict, source: str) -> EvidenceItem:
    return EvidenceItem(
        id=signal.get("id", "unknown"),
        severity=signal.get("severity", "info"),  # type: ignore[arg-type]
        title=signal.get("title", ""),
        details=signal.get("detail", signal.get("details", "")),
        source=source,
    )


def collect(
    *,
    url_manipulation: list[dict] | None = None,
    domain_signals: list[dict] | None = None,
    dns_signals: list[dict] | None = None,
    ssl_signals: list[dict] | None = None,
    redirect_signals: list[dict] | None = None,
    html_signals: list[dict] | None = None,
    spoofing_signals: list[dict] | None = None,
    login_page_signals: list[dict] | None = None,
    reachability_signals: list[dict] | None = None,
) -> list[EvidenceItem]:
    """
    Collect and deduplicate evidence from all analysis sources.

    Parameters correspond to the `.signals` list from each analyser.
    Pass None for any analyser that was not run.
    """
    items: list[EvidenceItem] = []
    seen_ids: set[str] = set()

    source_map = [
        (url_manipulation or [], "url_analyzer"),
        (domain_signals or [], "domain_analyzer"),
        (dns_signals or [], "dns_analyzer"),
        (ssl_signals or [], "ssl_analyzer"),
        (redirect_signals or [], "redirect_analyzer"),
        (html_signals or [], "html_analyzer"),
        (spoofing_signals or [], "spoofing_analyzer"),
        (login_page_signals or [], "login_page_analyzer"),
        (reachability_signals or [], "reachability_analyzer"),
    ]

    for signals, source in source_map:
        for signal in signals:
            item = _make_item(signal, source)
            # Allow duplicate IDs from different sources (e.g. two analysers
            # can both raise "possible_brand_impersonation").
            key = f"{source}:{item.id}"
            if key not in seen_ids:
                seen_ids.add(key)
                items.append(item)

    return items


def filter_by_severity(
    items: list[EvidenceItem],
    min_severity: str = "info",
) -> list[EvidenceItem]:
    """Return only items at or above *min_severity*."""
    order = list(SEVERITY_WEIGHTS.keys())
    min_idx = order.index(min_severity) if min_severity in order else len(order)
    return [
        item for item in items
        if order.index(item.severity) <= min_idx
    ]


def total_evidence_weight(items: list[EvidenceItem]) -> float:
    """Sum of severity weights — used by the risk engine."""
    return sum(item.weight for item in items)
