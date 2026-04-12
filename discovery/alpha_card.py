"""Alpha Card — structured artifact for a candidate exploitable weakness.

Each card clearly separates:
  - observed_fact: what the data shows (no interpretation)
  - interpretation: what this might mean (inference)
  - suggested_exploit: what kind of strategy could exploit this (speculative)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


CATEGORIES = (
    "regime_edge",          # A regime where fill quality is unusually strong or weak
    "role_mismatch",        # A regime where the wrong strategy role is dominant
    "bot_weakness",         # A pattern suggesting predictable bot behavior
    "danger_refinement",    # A narrower definition of a known danger zone
    "winner_trait",         # Something that differentiates winners from losers
    "inventory_exploit",    # An inventory-related opportunity
)

CONFIDENCE_LEVELS = ("high", "medium", "low")

# Card ID prefixes by category
_PREFIX = {
    "regime_edge": "RE",
    "role_mismatch": "RM",
    "bot_weakness": "BW",
    "danger_refinement": "DR",
    "winner_trait": "WT",
    "inventory_exploit": "IE",
}


@dataclass
class AlphaCard:
    """A candidate exploitable weakness backed by evidence."""

    card_id: str
    title: str
    category: str
    products: list[str]

    # Three-layer evidence structure
    observed_fact: str
    interpretation: str
    suggested_exploit: str

    # Regime and evidence
    regime_definition: dict[str, Any]
    evidence: dict[str, Any]
    baseline: dict[str, Any]
    sample_size: dict[str, int]

    # Quality
    confidence: str
    strength: float = 0.0                       # Numeric ranking score
    warnings: list[str] = field(default_factory=list)

    # Actionability
    candidate_strategy_style: str = ""
    recommended_experiment: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "title": self.title,
            "category": self.category,
            "products": self.products,
            "observed_fact": self.observed_fact,
            "interpretation": self.interpretation,
            "suggested_exploit": self.suggested_exploit,
            "regime_definition": _sanitize(self.regime_definition),
            "evidence": _sanitize(self.evidence),
            "baseline": _sanitize(self.baseline),
            "sample_size": self.sample_size,
            "confidence": self.confidence,
            "strength": round(self.strength, 4) if not math.isnan(self.strength) else None,
            "warnings": self.warnings,
            "candidate_strategy_style": self.candidate_strategy_style,
            "recommended_experiment": self.recommended_experiment,
        }


class CardCounter:
    """Assigns sequential IDs within each category prefix."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def next_id(self, category: str) -> str:
        prefix = _PREFIX.get(category, "XX")
        self._counts[category] = self._counts.get(category, 0) + 1
        return f"{prefix}{self._counts[category]:02d}"


def _sanitize(obj: Any) -> Any:
    """Make an object JSON-safe (NaN/inf → None)."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj
