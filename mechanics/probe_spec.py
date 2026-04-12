"""Probe specification and result types.

A Probe is a repeatable experiment that tests a specific hypothesis about
market or simulator behavior using backtest session data.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Spec: describes what a probe is
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProbeSpec:
    """Machine-readable description of a probe."""
    probe_id: str
    family: str          # passive_fill | taking | inventory | danger_zone
    title: str
    hypothesis: str
    required_data: tuple[str, ...]  # keys into session_ledger: traces, prices, strategy_fills, all_trades
    metrics_produced: tuple[str, ...]  # names of key metrics this probe emits


# ---------------------------------------------------------------------------
# Result: what a probe produces
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    """Structured output of a single probe execution."""
    probe_id: str
    family: str
    title: str
    hypothesis: str
    product: str
    dataset: str                  # description of source data
    sample_size: dict[str, int]   # sessions, fills, ticks as relevant
    metrics: dict[str, Any]       # probe-specific key metrics
    verdict: str                  # supported | refuted | inconclusive | insufficient_data
    confidence: str               # high | medium | low
    detail: str                   # human-readable explanation of findings
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Base class: all probes inherit from this
# ---------------------------------------------------------------------------

class Probe(ABC):
    """Base class for all mechanics probes."""

    spec: ProbeSpec  # subclasses must set this as a class attribute

    @abstractmethod
    def run(
        self,
        session_ledgers: dict[int, dict[str, pd.DataFrame]],
        product: str,
        dataset_label: str = "",
    ) -> ProbeResult:
        """Execute the probe on session data for one product.

        Args:
            session_ledgers: dict of session_id -> {traces, prices, all_trades, strategy_fills}
            product: "EMERALDS" or "TOMATOES"
            dataset_label: human-readable label for the data source
        """
        ...

    def _insufficient(self, product: str, dataset_label: str, reason: str) -> ProbeResult:
        """Helper to return an insufficient_data result."""
        return ProbeResult(
            probe_id=self.spec.probe_id,
            family=self.spec.family,
            title=self.spec.title,
            hypothesis=self.spec.hypothesis,
            product=product,
            dataset=dataset_label,
            sample_size={},
            metrics={},
            verdict="insufficient_data",
            confidence="low",
            detail=reason,
            warnings=[reason],
        )


# ---------------------------------------------------------------------------
# Registry: discover and look up probes
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[Probe]] = {}


def register_probe(cls: type[Probe]) -> type[Probe]:
    """Class decorator — registers a Probe subclass by its spec.probe_id."""
    if not hasattr(cls, "spec") or not isinstance(cls.spec, ProbeSpec):
        raise TypeError(f"{cls.__name__} must define a ProbeSpec as class attribute 'spec'")
    _REGISTRY[cls.spec.probe_id] = cls
    return cls


def get_probe(probe_id: str) -> Probe:
    """Instantiate a registered probe by ID."""
    cls = _REGISTRY.get(probe_id)
    if cls is None:
        raise KeyError(f"Unknown probe: {probe_id!r}. Known: {sorted(_REGISTRY)}")
    return cls()


def list_probes(family: str | None = None) -> list[ProbeSpec]:
    """List all registered probe specs, optionally filtered by family."""
    specs = [cls.spec for cls in _REGISTRY.values()]
    if family:
        specs = [s for s in specs if s.family == family]
    return sorted(specs, key=lambda s: s.probe_id)


def list_families() -> list[str]:
    """List all registered probe families."""
    return sorted({cls.spec.family for cls in _REGISTRY.values()})
