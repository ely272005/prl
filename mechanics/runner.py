"""Probe runner — loads backtest data and executes probes."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Ensure the repo root and upstream backtester are importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.event_ledger import build_event_ledger
from mechanics.probe_spec import (
    Probe,
    ProbeResult,
    get_probe,
    list_probes,
    list_families,
)

# Force probe registration by importing the probes package
import mechanics.probes  # noqa: F401


PRODUCTS = ["EMERALDS", "TOMATOES"]


class ProbeRunner:
    """Loads backtest outputs and runs probes against them.

    Can load from one or more output directories (each containing a
    sessions/ subdirectory with sample session CSVs).
    """

    def __init__(self, output_dirs: list[Path]):
        self.output_dirs = [Path(d).resolve() for d in output_dirs]
        self._session_ledgers: dict[int, dict[str, pd.DataFrame]] | None = None
        self._labels: list[str] = []
        self._total_sessions = 0

    @property
    def session_ledgers(self) -> dict[int, dict[str, pd.DataFrame]]:
        if self._session_ledgers is None:
            self._load()
        return self._session_ledgers

    @property
    def dataset_label(self) -> str:
        if self._labels:
            return ", ".join(self._labels)
        return "unknown"

    def _load(self) -> None:
        """Load and merge session ledgers from all output directories."""
        merged: dict[int, dict[str, pd.DataFrame]] = {}
        offset = 0

        for output_dir in self.output_dirs:
            if not output_dir.exists():
                print(f"Warning: {output_dir} does not exist, skipping.", file=sys.stderr)
                continue

            label = output_dir.name
            self._labels.append(label)

            ledger = build_event_ledger(output_dir)
            for sid, session_data in ledger["session_ledgers"].items():
                # Use offset to avoid session ID collision across directories
                merged[offset + sid] = session_data
            offset += max(ledger["session_ids"], default=-1) + 1

        self._session_ledgers = merged
        self._total_sessions = len(merged)

    def run_all(
        self,
        products: list[str] | None = None,
        families: list[str] | None = None,
    ) -> list[ProbeResult]:
        """Run all registered probes (optionally filtered) on all products.

        Args:
            products: list of product names to probe. Default: all.
            families: list of probe families to run. Default: all.
        """
        if products is None:
            products = PRODUCTS
        if families is not None:
            specs = []
            for fam in families:
                specs.extend(list_probes(family=fam))
        else:
            specs = list_probes()

        results: list[ProbeResult] = []
        for spec in specs:
            probe = get_probe(spec.probe_id)
            for product in products:
                result = probe.run(self.session_ledgers, product, self.dataset_label)
                results.append(result)

        return results

    def run_probe(
        self,
        probe_id: str,
        product: str,
    ) -> ProbeResult:
        """Run a single probe on a single product."""
        probe = get_probe(probe_id)
        return probe.run(self.session_ledgers, product, self.dataset_label)

    def run_family(
        self,
        family: str,
        products: list[str] | None = None,
    ) -> list[ProbeResult]:
        """Run all probes in a family."""
        if products is None:
            products = PRODUCTS
        return self.run_all(products=products, families=[family])

    def summary(self) -> dict[str, Any]:
        """Return metadata about the loaded dataset."""
        # Access property to trigger lazy loading
        _ = self.session_ledgers
        return {
            "output_dirs": [str(d) for d in self.output_dirs],
            "dataset_label": self.dataset_label,
            "total_sessions": self._total_sessions,
            "available_families": list_families(),
            "available_probes": [s.probe_id for s in list_probes()],
        }
