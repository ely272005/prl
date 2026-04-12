"""Discovery Scanner — orchestrator that loads data and runs all analysis.

Consumes:
  - Backtest output directories (session-level tick data)
  - Bank directory with research packets (cross-candidate comparison)
  - Mechanics probe reports (optional, for probe-driven cards)

Produces:
  - Alpha cards ranked by strength
  - Discovery metadata for reporting
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.event_ledger import build_event_ledger
from discovery.regimes import build_regime_profile
from discovery.comparison import run_comparison, load_packets_from_bank
from discovery.weakness import run_all_scanners
from discovery.alpha_card import AlphaCard

PRODUCTS = ["EMERALDS", "TOMATOES"]


class DiscoveryScanner:
    """Main orchestrator: loads data, runs analysis, produces alpha cards."""

    def __init__(
        self,
        output_dirs: list[Path],
        bank_dir: Path | None = None,
        probe_report_path: Path | None = None,
        products: list[str] | None = None,
    ):
        self.output_dirs = [Path(d).resolve() for d in output_dirs]
        self.bank_dir = Path(bank_dir).resolve() if bank_dir else None
        self.probe_report_path = Path(probe_report_path).resolve() if probe_report_path else None
        self.products = products or PRODUCTS

        # Lazy-loaded state
        self._session_ledgers: dict[int, dict[str, pd.DataFrame]] | None = None
        self._regime_profile: dict[str, Any] | None = None
        self._comparison: dict[str, Any] | None = None
        self._probe_results: list[dict[str, Any]] | None = None
        self._cards: list[AlphaCard] | None = None

    # ----- Data loading -----

    @property
    def session_ledgers(self) -> dict[int, dict[str, pd.DataFrame]]:
        if self._session_ledgers is None:
            self._load_sessions()
        return self._session_ledgers

    def _load_sessions(self) -> None:
        merged: dict[int, dict[str, pd.DataFrame]] = {}
        offset = 0
        for output_dir in self.output_dirs:
            if not output_dir.exists():
                print(f"Warning: {output_dir} does not exist, skipping.", file=sys.stderr)
                continue
            ledger = build_event_ledger(output_dir)
            for sid, session_data in ledger["session_ledgers"].items():
                merged[offset + sid] = session_data
            offset += max(ledger["session_ids"], default=-1) + 1
        self._session_ledgers = merged

    def _load_probe_results(self) -> list[dict[str, Any]]:
        """Load probe results from a mechanics JSON report."""
        if self._probe_results is not None:
            return self._probe_results

        self._probe_results = []
        if self.probe_report_path and self.probe_report_path.exists():
            try:
                report = json.loads(self.probe_report_path.read_text())
                for family_results in report.get("results_by_family", {}).values():
                    self._probe_results.extend(family_results)
            except (json.JSONDecodeError, OSError):
                pass
        return self._probe_results

    # ----- Analysis steps -----

    def build_regime_profile(self) -> dict[str, Any]:
        if self._regime_profile is not None:
            return self._regime_profile
        self._regime_profile = build_regime_profile(
            self.session_ledgers,
            products=self.products,
        )
        return self._regime_profile

    def run_comparison(self) -> dict[str, Any] | None:
        if self._comparison is not None:
            return self._comparison
        if self.bank_dir and self.bank_dir.exists():
            self._comparison = run_comparison(self.bank_dir)
        return self._comparison

    def discover(self, max_cards: int = 20) -> list[AlphaCard]:
        """Run full discovery pipeline and return ranked alpha cards."""
        if self._cards is not None:
            return self._cards

        # Step 1: Build regime profile from session data
        profile = self.build_regime_profile()
        regime_stats = profile.get("regime_stats", {})

        # Step 2: Run cross-candidate comparison (if bank available)
        comparison = self.run_comparison()

        # Step 3: Load probe results (if available)
        probe_results = self._load_probe_results()

        # Step 4: Run all weakness scanners
        self._cards = run_all_scanners(
            regime_stats=regime_stats,
            comparison=comparison,
            probe_results=probe_results,
            max_cards=max_cards,
        )

        return self._cards

    # ----- Metadata -----

    def summary(self) -> dict[str, Any]:
        """Return metadata about inputs and analysis state."""
        _ = self.session_ledgers  # trigger lazy load
        profile = self.build_regime_profile()

        result = {
            "output_dirs": [str(d) for d in self.output_dirs],
            "bank_dir": str(self.bank_dir) if self.bank_dir else None,
            "probe_report_path": str(self.probe_report_path) if self.probe_report_path else None,
            "products": self.products,
            "session_count": profile.get("session_count", 0),
        }

        # Add comparison summary if available
        comparison = self.run_comparison()
        if comparison:
            result["packet_count"] = comparison.get("packet_count", 0)
            result["winner_count"] = comparison.get("winner_count", 0)
            result["loser_count"] = comparison.get("loser_count", 0)

        # Add probe count
        probes = self._load_probe_results()
        result["probe_result_count"] = len(probes)

        return result
