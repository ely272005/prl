"""Tests for analytics/regime_analysis.py."""
from __future__ import annotations

from analytics.regime_analysis import label_session_regimes, summarize_regimes
from engine.event_ledger import build_event_ledger


class TestLabelSessionRegimes:
    def test_labels_present(self, output_dir):
        ledger = build_event_ledger(output_dir)
        for sid, session in ledger["session_ledgers"].items():
            labeled = label_session_regimes(session["traces"], session["prices"])
            assert "volatility_regime" in labeled.columns
            assert "spread_regime" in labeled.columns
            assert "inventory_regime" in labeled.columns
            assert "session_phase" in labeled.columns

    def test_emerald_volatility_always_low(self, output_dir):
        """EMERALDS fair=10000 constant -> volatility should be 'low'."""
        ledger = build_event_ledger(output_dir)
        for sid, session in ledger["session_ledgers"].items():
            labeled = label_session_regimes(session["traces"], session["prices"])
            emerald = labeled[labeled["product"] == "EMERALDS"]
            assert (emerald["volatility_regime"] == "low").all()

    def test_session_phase_order(self, output_dir):
        """Phase should progress: early -> mid -> late."""
        ledger = build_event_ledger(output_dir)
        for sid, session in ledger["session_ledgers"].items():
            labeled = label_session_regimes(session["traces"], session["prices"])
            for product, group in labeled.groupby("product"):
                phases = group["session_phase"].tolist()
                if len(phases) < 3:
                    continue
                assert phases[0] == "early"
                assert phases[-1] == "late"
                # mid should appear in between
                assert "mid" in phases

    def test_regime_labels_valid(self, output_dir):
        ledger = build_event_ledger(output_dir)
        for sid, session in ledger["session_ledgers"].items():
            labeled = label_session_regimes(session["traces"], session["prices"])
            valid_labels = {"low", "medium", "high", "unknown"}
            assert set(labeled["volatility_regime"].unique()) <= valid_labels
            assert set(labeled["spread_regime"].unique()) <= valid_labels
            assert set(labeled["inventory_regime"].unique()) <= valid_labels
            assert set(labeled["session_phase"].unique()) <= {"early", "mid", "late"}


class TestSummarizeRegimes:
    def test_summarize(self, output_dir):
        ledger = build_event_ledger(output_dir)
        summary = summarize_regimes(ledger["session_ledgers"])
        assert "by_product" in summary
        assert "provenance" in summary
        assert summary["provenance"]["scope"] == "sample"

    def test_products_present(self, output_dir):
        ledger = build_event_ledger(output_dir)
        summary = summarize_regimes(ledger["session_ledgers"])
        by_product = summary["by_product"]
        assert "EMERALDS" in by_product
        assert "TOMATOES" in by_product
        assert "volatility" in by_product["EMERALDS"]
        assert "spread" in by_product["EMERALDS"]
        assert "inventory" in by_product["EMERALDS"]
        assert "phase" in by_product["EMERALDS"]
