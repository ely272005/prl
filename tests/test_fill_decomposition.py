"""Tests for analytics/fill_decomposition.py."""
from __future__ import annotations

import pandas as pd

from analytics.fill_decomposition import (
    adverse_selection_rate,
    aggregate_fill_decomposition,
    fill_vs_fair_stats,
    fill_vs_mid_stats,
    maker_taker_volumes,
)
from engine.event_ledger import build_event_ledger


class TestMakerTakerVolumes:
    def test_basic_counts(self, output_dir):
        ledger = build_event_ledger(output_dir)
        for sid, session in ledger["session_ledgers"].items():
            fills = session["strategy_fills"]
            volumes = maker_taker_volumes(fills)
            assert volumes["taker_fill_count"] > 0
            assert volumes["maker_fill_count"] > 0
            assert 0 < volumes["passive_fill_rate"] < 1

    def test_empty_fills(self):
        empty = pd.DataFrame(columns=["strategy_role", "quantity"])
        volumes = maker_taker_volumes(empty)
        assert volumes["taker_fill_count"] == 0
        assert volumes["passive_fill_rate"] == 0.0


class TestFillVsFairStats:
    def test_per_product(self, output_dir):
        ledger = build_event_ledger(output_dir)
        for sid, session in ledger["session_ledgers"].items():
            stats = fill_vs_fair_stats(session["strategy_fills"])
            assert "EMERALDS" in stats
            assert "mean" in stats["EMERALDS"]
            assert "count" in stats["EMERALDS"]
            assert stats["EMERALDS"]["count"] > 0

    def test_empty(self):
        empty = pd.DataFrame()
        assert fill_vs_fair_stats(empty) == {}


class TestFillVsMidStats:
    def test_per_product(self, output_dir):
        ledger = build_event_ledger(output_dir)
        for sid, session in ledger["session_ledgers"].items():
            stats = fill_vs_mid_stats(session["strategy_fills"])
            assert "EMERALDS" in stats


class TestAdverseSelectionRate:
    def test_emerald_rate_near_zero(self, output_dir):
        """EMERALDS fair is constant -> delta_fair = 0 -> no adverse selection."""
        ledger = build_event_ledger(output_dir)
        for sid, session in ledger["session_ledgers"].items():
            adv = adverse_selection_rate(
                session["strategy_fills"], session["traces"], forward_ticks=10
            )
            if "EMERALDS" in adv and adv["EMERALDS"]["count"] > 0:
                assert adv["EMERALDS"]["rate"] == 0.0

    def test_tomato_rate_is_valid(self, output_dir):
        """TOMATOES has a random walk -> some adverse selection expected."""
        ledger = build_event_ledger(output_dir)
        for sid, session in ledger["session_ledgers"].items():
            adv = adverse_selection_rate(
                session["strategy_fills"], session["traces"], forward_ticks=5
            )
            if "TOMATOES" in adv:
                assert 0.0 <= adv["TOMATOES"]["rate"] <= 1.0


class TestAggregateDecomposition:
    def test_aggregate(self, output_dir):
        ledger = build_event_ledger(output_dir)
        result = aggregate_fill_decomposition(ledger["session_ledgers"])
        assert result["volumes"]["taker_fill_count"] > 0
        assert result["volumes"]["maker_fill_count"] > 0
        assert 0 < result["volumes"]["passive_fill_rate"] < 1
        assert result["provenance"]["scope"] == "sample"
        assert "EMERALDS" in result["fill_vs_fair"]
