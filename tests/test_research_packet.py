"""Tests for engine/research_packet.py."""
from __future__ import annotations

import math

from analytics.fill_decomposition import aggregate_fill_decomposition
from analytics.regime_analysis import summarize_regimes
from engine.event_ledger import build_event_ledger
from engine.research_packet import (
    build_packet,
    compute_confidence,
    compute_drawdown,
    compute_kill,
    compute_pnl_concentration,
    compute_promote,
)


class TestComputeConfidence:
    def test_high(self):
        rating, _ = compute_confidence(200, 20, 1.5)
        assert rating == "HIGH"

    def test_medium(self):
        rating, _ = compute_confidence(100, 10, 3.0)
        assert rating == "MEDIUM"

    def test_low_few_sessions(self):
        rating, _ = compute_confidence(20, 2, 1.0)
        assert rating == "LOW"

    def test_low_high_cv(self):
        rating, _ = compute_confidence(200, 20, 6.0)
        assert rating == "LOW"


class TestComputeDrawdown:
    def test_monotonic_increase(self):
        """No drawdown on a monotonically increasing series."""
        series = [float(i) for i in range(100)]
        dd = compute_drawdown(series)
        assert dd["max_drawdown"] == 0.0
        assert dd["recovered"] is True

    def test_simple_drawdown(self):
        series = [0.0, 10.0, 5.0, 3.0, 8.0, 12.0]
        dd = compute_drawdown(series)
        assert dd["max_drawdown"] == 3.0 - 10.0  # = -7.0
        assert dd["recovered"] is True
        assert dd["recovery_ticks"] == 5 - 3  # tick 5 recovers to 10+ at index 5 (val=12)

    def test_unrecovered_drawdown(self):
        series = [0.0, 10.0, 5.0, 3.0, 4.0]
        dd = compute_drawdown(series)
        assert dd["max_drawdown"] == -7.0
        assert dd["recovered"] is False
        assert math.isnan(dd["recovery_ticks"])

    def test_short_series(self):
        dd = compute_drawdown([5.0])
        assert dd["max_drawdown"] == 0.0


class TestComputeKill:
    def test_kill_triggered(self):
        dist = {
            "meanConfidenceHigh95": -10.0,
            "positiveRate": 0.20,
            "mean": -50.0,
            "std": 30.0,
        }
        result = compute_kill(dist, "HIGH")
        assert result["recommended"] is True

    def test_no_kill_positive(self):
        dist = {
            "meanConfidenceHigh95": 50.0,
            "positiveRate": 0.70,
            "mean": 100.0,
            "std": 50.0,
        }
        result = compute_kill(dist, "HIGH")
        assert result["recommended"] is False

    def test_no_kill_low_confidence(self):
        dist = {
            "meanConfidenceHigh95": -10.0,
            "positiveRate": 0.20,
            "mean": -50.0,
            "std": 30.0,
        }
        result = compute_kill(dist, "LOW")
        assert result["recommended"] is False


class TestComputePromote:
    def test_promote_triggered(self):
        dist = {
            "meanConfidenceLow95": 10.0,
            "positiveRate": 0.75,
            "sharpeLike": 0.5,
            "mean": 100.0,
        }
        dd = {"mean_max_drawdown": -100.0}  # abs < 3 * 100
        result = compute_promote(dist, "HIGH", dd)
        assert result["recommended"] is True

    def test_no_promote_low_confidence(self):
        dist = {
            "meanConfidenceLow95": 10.0,
            "positiveRate": 0.75,
            "sharpeLike": 0.5,
            "mean": 100.0,
        }
        dd = {"mean_max_drawdown": -100.0}
        result = compute_promote(dist, "MEDIUM", dd)
        assert result["recommended"] is False

    def test_no_promote_low_sharpe(self):
        dist = {
            "meanConfidenceLow95": 10.0,
            "positiveRate": 0.75,
            "sharpeLike": 0.1,
            "mean": 100.0,
        }
        dd = {"mean_max_drawdown": -100.0}
        result = compute_promote(dist, "HIGH", dd)
        assert result["recommended"] is False


class TestPnlConcentration:
    def test_uniform(self):
        values = [100.0] * 100
        c = compute_pnl_concentration(values)
        assert abs(c["top_10_pct_share"] - 0.10) < 0.02
        assert c["gini"] < 0.05

    def test_concentrated(self):
        values = [0.0] * 90 + [1000.0] * 10
        c = compute_pnl_concentration(values)
        assert c["top_10_pct_share"] == 1.0


class TestBuildPacket:
    def test_packet_structure(self, output_dir, dashboard_dict):
        ledger = build_event_ledger(output_dir)
        fill_decomp = aggregate_fill_decomposition(ledger["session_ledgers"])
        regimes = summarize_regimes(ledger["session_ledgers"])
        result = build_packet(dashboard_dict, ledger, fill_decomp, regimes)

        assert "short" in result
        assert "full" in result

        short = result["short"]
        assert "candidate_id" in short
        assert "confidence" in short
        assert short["confidence"] in ("HIGH", "MEDIUM", "LOW")
        assert "pnl" in short
        assert "mean" in short["pnl"]
        assert "kill" in short
        assert "promote" in short
        assert "diagnosis" in short
        assert isinstance(short["diagnosis"], str)
        assert len(short["diagnosis"]) > 10

    def test_full_extends_short(self, output_dir, dashboard_dict):
        ledger = build_event_ledger(output_dir)
        fill_decomp = aggregate_fill_decomposition(ledger["session_ledgers"])
        regimes = summarize_regimes(ledger["session_ledgers"])
        result = build_packet(dashboard_dict, ledger, fill_decomp, regimes)

        full = result["full"]
        assert "pnl_full" in full
        assert "fill_quality_full" in full
        assert "inventory_full" in full
        assert "regime_analysis" in full
        assert "sessions" in full

    def test_provenance_warnings(self, output_dir, dashboard_dict):
        ledger = build_event_ledger(output_dir)
        fill_decomp = aggregate_fill_decomposition(ledger["session_ledgers"])
        regimes = summarize_regimes(ledger["session_ledgers"])
        result = build_packet(dashboard_dict, ledger, fill_decomp, regimes)

        short = result["short"]
        assert isinstance(short["warnings"], list)
        # Should have sample-only warnings
        assert any("sample" in w.lower() for w in short["warnings"])
