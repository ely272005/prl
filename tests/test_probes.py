"""Tests for individual probe computations on synthetic fixture data.

Uses the existing conftest.py fixtures (output_dir with 2 synthetic sessions).
"""
import math
import pytest
from pathlib import Path

from engine.event_ledger import build_event_ledger
from mechanics.probe_spec import get_probe, ProbeResult
import mechanics.probes  # noqa: F401


@pytest.fixture
def session_ledgers(output_dir):
    """Build session ledgers from the synthetic output_dir fixture."""
    ledger = build_event_ledger(output_dir)
    return ledger["session_ledgers"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_valid_result(r: ProbeResult):
    """Basic structural checks on a probe result."""
    assert r.probe_id
    assert r.family
    assert r.product in ("EMERALDS", "TOMATOES")
    assert r.verdict in ("supported", "refuted", "inconclusive", "insufficient_data")
    assert r.confidence in ("high", "medium", "low")
    assert isinstance(r.metrics, dict)
    assert isinstance(r.warnings, list)


# ---------------------------------------------------------------------------
# Passive fill probes
# ---------------------------------------------------------------------------

class TestSpreadVsMakerEdge:
    def test_runs_on_emeralds(self, session_ledgers):
        probe = get_probe("pf01_spread_vs_maker_edge")
        r = probe.run(session_ledgers, "EMERALDS", "test")
        _assert_valid_result(r)
        # Synthetic fixture has only ~10 EMERALDS maker fills (below 20-fill threshold)
        if r.verdict != "insufficient_data":
            assert "edge_by_spread_bucket" in r.metrics

    def test_runs_on_tomatoes(self, session_ledgers):
        probe = get_probe("pf01_spread_vs_maker_edge")
        r = probe.run(session_ledgers, "TOMATOES", "test")
        _assert_valid_result(r)


class TestMakerAdverseTiming:
    def test_runs_on_emeralds(self, session_ledgers):
        probe = get_probe("pf02_maker_adverse_timing")
        r = probe.run(session_ledgers, "EMERALDS", "test")
        _assert_valid_result(r)
        # Synthetic fixture has only ~10 EMERALDS maker fills (below 20-fill threshold)
        if r.verdict != "insufficient_data":
            assert "maker_adverse_rate_by_horizon" in r.metrics

    def test_adverse_rates_are_bounded(self, session_ledgers):
        probe = get_probe("pf02_maker_adverse_timing")
        r = probe.run(session_ledgers, "EMERALDS", "test")
        for horizon, rate in r.metrics.get("maker_adverse_rate_by_horizon", {}).items():
            if not math.isnan(rate):
                assert 0.0 <= rate <= 1.0, f"Adverse rate {rate} out of bounds at horizon {horizon}"


class TestMakerFillRateByInventory:
    def test_runs_on_emeralds(self, session_ledgers):
        probe = get_probe("pf03_maker_fill_rate_by_inventory")
        r = probe.run(session_ledgers, "EMERALDS", "test")
        _assert_valid_result(r)
        assert "fills_per_tick_by_inventory" in r.metrics


# ---------------------------------------------------------------------------
# Taking probes
# ---------------------------------------------------------------------------

class TestTakeEdgeByDistance:
    def test_runs_on_emeralds(self, session_ledgers):
        probe = get_probe("tk01_take_edge_by_distance")
        r = probe.run(session_ledgers, "EMERALDS", "test")
        _assert_valid_result(r)
        assert "edge_by_distance_bucket" in r.metrics

    def test_emerald_taker_buys_have_negative_edge(self, session_ledgers):
        """Synthetic EMERALDS taker buys are at 10002, fair=10000 -> edge = -2."""
        probe = get_probe("tk01_take_edge_by_distance")
        r = probe.run(session_ledgers, "EMERALDS", "test")
        if r.verdict != "insufficient_data":
            # All taker fills are at distance 2 from fair, fill_vs_fair = -2
            buckets = r.metrics.get("edge_by_distance_bucket", {})
            # Distance=2 falls into "2-3" bucket
            if "2-3" in buckets:
                assert buckets["2-3"]["mean"] < 0


class TestTakeEdgeBySpread:
    def test_runs_on_tomatoes(self, session_ledgers):
        probe = get_probe("tk02_take_edge_by_spread")
        r = probe.run(session_ledgers, "TOMATOES", "test")
        _assert_valid_result(r)


class TestTakeDirectionVsTrend:
    def test_runs_on_emeralds(self, session_ledgers):
        probe = get_probe("tk03_take_direction_vs_trend")
        r = probe.run(session_ledgers, "EMERALDS", "test")
        _assert_valid_result(r)


# ---------------------------------------------------------------------------
# Inventory probes
# ---------------------------------------------------------------------------

class TestPnlByInventoryLevel:
    def test_runs_on_emeralds(self, session_ledgers):
        probe = get_probe("inv01_pnl_by_inventory_level")
        r = probe.run(session_ledgers, "EMERALDS", "test")
        _assert_valid_result(r)
        assert "pnl_rate_by_position_bucket" in r.metrics

    def test_runs_on_tomatoes(self, session_ledgers):
        probe = get_probe("inv01_pnl_by_inventory_level")
        r = probe.run(session_ledgers, "TOMATOES", "test")
        _assert_valid_result(r)


class TestFillQualityByInventory:
    def test_runs_on_emeralds(self, session_ledgers):
        probe = get_probe("inv02_fill_quality_by_inventory")
        r = probe.run(session_ledgers, "EMERALDS", "test")
        _assert_valid_result(r)
        assert "edge_by_inventory_and_direction" in r.metrics


# ---------------------------------------------------------------------------
# Danger zone probes
# ---------------------------------------------------------------------------

class TestWideSpreadLossAssociation:
    def test_runs_on_emeralds(self, session_ledgers):
        probe = get_probe("dz01_wide_spread_loss")
        r = probe.run(session_ledgers, "EMERALDS", "test")
        _assert_valid_result(r)
        assert "pnl_rate_by_spread_bucket" in r.metrics

    def test_runs_on_tomatoes(self, session_ledgers):
        probe = get_probe("dz01_wide_spread_loss")
        r = probe.run(session_ledgers, "TOMATOES", "test")
        _assert_valid_result(r)


class TestSessionPhaseRisk:
    def test_runs_on_emeralds(self, session_ledgers):
        probe = get_probe("dz02_session_phase_risk")
        r = probe.run(session_ledgers, "EMERALDS", "test")
        _assert_valid_result(r)
        assert "fill_edge_by_phase" in r.metrics
        assert "pnl_rate_by_phase" in r.metrics


# ---------------------------------------------------------------------------
# Cross-probe: all probes should not crash on empty data
# ---------------------------------------------------------------------------

class TestEmptyData:
    def test_all_probes_handle_empty_sessions(self):
        """Every probe should return insufficient_data on empty input, not crash."""
        import mechanics.probes  # noqa: F401
        from mechanics.probe_spec import list_probes

        empty_ledgers = {}
        for spec in list_probes():
            probe = get_probe(spec.probe_id)
            r = probe.run(empty_ledgers, "EMERALDS", "empty")
            assert r.verdict == "insufficient_data", f"{spec.probe_id} didn't return insufficient_data on empty input"
