"""End-to-end integration test — runs the full pipeline on synthetic data."""
from __future__ import annotations

import json

from analytics.fill_decomposition import aggregate_fill_decomposition
from analytics.regime_analysis import summarize_regimes
from engine.event_ledger import build_event_ledger
from engine.research_packet import build_packet
from memory.store import PacketStore


class TestEndToEnd:
    def test_full_pipeline(self, output_dir, dashboard_dict, tmp_path):
        """Run the complete observability pipeline and verify output."""
        # Step 1: Event ledger
        ledger = build_event_ledger(output_dir)
        assert ledger["provenance"]["sample_count"] == 2

        # Step 2: Fill decomposition
        fill_decomp = aggregate_fill_decomposition(ledger["session_ledgers"])
        assert fill_decomp["volumes"]["taker_fill_count"] > 0
        assert fill_decomp["volumes"]["maker_fill_count"] > 0

        # Step 3: Regime analysis
        regimes = summarize_regimes(ledger["session_ledgers"])
        assert "EMERALDS" in regimes["by_product"]

        # Step 4: Build packet
        packet = build_packet(
            dashboard=dashboard_dict,
            event_ledger=ledger,
            fill_decomp=fill_decomp,
            regime_summary=regimes,
            strategy_path="/test/strategy.py",
        )
        short = packet["short"]
        full = packet["full"]

        # Validate short structure
        assert short["candidate_id_method"] in ("content_hash", "metadata_fallback")
        assert short["confidence"] in ("HIGH", "MEDIUM", "LOW")
        assert short["pnl"]["mean"] > 0
        assert short["fill_quality"] is not None
        assert short["drawdown"] is not None
        assert isinstance(short["diagnosis"], str)
        assert short["kill"]["recommended"] is False  # profitable strategy
        assert len(short["warnings"]) > 0  # sample-only warnings

        # Efficiency and scale fields (Step 3.5 Tasks 2/3)
        assert short["efficiency"]["pnl_per_fill"] is not None
        assert short["efficiency"]["pnl_per_tick"] is not None
        assert short["efficiency"]["total_strategy_fills"] > 0
        assert short["scale"]["ticks_per_session"] is not None
        assert short["scale"]["ticks_per_session_source"] == "auto_detected"
        assert short["simulation_confidence"] == short["confidence"]
        assert "not calibrated" in short["external_validity_note"].lower()

        # Validate full extends short
        assert "fill_quality_full" in full
        assert "inventory_full" in full
        assert "regime_analysis" in full

        # Step 5: Store and retrieve
        db_path = tmp_path / "test_e2e.db"
        store = PacketStore(db_path)
        run_id = store.store(short, full)
        retrieved = store.get_by_run_id(run_id)
        assert retrieved is not None
        assert retrieved["packet_short"]["candidate_id"] == short["candidate_id"]
        assert retrieved["confidence"] == short["confidence"]

        # Verify round-trip JSON fidelity
        assert retrieved["packet_short"]["pnl"]["mean"] == short["pnl"]["mean"]
        store.close()

    def test_pipeline_no_sample_sessions(self, output_dir, dashboard_dict, tmp_path):
        """Pipeline should work (with nulls) when no sample sessions exist."""
        # Use a directory with only session_summary.csv, no sessions/ dir
        import shutil
        minimal_dir = tmp_path / "minimal"
        minimal_dir.mkdir()
        shutil.copy(output_dir / "session_summary.csv", minimal_dir / "session_summary.csv")

        ledger = build_event_ledger(minimal_dir)
        assert ledger["provenance"]["sample_count"] == 0

        fill_decomp = aggregate_fill_decomposition(ledger["session_ledgers"])
        assert fill_decomp["volumes"]["taker_fill_count"] == 0

        regimes = summarize_regimes(ledger["session_ledgers"])
        assert regimes["by_product"] == {}

        packet = build_packet(
            dashboard=dashboard_dict,
            event_ledger=ledger,
            fill_decomp=fill_decomp,
            regime_summary=regimes,
        )
        short = packet["short"]
        # Fill quality and drawdown should be None (no sample data)
        assert short["fill_quality"] is None
        assert short["drawdown"] is None
        # Still should have P&L stats from dashboard
        assert short["pnl"]["mean"] > 0
