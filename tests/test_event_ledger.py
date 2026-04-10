"""Tests for engine/event_ledger.py."""
from __future__ import annotations

import pandas as pd

from engine.event_ledger import (
    build_event_ledger,
    build_session_ledger,
    classify_fill,
    load_price_rows,
    load_session_summaries,
    load_trace_rows,
    load_trade_rows,
)


# ---------------------------------------------------------------------------
# classify_fill — exhaustive tag validation
# ---------------------------------------------------------------------------

class TestClassifyFill:
    """Validate all 6 buyer/seller tag combinations from Rust main.rs."""

    def test_strategy_taker_buy(self):
        result = classify_fill("SUBMISSION", "BOT")
        assert result["strategy_involved"] is True
        assert result["strategy_side"] == "buy"
        assert result["strategy_role"] == "taker"

    def test_strategy_taker_sell(self):
        result = classify_fill("BOT", "SUBMISSION")
        assert result["strategy_involved"] is True
        assert result["strategy_side"] == "sell"
        assert result["strategy_role"] == "taker"

    def test_strategy_maker_sell(self):
        """BOT_TAKER buys from strategy's passive ask."""
        result = classify_fill("BOT_TAKER", "SUBMISSION")
        assert result["strategy_involved"] is True
        assert result["strategy_side"] == "sell"
        assert result["strategy_role"] == "maker"

    def test_strategy_maker_buy(self):
        """BOT_TAKER sells to strategy's passive bid."""
        result = classify_fill("SUBMISSION", "BOT_TAKER")
        assert result["strategy_involved"] is True
        assert result["strategy_side"] == "buy"
        assert result["strategy_role"] == "maker"

    def test_bot_vs_bot_buy(self):
        result = classify_fill("BOT_TAKER", "BOT_MAKER")
        assert result["strategy_involved"] is False
        assert result["strategy_side"] is None
        assert result["strategy_role"] is None

    def test_bot_vs_bot_sell(self):
        result = classify_fill("BOT_MAKER", "BOT_TAKER")
        assert result["strategy_involved"] is False
        assert result["strategy_side"] is None
        assert result["strategy_role"] is None


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

class TestLoadCSVs:
    def test_load_traces(self, output_dir):
        session_path = output_dir / "sessions" / "session_00000"
        df = load_trace_rows(session_path)
        assert not df.empty
        assert set(df["product"].unique()) == {"EMERALDS", "TOMATOES"}
        assert "fair_value" in df.columns
        assert "mtm_pnl" in df.columns

    def test_load_prices(self, output_dir):
        session_path = output_dir / "sessions" / "session_00000"
        df = load_price_rows(session_path)
        assert not df.empty
        assert "mid_price" in df.columns
        assert "bid1" in df.columns

    def test_load_trades(self, output_dir):
        session_path = output_dir / "sessions" / "session_00000"
        df = load_trade_rows(session_path)
        assert not df.empty
        assert "strategy_involved" in df.columns
        assert "strategy_role" in df.columns
        # Should have both strategy and non-strategy fills
        assert df["strategy_involved"].any()
        assert not df["strategy_involved"].all()

    def test_trade_maker_taker_tags(self, output_dir):
        session_path = output_dir / "sessions" / "session_00000"
        df = load_trade_rows(session_path)
        strategy = df[df["strategy_involved"]]
        roles = set(strategy["strategy_role"].unique())
        assert "maker" in roles
        assert "taker" in roles

    def test_load_session_summaries(self, output_dir):
        df = load_session_summaries(output_dir)
        assert len(df) == 3
        assert "total_pnl" in df.columns
        assert "emerald_position" in df.columns


# ---------------------------------------------------------------------------
# Session ledger
# ---------------------------------------------------------------------------

class TestSessionLedger:
    def test_build_session_ledger(self, output_dir):
        session_path = output_dir / "sessions" / "session_00000"
        ledger = build_session_ledger(session_path)
        assert "traces" in ledger
        assert "prices" in ledger
        assert "all_trades" in ledger
        assert "strategy_fills" in ledger

    def test_strategy_fills_have_edge(self, output_dir):
        session_path = output_dir / "sessions" / "session_00000"
        ledger = build_session_ledger(session_path)
        fills = ledger["strategy_fills"]
        assert not fills.empty
        assert "fill_vs_fair" in fills.columns
        assert "fill_vs_mid" in fills.columns

    def test_fill_vs_fair_sign_convention(self, output_dir):
        """Buying below fair should give positive fill_vs_fair."""
        session_path = output_dir / "sessions" / "session_00000"
        ledger = build_session_ledger(session_path)
        fills = ledger["strategy_fills"]
        # Maker buys at 9999, EMERALD fair = 10000 -> fill_vs_fair = 10000 - 9999 = 1.0
        maker_buys = fills[
            (fills["strategy_role"] == "maker")
            & (fills["strategy_side"] == "buy")
            & (fills["product"] == "EMERALDS")
        ]
        if not maker_buys.empty:
            assert (maker_buys["fill_vs_fair"] > 0).all(), (
                "Buying below fair should yield positive fill_vs_fair"
            )


# ---------------------------------------------------------------------------
# Full event ledger
# ---------------------------------------------------------------------------

class TestBuildEventLedger:
    def test_build_event_ledger(self, output_dir):
        ledger = build_event_ledger(output_dir)
        assert len(ledger["session_ids"]) == 2
        assert ledger["provenance"]["scope"] == "sample"
        assert ledger["provenance"]["sample_count"] == 2

    def test_missing_sessions_dir(self, tmp_path):
        ledger = build_event_ledger(tmp_path)
        assert ledger["session_ids"] == []
        assert ledger["provenance"]["sample_count"] == 0
