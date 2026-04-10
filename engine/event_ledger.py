"""Event Ledger V1 — reads sample-session CSVs and builds a unified DataFrame.

Reads trace, price, and trade CSVs produced by the Rust MC simulator for
sample sessions.  Outputs a per-tick DataFrame with provenance metadata.

CSV conventions (from Rust main.rs):
  - trace / price / trade CSVs use `;` delimiter
  - session_summary.csv uses `,` delimiter
  - Trade buyer/seller tags: SUBMISSION, BOT, BOT_TAKER, BOT_MAKER
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# CSV readers (match upstream delimiter conventions)
# ---------------------------------------------------------------------------

def _read_csv(path: Path, delimiter: str = ";") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter=delimiter))


# ---------------------------------------------------------------------------
# Trade tag constants (validated in Step 2.5 Trade Tag Memo)
# ---------------------------------------------------------------------------

STRATEGY_TAG = "SUBMISSION"
BOT_PASSIVE_TAG = "BOT"
BOT_TAKER_TAG = "BOT_TAKER"
BOT_MAKER_TAG = "BOT_MAKER"


def classify_fill(buyer: str, seller: str) -> dict[str, Any]:
    """Return strategy_involved, strategy_side, strategy_role for a fill.

    Rules (from Rust main.rs, exhaustive):
      SUBMISSION/BOT       -> strategy BUY,  TAKER
      BOT/SUBMISSION       -> strategy SELL, TAKER
      BOT_TAKER/SUBMISSION -> strategy SELL, MAKER
      SUBMISSION/BOT_TAKER -> strategy BUY,  MAKER
      BOT_TAKER/BOT_MAKER  -> no strategy involvement
      BOT_MAKER/BOT_TAKER  -> no strategy involvement
    """
    if buyer == STRATEGY_TAG and seller == BOT_PASSIVE_TAG:
        return {"strategy_involved": True, "strategy_side": "buy", "strategy_role": "taker"}
    if buyer == BOT_PASSIVE_TAG and seller == STRATEGY_TAG:
        return {"strategy_involved": True, "strategy_side": "sell", "strategy_role": "taker"}
    if buyer == BOT_TAKER_TAG and seller == STRATEGY_TAG:
        return {"strategy_involved": True, "strategy_side": "sell", "strategy_role": "maker"}
    if buyer == STRATEGY_TAG and seller == BOT_TAKER_TAG:
        return {"strategy_involved": True, "strategy_side": "buy", "strategy_role": "maker"}
    return {"strategy_involved": False, "strategy_side": None, "strategy_role": None}


# ---------------------------------------------------------------------------
# Loaders for individual CSV types
# ---------------------------------------------------------------------------

def load_trace_rows(session_dir: Path) -> pd.DataFrame:
    """Load all trace CSV files for a session into a single DataFrame."""
    round_dir = session_dir / "round0"
    frames = []
    for path in sorted(round_dir.glob("trace_round_0_day_*.csv")):
        day = int(path.stem.split("_")[-1])
        rows = _read_csv(path, ";")
        for row in rows:
            frames.append({
                "day": day,
                "timestamp": int(row["timestamp"]),
                "product": row["product"],
                "fair_value": float(row["fair_value"]),
                "position": int(row["position"]),
                "cash": float(row["cash"]),
                "mtm_pnl": float(row["mtm_pnl"]),
            })
    if not frames:
        return pd.DataFrame(columns=[
            "day", "timestamp", "product", "fair_value",
            "position", "cash", "mtm_pnl",
        ])
    return pd.DataFrame(frames)


def load_price_rows(session_dir: Path) -> pd.DataFrame:
    """Load all price CSV files for a session."""
    round_dir = session_dir / "round0"
    frames = []
    for path in sorted(round_dir.glob("prices_round_0_day_*.csv")):
        day = int(path.stem.split("_")[-1])
        rows = _read_csv(path, ";")
        for row in rows:
            bid1 = float(row["bid_price_1"]) if row.get("bid_price_1") not in ("", None) else math.nan
            ask1 = float(row["ask_price_1"]) if row.get("ask_price_1") not in ("", None) else math.nan
            mid = float(row["mid_price"])
            frames.append({
                "day": day,
                "timestamp": int(row["timestamp"]),
                "product": row["product"],
                "bid1": bid1,
                "ask1": ask1,
                "mid_price": mid,
            })
    if not frames:
        return pd.DataFrame(columns=[
            "day", "timestamp", "product", "bid1", "ask1", "mid_price",
        ])
    return pd.DataFrame(frames)


def load_trade_rows(session_dir: Path) -> pd.DataFrame:
    """Load all trade CSV files for a session with maker/taker classification."""
    round_dir = session_dir / "round0"
    frames = []
    for path in sorted(round_dir.glob("trades_round_0_day_*.csv")):
        day = int(path.stem.split("_")[-1])
        rows = _read_csv(path, ";")
        for row in rows:
            buyer = row.get("buyer", "")
            seller = row.get("seller", "")
            classification = classify_fill(buyer, seller)
            frames.append({
                "day": day,
                "timestamp": int(row["timestamp"]),
                "product": row["symbol"],
                "price": float(row["price"]),
                "quantity": int(row["quantity"]),
                "buyer": buyer,
                "seller": seller,
                **classification,
            })
    if not frames:
        return pd.DataFrame(columns=[
            "day", "timestamp", "product", "price", "quantity",
            "buyer", "seller", "strategy_involved", "strategy_side",
            "strategy_role",
        ])
    return pd.DataFrame(frames)


# ---------------------------------------------------------------------------
# Event Ledger builder
# ---------------------------------------------------------------------------

def _merge_tick_context(
    trades: pd.DataFrame,
    traces: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Enrich strategy fills with fair value and mid price at fill time."""
    strategy_fills = trades[trades["strategy_involved"]].copy()
    if strategy_fills.empty:
        strategy_fills["fair_value"] = pd.Series(dtype=float)
        strategy_fills["mid_price"] = pd.Series(dtype=float)
        strategy_fills["fill_vs_fair"] = pd.Series(dtype=float)
        strategy_fills["fill_vs_mid"] = pd.Series(dtype=float)
        return strategy_fills

    # Build lookup dicts keyed on (day, timestamp, product)
    trace_lookup: dict[tuple, float] = {}
    for _, row in traces.iterrows():
        trace_lookup[(row["day"], row["timestamp"], row["product"])] = row["fair_value"]

    price_lookup: dict[tuple, float] = {}
    for _, row in prices.iterrows():
        price_lookup[(row["day"], row["timestamp"], row["product"])] = row["mid_price"]

    fair_values = []
    mid_prices = []
    for _, fill in strategy_fills.iterrows():
        key = (fill["day"], fill["timestamp"], fill["product"])
        fair_values.append(trace_lookup.get(key, math.nan))
        mid_prices.append(price_lookup.get(key, math.nan))

    strategy_fills["fair_value"] = fair_values
    strategy_fills["mid_price"] = mid_prices

    # fill_vs_fair: positive = good for strategy
    # buy:  fair - price  (bought below fair = good)
    # sell: price - fair  (sold above fair = good)
    direction = strategy_fills["strategy_side"].map({"buy": -1.0, "sell": 1.0})
    strategy_fills["fill_vs_fair"] = (strategy_fills["price"] - strategy_fills["fair_value"]) * direction
    strategy_fills["fill_vs_mid"] = (strategy_fills["price"] - strategy_fills["mid_price"]) * direction

    return strategy_fills


def build_session_ledger(session_dir: Path) -> dict[str, pd.DataFrame]:
    """Build event ledger for a single sample session.

    Returns dict with keys:
      - "traces": per-tick state (position, cash, mtm_pnl, fair_value)
      - "prices": per-tick book snapshot (bid1, ask1, mid)
      - "all_trades": all fills including bot-vs-bot
      - "strategy_fills": enriched strategy fills with fair/mid/edge
    """
    traces = load_trace_rows(session_dir)
    prices = load_price_rows(session_dir)
    trades = load_trade_rows(session_dir)
    strategy_fills = _merge_tick_context(trades, traces, prices)
    return {
        "traces": traces,
        "prices": prices,
        "all_trades": trades,
        "strategy_fills": strategy_fills,
    }


def build_event_ledger(output_dir: Path) -> dict[str, Any]:
    """Build event ledger across all sample sessions.

    Args:
        output_dir: The backtest output directory containing sessions/ subdirectory.

    Returns:
        Dict with:
          - "session_ledgers": dict mapping session_id -> build_session_ledger() output
          - "session_ids": list of session IDs with trace data
          - "provenance": metadata about evidence scope
    """
    sessions_dir = output_dir / "sessions"
    if not sessions_dir.exists():
        return {
            "session_ledgers": {},
            "session_ids": [],
            "provenance": {
                "scope": "sample",
                "sample_count": 0,
                "warning": "No sample session data found.",
            },
        }

    session_ledgers: dict[int, dict[str, pd.DataFrame]] = {}
    session_ids: list[int] = []

    for session_path in sorted(sessions_dir.iterdir()):
        if not session_path.is_dir() or not session_path.name.startswith("session_"):
            continue
        session_id = int(session_path.name.split("_")[-1])
        round_dir = session_path / "round0"
        if not round_dir.exists():
            continue
        ledger = build_session_ledger(session_path)
        session_ledgers[session_id] = ledger
        session_ids.append(session_id)

    return {
        "session_ledgers": session_ledgers,
        "session_ids": session_ids,
        "provenance": {
            "scope": "sample",
            "sample_count": len(session_ids),
            "warning": (
                f"Event ledger built from {len(session_ids)} sample sessions, "
                "not the full ensemble."
                if session_ids
                else "No sample session data found."
            ),
        },
    }


def load_session_summaries(output_dir: Path) -> pd.DataFrame:
    """Load session_summary.csv (all sessions, comma-delimited)."""
    path = output_dir / "session_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    rows = _read_csv(path, ",")
    records = []
    for row in rows:
        records.append({
            "session_id": int(row["session_id"]),
            "total_pnl": float(row["total_pnl"]),
            "emerald_pnl": float(row["emerald_pnl"]),
            "tomato_pnl": float(row["tomato_pnl"]),
            "emerald_position": int(row["emerald_position"]),
            "tomato_position": int(row["tomato_position"]),
            "emerald_cash": float(row["emerald_cash"]),
            "tomato_cash": float(row["tomato_cash"]),
            "total_slope_per_step": float(row.get("total_slope_per_step", 0) or 0),
            "total_r2": float(row.get("total_r2", 0) or 0),
            "emerald_slope_per_step": float(row.get("emerald_slope_per_step", 0) or 0),
            "emerald_r2": float(row.get("emerald_r2", 0) or 0),
            "tomato_slope_per_step": float(row.get("tomato_slope_per_step", 0) or 0),
            "tomato_r2": float(row.get("tomato_r2", 0) or 0),
        })
    return pd.DataFrame(records)
