"""Fill Decomposition — maker/taker volumes, fill edge, adverse selection.

All formulas match the Step 2.5 Metric Definitions Sheet exactly.
Evidence scope: SAMPLE sessions only (requires per-tick trade + trace CSVs).
"""
from __future__ import annotations

import math
import statistics
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Maker / Taker volume decomposition
# ---------------------------------------------------------------------------

def maker_taker_volumes(strategy_fills: pd.DataFrame) -> dict[str, Any]:
    """Count and sum maker vs taker fills.

    Args:
        strategy_fills: DataFrame from event_ledger with strategy_role column.
    """
    if strategy_fills.empty:
        return {
            "taker_fill_count": 0,
            "maker_fill_count": 0,
            "taker_volume": 0,
            "maker_volume": 0,
            "passive_fill_rate": 0.0,
        }
    maker = strategy_fills[strategy_fills["strategy_role"] == "maker"]
    taker = strategy_fills[strategy_fills["strategy_role"] == "taker"]
    taker_count = len(taker)
    maker_count = len(maker)
    total = taker_count + maker_count
    return {
        "taker_fill_count": taker_count,
        "maker_fill_count": maker_count,
        "taker_volume": int(taker["quantity"].sum()) if not taker.empty else 0,
        "maker_volume": int(maker["quantity"].sum()) if not maker.empty else 0,
        "passive_fill_rate": maker_count / total if total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Fill edge metrics
# ---------------------------------------------------------------------------

def fill_vs_fair_stats(strategy_fills: pd.DataFrame) -> dict[str, Any]:
    """Compute fill_vs_fair statistics per product.

    fill_vs_fair: (fill_price - fair_value) * direction
      buy  direction = -1  -> fair - price   (bought below fair = positive = good)
      sell direction = +1  -> price - fair   (sold above fair = positive = good)

    Returns dict keyed by product with mean, median, std.
    """
    result: dict[str, Any] = {}
    if strategy_fills.empty or "fill_vs_fair" not in strategy_fills.columns:
        return result

    for product, group in strategy_fills.groupby("product"):
        values = group["fill_vs_fair"].dropna().tolist()
        if not values:
            result[product] = {"mean": 0.0, "median": 0.0, "std": 0.0, "count": 0}
            continue
        result[product] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) >= 2 else 0.0,
            "count": len(values),
        }
    return result


def fill_vs_mid_stats(strategy_fills: pd.DataFrame) -> dict[str, Any]:
    """Compute fill_vs_mid statistics per product.

    Same convention as fill_vs_fair but relative to mid price.
    """
    result: dict[str, Any] = {}
    if strategy_fills.empty or "fill_vs_mid" not in strategy_fills.columns:
        return result

    for product, group in strategy_fills.groupby("product"):
        values = group["fill_vs_mid"].dropna().tolist()
        if not values:
            result[product] = {"mean": 0.0, "median": 0.0, "std": 0.0, "count": 0}
            continue
        result[product] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) >= 2 else 0.0,
            "count": len(values),
        }
    return result


# ---------------------------------------------------------------------------
# Adverse selection rate
# ---------------------------------------------------------------------------

def adverse_selection_rate(
    strategy_fills: pd.DataFrame,
    traces: pd.DataFrame,
    forward_ticks: int = 100,
) -> dict[str, Any]:
    """Fraction of fills where fair value moved against the strategy.

    For each fill at tick t with direction d:
      Dfair = fair_{t+k} - fair_t
      Adverse if d * Dfair < 0
        buy (d=+1): adverse if fair dropped after buy
        sell (d=-1): adverse if fair rose after sell

    Note: d here uses the sign convention: +1 for buy, -1 for sell.
    This differs from the fill_vs_fair direction sign.

    Args:
        strategy_fills: enriched fills with fair_value, strategy_side.
        traces: trace DataFrame with fair_value time series.
        forward_ticks: number of ticks ahead to evaluate (1 tick = 1 row in trace).
    """
    result: dict[str, Any] = {}
    if strategy_fills.empty or traces.empty:
        return result

    for product in strategy_fills["product"].unique():
        product_fills = strategy_fills[strategy_fills["product"] == product]
        product_traces = traces[traces["product"] == product].sort_values(
            ["day", "timestamp"]
        ).reset_index(drop=True)

        if product_traces.empty or product_fills.empty:
            result[product] = {"rate": 0.0, "count": 0, "adverse_count": 0}
            continue

        # Build timestamp -> trace index lookup
        ts_to_idx: dict[tuple[int, int], int] = {}
        for idx, row in product_traces.iterrows():
            ts_to_idx[(int(row["day"]), int(row["timestamp"]))] = idx

        adverse_count = 0
        evaluated_count = 0
        max_idx = len(product_traces) - 1

        for _, fill in product_fills.iterrows():
            key = (int(fill["day"]), int(fill["timestamp"]))
            fill_idx = ts_to_idx.get(key)
            if fill_idx is None:
                continue
            future_idx = fill_idx + forward_ticks
            if future_idx > max_idx:
                continue  # not enough future data

            fair_now = product_traces.iloc[fill_idx]["fair_value"]
            fair_future = product_traces.iloc[future_idx]["fair_value"]
            delta_fair = fair_future - fair_now

            # direction: +1 for buy, -1 for sell
            d = 1.0 if fill["strategy_side"] == "buy" else -1.0
            evaluated_count += 1
            if d * delta_fair < 0:
                adverse_count += 1

        rate = adverse_count / evaluated_count if evaluated_count > 0 else 0.0
        result[product] = {
            "rate": rate,
            "count": evaluated_count,
            "adverse_count": adverse_count,
        }

    return result


# ---------------------------------------------------------------------------
# Aggregate fill decomposition for a set of sample sessions
# ---------------------------------------------------------------------------

def aggregate_fill_decomposition(
    session_ledgers: dict[int, dict[str, pd.DataFrame]],
    forward_ticks: int = 100,
) -> dict[str, Any]:
    """Run all fill analytics across sample sessions and aggregate.

    Args:
        session_ledgers: dict of session_id -> build_session_ledger() output.
        forward_ticks: ticks ahead for adverse selection calculation.
    """
    all_fills = []
    all_volumes: list[dict[str, Any]] = []
    all_adverse: dict[str, list[dict[str, Any]]] = {}

    for session_id, ledger in session_ledgers.items():
        fills = ledger["strategy_fills"]
        traces = ledger["traces"]

        if not fills.empty:
            fills_with_session = fills.copy()
            fills_with_session["session_id"] = session_id
            all_fills.append(fills_with_session)

        all_volumes.append(maker_taker_volumes(fills))

        adv = adverse_selection_rate(fills, traces, forward_ticks)
        for product, stats in adv.items():
            all_adverse.setdefault(product, []).append(stats)

    # Aggregate fills
    combined_fills = pd.concat(all_fills, ignore_index=True) if all_fills else pd.DataFrame()

    # Aggregate volumes
    total_taker = sum(v["taker_fill_count"] for v in all_volumes)
    total_maker = sum(v["maker_fill_count"] for v in all_volumes)
    total_fills = total_taker + total_maker

    # Aggregate adverse selection
    adverse_agg: dict[str, Any] = {}
    for product, stats_list in all_adverse.items():
        total_evaluated = sum(s["count"] for s in stats_list)
        total_adverse = sum(s["adverse_count"] for s in stats_list)
        adverse_agg[product] = {
            "rate": total_adverse / total_evaluated if total_evaluated > 0 else 0.0,
            "count": total_evaluated,
            "adverse_count": total_adverse,
        }

    return {
        "volumes": {
            "taker_fill_count": total_taker,
            "maker_fill_count": total_maker,
            "passive_fill_rate": total_maker / total_fills if total_fills > 0 else 0.0,
        },
        "fill_vs_fair": fill_vs_fair_stats(combined_fills),
        "fill_vs_mid": fill_vs_mid_stats(combined_fills),
        "adverse_selection": adverse_agg,
        "provenance": {
            "scope": "sample",
            "session_count": len(session_ledgers),
            "total_strategy_fills": total_fills,
            "forward_ticks": forward_ticks,
            "warning": (
                f"Fill metrics from {len(session_ledgers)} sample sessions "
                f"({total_fills} strategy fills), not the full ensemble."
            ),
        },
    }
