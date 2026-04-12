"""Shared helpers for probe implementations.

Provides data enrichment and bucketing utilities that multiple probes reuse.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Enrichment: add spread and position context to strategy fills
# ---------------------------------------------------------------------------

def enrich_fills(
    strategy_fills: pd.DataFrame,
    traces: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Add spread and position_at_fill columns to strategy fills.

    Returns a copy; does not mutate the input.
    """
    if strategy_fills.empty:
        return strategy_fills.assign(
            spread=pd.Series(dtype=float),
            position_at_fill=pd.Series(dtype=int),
        )

    # Build spread lookup: (day, timestamp, product) -> ask1 - bid1
    spread_lookup: dict[tuple, float] = {}
    if not prices.empty:
        for _, row in prices.iterrows():
            bid1, ask1 = row["bid1"], row["ask1"]
            if not (math.isnan(bid1) or math.isnan(ask1)):
                spread_lookup[(int(row["day"]), int(row["timestamp"]), row["product"])] = ask1 - bid1

    # Build position lookup: (day, timestamp, product) -> position
    pos_lookup: dict[tuple, int] = {}
    if not traces.empty:
        for _, row in traces.iterrows():
            pos_lookup[(int(row["day"]), int(row["timestamp"]), row["product"])] = int(row["position"])

    spreads = []
    positions = []
    for _, fill in strategy_fills.iterrows():
        key = (int(fill["day"]), int(fill["timestamp"]), fill["product"])
        spreads.append(spread_lookup.get(key, math.nan))
        positions.append(pos_lookup.get(key, 0))

    result = strategy_fills.copy()
    result["spread"] = spreads
    result["position_at_fill"] = positions
    return result


# ---------------------------------------------------------------------------
# Forward fair value lookup
# ---------------------------------------------------------------------------

def build_fair_index(traces: pd.DataFrame, product: str) -> pd.DataFrame:
    """Build an ordered fair value index for one product.

    Returns DataFrame with columns: day, timestamp, fair_value, mtm_pnl, position
    sorted by (day, timestamp) with a clean integer index.
    """
    df = traces[traces["product"] == product].copy()
    if df.empty:
        return df
    df = df.sort_values(["day", "timestamp"]).reset_index(drop=True)
    return df


def fair_change_at_fill(
    fills: pd.DataFrame,
    fair_index: pd.DataFrame,
    forward_ticks: int,
) -> list[float]:
    """For each fill, compute fair value change over next N ticks.

    Returns a list parallel to fills rows. NaN if not enough future data.
    """
    if fills.empty or fair_index.empty:
        return []

    # Build (day, timestamp) -> index in fair_index
    ts_to_idx: dict[tuple[int, int], int] = {}
    for idx, row in fair_index.iterrows():
        ts_to_idx[(int(row["day"]), int(row["timestamp"]))] = idx

    max_idx = len(fair_index) - 1
    changes = []
    for _, fill in fills.iterrows():
        key = (int(fill["day"]), int(fill["timestamp"]))
        fill_idx = ts_to_idx.get(key)
        if fill_idx is None or fill_idx + forward_ticks > max_idx:
            changes.append(math.nan)
        else:
            fair_now = fair_index.iloc[fill_idx]["fair_value"]
            fair_future = fair_index.iloc[fill_idx + forward_ticks]["fair_value"]
            changes.append(fair_future - fair_now)
    return changes


def fair_change_before_fill(
    fills: pd.DataFrame,
    fair_index: pd.DataFrame,
    lookback_ticks: int,
) -> list[float]:
    """For each fill, compute fair value change over preceding N ticks.

    Returns a list parallel to fills rows. NaN if not enough history.
    """
    if fills.empty or fair_index.empty:
        return []

    ts_to_idx: dict[tuple[int, int], int] = {}
    for idx, row in fair_index.iterrows():
        ts_to_idx[(int(row["day"]), int(row["timestamp"]))] = idx

    changes = []
    for _, fill in fills.iterrows():
        key = (int(fill["day"]), int(fill["timestamp"]))
        fill_idx = ts_to_idx.get(key)
        if fill_idx is None or fill_idx - lookback_ticks < 0:
            changes.append(math.nan)
        else:
            fair_now = fair_index.iloc[fill_idx]["fair_value"]
            fair_past = fair_index.iloc[fill_idx - lookback_ticks]["fair_value"]
            changes.append(fair_now - fair_past)
    return changes


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------

def bucket_value(value: float, edges: list[float], labels: list[str]) -> str:
    """Assign a value to a named bucket.

    edges defines the upper boundaries (exclusive) for each bucket.
    labels must have len(edges) + 1 entries (one for each interval plus overflow).

    Example: edges=[2,4,6], labels=["<2","2-4","4-6","6+"]
    """
    for i, edge in enumerate(edges):
        if value < edge:
            return labels[i]
    return labels[-1]


# Reusable bucket definitions
SPREAD_EDGES = [2.0, 4.0, 6.0, 8.0]
SPREAD_LABELS = ["<2", "2-4", "4-6", "6-8", "8+"]

DISTANCE_EDGES = [1.0, 2.0, 3.0, 5.0]
DISTANCE_LABELS = ["<1", "1-2", "2-3", "3-5", "5+"]

POSITION_EDGES = [-7.0, -3.0, 3.0, 7.0]
POSITION_LABELS = ["deep_short", "short", "flat", "long", "deep_long"]

ABS_POSITION_EDGES = [3.0, 7.0]
ABS_POSITION_LABELS = ["low", "medium", "high"]


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def bucket_stats(values: list[float]) -> dict[str, Any]:
    """Compute mean, median, std, count for a list of floats (NaN-safe)."""
    clean = [v for v in values if not math.isnan(v)]
    n = len(clean)
    if n == 0:
        return {"mean": math.nan, "median": math.nan, "std": math.nan, "count": 0}
    mean = sum(clean) / n
    sorted_v = sorted(clean)
    median = sorted_v[n // 2] if n % 2 == 1 else (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2
    if n >= 2:
        var = sum((v - mean) ** 2 for v in clean) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return {"mean": mean, "median": median, "std": std, "count": n}
