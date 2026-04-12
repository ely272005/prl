"""Extended regime extraction and regime-conditioned fill analysis.

Extends the 4 base regime dimensions (volatility, spread, inventory, phase)
from analytics/regime_analysis.py with additional dimensions:

  - spread_bucket:      absolute spread width bucketed (from _helpers)
  - position_bucket:    signed position bucketed
  - abs_position_bucket: |position| bucketed
  - trend_10:           10-tick fair value direction (up / flat / down)
  - fair_vs_mid:        sign of fair_value - mid_price
  - maker_friendly:     composite label from spread + volatility

All dimensions are simple, interpretable, and derivable from existing data.
"""
from __future__ import annotations

import math
import statistics
from typing import Any

import pandas as pd

from analytics.regime_analysis import label_session_regimes
from mechanics.probes._helpers import (
    bucket_value,
    bucket_stats,
    enrich_fills,
    SPREAD_EDGES,
    SPREAD_LABELS,
    POSITION_EDGES,
    POSITION_LABELS,
    ABS_POSITION_EDGES,
    ABS_POSITION_LABELS,
)

# ---------------------------------------------------------------------------
# Regime dimension definitions
# ---------------------------------------------------------------------------

REGIME_DIMENSIONS = [
    "spread_bucket",
    "position_bucket",
    "abs_position_bucket",
    "volatility_regime",
    "spread_regime",
    "inventory_regime",
    "session_phase",
    "trend_10",
    "fair_vs_mid",
    "maker_friendly",
]

# Trend thresholds (fair value change over 10 ticks)
TREND_FLAT_THRESHOLD = 0.5  # absolute change < this → "flat"


# ---------------------------------------------------------------------------
# Extended per-tick regime labeling
# ---------------------------------------------------------------------------

def label_extended_regimes(
    traces: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Add extended regime columns to traces for one session.

    Starts with the 4 base dimensions from regime_analysis, then adds:
      - spread_bucket, position_bucket, abs_position_bucket
      - trend_10, fair_vs_mid, maker_friendly
    """
    if traces.empty:
        cols = {dim: pd.Series(dtype=str) for dim in REGIME_DIMENSIONS}
        return traces.assign(**cols)

    # Base regime labeling (volatility, spread_regime, inventory_regime, session_phase)
    labeled = label_session_regimes(traces, prices)

    # Build price lookup for spread and mid
    spread_lookup: dict[tuple, float] = {}
    mid_lookup: dict[tuple, float] = {}
    if not prices.empty:
        for _, row in prices.iterrows():
            key = (int(row["day"]), int(row["timestamp"]), row["product"])
            bid1, ask1 = row["bid1"], row["ask1"]
            if not (math.isnan(bid1) or math.isnan(ask1)):
                spread_lookup[key] = ask1 - bid1
            mid_lookup[key] = row["mid_price"]

    # --- Spread bucket (absolute width) ---
    spread_buckets = []
    for _, row in labeled.iterrows():
        key = (int(row["day"]), int(row["timestamp"]), row["product"])
        s = spread_lookup.get(key, math.nan)
        if math.isnan(s):
            spread_buckets.append("unknown")
        else:
            spread_buckets.append(bucket_value(s, SPREAD_EDGES, SPREAD_LABELS))
    labeled["spread_bucket"] = spread_buckets

    # --- Position bucket (signed) ---
    labeled["position_bucket"] = [
        bucket_value(float(p), POSITION_EDGES, POSITION_LABELS)
        for p in labeled["position"]
    ]

    # --- Abs position bucket ---
    labeled["abs_position_bucket"] = [
        bucket_value(abs(float(p)), ABS_POSITION_EDGES, ABS_POSITION_LABELS)
        for p in labeled["position"]
    ]

    # --- Trend 10: fair value direction over last 10 ticks ---
    trend_labels = []
    for product, group in labeled.groupby("product", sort=False):
        fv = group["fair_value"].tolist()
        indices = group.index.tolist()
        for i, idx in enumerate(indices):
            if i < 10:
                trend_labels.append((idx, "unknown"))
            else:
                change = fv[i] - fv[i - 10]
                if change > TREND_FLAT_THRESHOLD:
                    trend_labels.append((idx, "up"))
                elif change < -TREND_FLAT_THRESHOLD:
                    trend_labels.append((idx, "down"))
                else:
                    trend_labels.append((idx, "flat"))
    # Sort by original index to maintain row order
    trend_labels.sort(key=lambda x: x[0])
    labeled["trend_10"] = [t[1] for t in trend_labels]

    # --- Fair vs mid: sign of (fair_value - mid_price) ---
    fvm_labels = []
    for _, row in labeled.iterrows():
        key = (int(row["day"]), int(row["timestamp"]), row["product"])
        mid = mid_lookup.get(key, math.nan)
        if math.isnan(mid):
            fvm_labels.append("unknown")
        else:
            diff = row["fair_value"] - mid
            if diff > 0.5:
                fvm_labels.append("above_mid")
            elif diff < -0.5:
                fvm_labels.append("below_mid")
            else:
                fvm_labels.append("at_mid")
    labeled["fair_vs_mid"] = fvm_labels

    # --- Maker friendly: composite of spread + volatility ---
    mf_labels = []
    for _, row in labeled.iterrows():
        spread_r = row.get("spread_regime", "unknown")
        vol_r = row.get("volatility_regime", "unknown")
        if spread_r in ("high", "medium") and vol_r in ("low", "medium"):
            mf_labels.append("maker_friendly")
        elif spread_r == "low" and vol_r == "high":
            mf_labels.append("taker_friendly")
        else:
            mf_labels.append("neutral")
    labeled["maker_friendly"] = mf_labels

    return labeled


# ---------------------------------------------------------------------------
# Regime-conditioned fill analysis
# ---------------------------------------------------------------------------

def label_fills_with_regimes(
    strategy_fills: pd.DataFrame,
    labeled_traces: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Add regime labels and enrichment to strategy fills.

    Merges regime labels from labeled traces onto fills by (day, timestamp, product),
    and adds spread / position_at_fill from enrich_fills.
    """
    if strategy_fills.empty:
        return strategy_fills

    # Get original traces for enrich_fills (it needs raw traces)
    raw_cols = ["day", "timestamp", "product", "fair_value", "position", "cash", "mtm_pnl"]
    available = [c for c in raw_cols if c in labeled_traces.columns]
    raw_traces = labeled_traces[available]

    enriched = enrich_fills(strategy_fills, raw_traces, prices)

    # Build regime lookup from labeled traces
    regime_cols = [c for c in REGIME_DIMENSIONS if c in labeled_traces.columns]
    regime_lookup: dict[tuple, dict[str, str]] = {}
    for _, row in labeled_traces.iterrows():
        key = (int(row["day"]), int(row["timestamp"]), row["product"])
        regime_lookup[key] = {col: row[col] for col in regime_cols}

    # Assign regime labels to each fill
    for col in regime_cols:
        values = []
        for _, fill in enriched.iterrows():
            key = (int(fill["day"]), int(fill["timestamp"]), fill["product"])
            regime = regime_lookup.get(key, {})
            values.append(regime.get(col, "unknown"))
        enriched[col] = values

    return enriched


def compute_regime_edge_stats(
    labeled_fills: pd.DataFrame,
    product: str,
    regime_dim: str,
    role: str | None = None,
    min_fills: int = 20,
) -> dict[str, Any]:
    """Compute fill_vs_fair statistics per regime label for one dimension.

    Args:
        labeled_fills: fills with regime columns and fill_vs_fair.
        product: filter to this product.
        regime_dim: which regime dimension column to group by.
        role: "maker" or "taker" or None for all.
        min_fills: minimum fills per bucket to include.

    Returns:
        dict with keys: by_label (stats per label), baseline (overall stats),
        dimension, product, role, total_fills.
    """
    if labeled_fills.empty or "product" not in labeled_fills.columns:
        return {
            "dimension": regime_dim,
            "product": product,
            "role": role,
            "total_fills": 0,
            "baseline": bucket_stats([]),
            "by_label": {},
        }

    df = labeled_fills[labeled_fills["product"] == product].copy()
    if role:
        df = df[df["strategy_role"] == role]

    if df.empty or regime_dim not in df.columns or "fill_vs_fair" not in df.columns:
        return {
            "dimension": regime_dim,
            "product": product,
            "role": role,
            "total_fills": 0,
            "baseline": bucket_stats([]),
            "by_label": {},
        }

    # Baseline stats
    all_values = df["fill_vs_fair"].dropna().tolist()
    baseline = bucket_stats(all_values)

    # Per-label stats
    by_label: dict[str, dict[str, Any]] = {}
    for label, group in df.groupby(regime_dim):
        if label == "unknown":
            continue
        values = group["fill_vs_fair"].dropna().tolist()
        if len(values) < min_fills:
            continue
        stats = bucket_stats(values)
        stats["label"] = label
        by_label[label] = stats

    return {
        "dimension": regime_dim,
        "product": product,
        "role": role,
        "total_fills": len(all_values),
        "baseline": baseline,
        "by_label": by_label,
    }


def build_regime_profile(
    session_ledgers: dict[int, dict[str, pd.DataFrame]],
    products: list[str] | None = None,
) -> dict[str, Any]:
    """Build a full regime profile across all sessions.

    Labels every tick and every fill with extended regimes,
    then computes regime-conditioned stats for each product.

    Returns:
        dict with keys: labeled_fills (DataFrame), regime_stats (nested dict),
        session_count, products.
    """
    if products is None:
        products = ["EMERALDS", "TOMATOES"]

    all_labeled_fills: list[pd.DataFrame] = []
    session_count = 0

    for sid, ledger in session_ledgers.items():
        traces = ledger["traces"]
        prices = ledger["prices"]
        fills = ledger["strategy_fills"]

        if traces.empty:
            continue
        session_count += 1

        labeled_traces = label_extended_regimes(traces, prices)
        labeled_fills = label_fills_with_regimes(fills, labeled_traces, prices)
        if not labeled_fills.empty:
            labeled_fills = labeled_fills.copy()
            labeled_fills["session_id"] = sid
            all_labeled_fills.append(labeled_fills)

    if not all_labeled_fills:
        return {
            "labeled_fills": pd.DataFrame(),
            "regime_stats": {},
            "session_count": 0,
            "products": products,
        }

    combined_fills = pd.concat(all_labeled_fills, ignore_index=True)

    # Compute regime-conditioned edge stats for each product × dimension × role
    regime_stats: dict[str, dict[str, dict[str, Any]]] = {}
    for product in products:
        regime_stats[product] = {}
        for dim in REGIME_DIMENSIONS:
            regime_stats[product][dim] = {}
            for role in [None, "maker", "taker"]:
                role_key = role or "all"
                regime_stats[product][dim][role_key] = compute_regime_edge_stats(
                    combined_fills, product, dim, role=role,
                )

    return {
        "labeled_fills": combined_fills,
        "regime_stats": regime_stats,
        "session_count": session_count,
        "products": products,
    }
