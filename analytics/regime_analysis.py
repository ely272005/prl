"""Regime Analysis — label ticks and sessions by market regime.

Computes four regime dimensions from event ledger data:
  - volatility_regime:  fair value volatility (TOMATOES only; EMERALDS is constant)
  - spread_regime:      bid-ask spread width
  - inventory_regime:   absolute position level
  - session_phase:      early / mid / late within a session

Evidence scope: SAMPLE sessions only.
"""
from __future__ import annotations

import math
import statistics
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Regime thresholds
#
# These are percentile-based within each session.  The defaults below split
# into thirds (low / medium / high) using the 33rd and 67th percentiles.
# ---------------------------------------------------------------------------

LOW_PERCENTILE = 0.33
HIGH_PERCENTILE = 0.67

# Session phase boundaries as fraction of total ticks
EARLY_PHASE_END = 0.10
LATE_PHASE_START = 0.90


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = q * (len(s) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return s[lo]
    w = idx - lo
    return s[lo] * (1.0 - w) + s[hi] * w


def _label_by_thresholds(value: float, low_th: float, high_th: float) -> str:
    if value <= low_th:
        return "low"
    if value >= high_th:
        return "high"
    return "medium"


# ---------------------------------------------------------------------------
# Per-tick regime labeling for a single session
# ---------------------------------------------------------------------------

def label_session_regimes(
    traces: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Add regime columns to the trace DataFrame for one session.

    Returns a new DataFrame with columns added:
      - volatility_regime (per product, only meaningful for TOMATOES)
      - spread_regime (per product)
      - inventory_regime (per product)
      - session_phase
    """
    if traces.empty:
        return traces.assign(
            volatility_regime=pd.Series(dtype=str),
            spread_regime=pd.Series(dtype=str),
            inventory_regime=pd.Series(dtype=str),
            session_phase=pd.Series(dtype=str),
        )

    result = traces.copy()
    result = result.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)

    # --- Volatility regime (rolling absolute fair value changes) ---
    vol_labels = []
    for product, group in result.groupby("product"):
        fv = group["fair_value"].tolist()
        # Absolute tick-to-tick fair value changes
        abs_changes = [0.0] + [abs(fv[i] - fv[i - 1]) for i in range(1, len(fv))]
        # Rolling window of 50 ticks for smoothing
        window = 50
        rolling_vol = []
        for i in range(len(abs_changes)):
            start = max(0, i - window + 1)
            rolling_vol.append(statistics.fmean(abs_changes[start : i + 1]))

        if product == "EMERALDS":
            # Fair value is constant 10000 — volatility is always 0
            vol_labels.extend(["low"] * len(group))
        else:
            low_th = _quantile(rolling_vol, LOW_PERCENTILE)
            high_th = _quantile(rolling_vol, HIGH_PERCENTILE)
            vol_labels.extend(
                _label_by_thresholds(v, low_th, high_th) for v in rolling_vol
            )
    result["volatility_regime"] = vol_labels

    # --- Spread regime ---
    # Merge price data to get bid-ask spread
    spread_lookup: dict[tuple, float] = {}
    if not prices.empty:
        for _, row in prices.iterrows():
            bid1 = row["bid1"]
            ask1 = row["ask1"]
            if not (math.isnan(bid1) or math.isnan(ask1)):
                spread_lookup[(int(row["day"]), int(row["timestamp"]), row["product"])] = ask1 - bid1

    spread_labels = []
    for product, group in result.groupby("product"):
        spreads = []
        for _, row in group.iterrows():
            key = (int(row["day"]), int(row["timestamp"]), row["product"])
            spreads.append(spread_lookup.get(key, math.nan))

        valid_spreads = [s for s in spreads if not math.isnan(s)]
        if not valid_spreads:
            spread_labels.extend(["unknown"] * len(group))
        else:
            low_th = _quantile(valid_spreads, LOW_PERCENTILE)
            high_th = _quantile(valid_spreads, HIGH_PERCENTILE)
            for s in spreads:
                if math.isnan(s):
                    spread_labels.append("unknown")
                else:
                    spread_labels.append(_label_by_thresholds(s, low_th, high_th))
    result["spread_regime"] = spread_labels

    # --- Inventory regime ---
    inv_labels = []
    for product, group in result.groupby("product"):
        positions = [abs(int(p)) for p in group["position"].tolist()]
        if not positions or max(positions) == 0:
            inv_labels.extend(["low"] * len(group))
        else:
            low_th = _quantile([float(p) for p in positions], LOW_PERCENTILE)
            high_th = _quantile([float(p) for p in positions], HIGH_PERCENTILE)
            inv_labels.extend(
                _label_by_thresholds(float(p), low_th, high_th) for p in positions
            )
    result["inventory_regime"] = inv_labels

    # --- Session phase ---
    phase_labels = []
    for product, group in result.groupby("product"):
        n = len(group)
        for i in range(n):
            frac = i / max(n - 1, 1)
            if frac <= EARLY_PHASE_END:
                phase_labels.append("early")
            elif frac >= LATE_PHASE_START:
                phase_labels.append("late")
            else:
                phase_labels.append("mid")
    result["session_phase"] = phase_labels

    return result


# ---------------------------------------------------------------------------
# Regime summary aggregation
# ---------------------------------------------------------------------------

def _regime_pnl_summary(
    labeled: pd.DataFrame,
    regime_col: str,
) -> dict[str, dict[str, float]]:
    """Compute mean mtm_pnl change per tick within each regime label."""
    result: dict[str, dict[str, float]] = {}
    for label, group in labeled.groupby(regime_col):
        if label == "unknown":
            continue
        pnl_vals = group["mtm_pnl"].tolist()
        if len(pnl_vals) < 2:
            result[label] = {"mean_pnl_change": 0.0, "tick_count": len(pnl_vals)}
            continue
        changes = [pnl_vals[i] - pnl_vals[i - 1] for i in range(1, len(pnl_vals))]
        result[label] = {
            "mean_pnl_change": statistics.fmean(changes) if changes else 0.0,
            "tick_count": len(pnl_vals),
        }
    return result


def summarize_regimes(
    session_ledgers: dict[int, dict[str, pd.DataFrame]],
) -> dict[str, Any]:
    """Compute regime summary across all sample sessions.

    Returns per-product, per-regime-dimension aggregates.
    """
    all_labeled: list[pd.DataFrame] = []

    for session_id, ledger in session_ledgers.items():
        traces = ledger["traces"]
        prices = ledger["prices"]
        if traces.empty:
            continue
        labeled = label_session_regimes(traces, prices)
        labeled["session_id"] = session_id
        all_labeled.append(labeled)

    if not all_labeled:
        return {
            "by_product": {},
            "provenance": {
                "scope": "sample",
                "session_count": 0,
                "warning": "No sample session data for regime analysis.",
            },
        }

    combined = pd.concat(all_labeled, ignore_index=True)
    by_product: dict[str, dict[str, Any]] = {}

    for product, product_df in combined.groupby("product"):
        by_product[product] = {
            "volatility": _regime_pnl_summary(product_df, "volatility_regime"),
            "spread": _regime_pnl_summary(product_df, "spread_regime"),
            "inventory": _regime_pnl_summary(product_df, "inventory_regime"),
            "phase": _regime_pnl_summary(product_df, "session_phase"),
        }

    return {
        "by_product": by_product,
        "provenance": {
            "scope": "sample",
            "session_count": len(session_ledgers),
            "warning": (
                f"Regime analysis from {len(session_ledgers)} sample sessions. "
                "Regime boundaries may shift with different RNG seeds."
            ),
        },
    }
