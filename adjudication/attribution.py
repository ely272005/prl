"""Mechanism attribution — explains where gains or losses came from.

Does not attempt to be magical. Uses disciplined, explicit comparisons
between candidate and parent packet metrics to attribute changes to
specific mechanisms: product shifts, role shifts, aggressiveness changes,
quality changes, risk changes.
"""
from __future__ import annotations

import math
from typing import Any


def attribute_mechanism(
    candidate: dict[str, Any],
    parent: dict[str, Any],
) -> dict[str, Any]:
    """Attribute the delta between candidate and parent to specific mechanisms.

    Returns a dict with:
      - attributions: list of mechanism explanations
      - dominant_mechanism: the biggest driver
      - product_attribution: which product drove the change
      - role_attribution: maker vs taker shift
      - quality_attribution: fill quality change
      - risk_attribution: risk profile change
    """
    cps = candidate.get("packet_short", candidate)
    pps = parent.get("packet_short", parent)

    attributions: list[dict[str, Any]] = []

    # 1. Product attribution
    prod_attr = _attribute_product(cps, pps)
    if prod_attr:
        attributions.append(prod_attr)

    # 2. Role shift (maker vs taker)
    role_attr = _attribute_role_shift(cps, pps)
    if role_attr:
        attributions.append(role_attr)

    # 3. Aggressiveness (fill count change)
    agg_attr = _attribute_aggressiveness(cps, pps)
    if agg_attr:
        attributions.append(agg_attr)

    # 4. Fill quality change
    quality_attr = _attribute_fill_quality(cps, pps)
    if quality_attr:
        attributions.append(quality_attr)

    # 5. Risk profile change (std, drawdown)
    risk_attr = _attribute_risk_change(cps, pps)
    if risk_attr:
        attributions.append(risk_attr)

    # 6. Sharpe decomposition (mean vs std)
    sharpe_attr = _attribute_sharpe(cps, pps)
    if sharpe_attr:
        attributions.append(sharpe_attr)

    # Determine dominant mechanism
    dominant = None
    if attributions:
        dominant = max(attributions, key=lambda a: abs(a.get("magnitude", 0)))

    return {
        "attributions": attributions,
        "dominant_mechanism": dominant["mechanism"] if dominant else "no_clear_driver",
        "product_attribution": prod_attr,
        "role_attribution": role_attr,
        "quality_attribution": quality_attr,
        "risk_attribution": risk_attr,
        "summary": _build_summary(attributions, cps, pps),
    }


# ---------------------------------------------------------------------------
# Individual attribution functions
# ---------------------------------------------------------------------------

def _attribute_product(cps: dict, pps: dict) -> dict[str, Any] | None:
    """Which product drove the PnL change?"""
    c_em = _g(cps, "per_product", "emerald", "mean") or 0
    p_em = _g(pps, "per_product", "emerald", "mean") or 0
    c_tom = _g(cps, "per_product", "tomato", "mean") or 0
    p_tom = _g(pps, "per_product", "tomato", "mean") or 0

    em_delta = c_em - p_em
    tom_delta = c_tom - p_tom
    total_delta = em_delta + tom_delta

    if abs(total_delta) < 10:  # negligible
        return None

    if abs(total_delta) > 0:
        em_share = em_delta / total_delta if total_delta != 0 else 0.5
        tom_share = tom_delta / total_delta if total_delta != 0 else 0.5
    else:
        em_share = 0.5
        tom_share = 0.5

    if abs(em_share) > 0.8:
        source = "EMERALDS"
    elif abs(tom_share) > 0.8:
        source = "TOMATOES"
    else:
        source = "both products"

    return {
        "mechanism": "product_shift",
        "magnitude": abs(total_delta),
        "detail": {
            "emerald_delta": round(em_delta, 1),
            "tomato_delta": round(tom_delta, 1),
            "emerald_share": round(em_share, 3),
            "tomato_share": round(tom_share, 3),
            "dominant_product": source,
        },
        "description": (
            f"PnL change ({total_delta:+.0f}) came from: "
            f"EMERALDS {em_delta:+.0f} ({em_share:.0%}), "
            f"TOMATOES {tom_delta:+.0f} ({tom_share:.0%})."
        ),
    }


def _attribute_role_shift(cps: dict, pps: dict) -> dict[str, Any] | None:
    """Did the maker/taker balance shift?"""
    c_passive = _g(cps, "fill_quality", "passive_fill_rate")
    p_passive = _g(pps, "fill_quality", "passive_fill_rate")

    if c_passive is None or p_passive is None:
        return None

    shift = c_passive - p_passive
    if abs(shift) < 0.02:
        return None

    direction = "more passive/maker" if shift > 0 else "more active/taker"

    return {
        "mechanism": "role_shift",
        "magnitude": abs(shift) * 100,  # in percentage points
        "detail": {
            "parent_passive_rate": round(p_passive, 4),
            "candidate_passive_rate": round(c_passive, 4),
            "shift": round(shift, 4),
            "direction": direction,
        },
        "description": (
            f"Strategy became {direction}: "
            f"passive fill rate {p_passive:.3f} → {c_passive:.3f} ({shift:+.3f})."
        ),
    }


def _attribute_aggressiveness(cps: dict, pps: dict) -> dict[str, Any] | None:
    """Did overall fill count change significantly?"""
    c_taker = _g(cps, "fill_quality", "taker_fill_count") or 0
    c_maker = _g(cps, "fill_quality", "maker_fill_count") or 0
    p_taker = _g(pps, "fill_quality", "taker_fill_count") or 0
    p_maker = _g(pps, "fill_quality", "maker_fill_count") or 0

    c_total = c_taker + c_maker
    p_total = p_taker + p_maker

    if p_total == 0:
        return None

    rel_change = (c_total - p_total) / p_total

    if abs(rel_change) < 0.10:  # less than 10% change
        return None

    direction = "more aggressive" if rel_change > 0 else "less aggressive"

    return {
        "mechanism": "aggressiveness_change",
        "magnitude": abs(rel_change) * 100,
        "detail": {
            "parent_fills": p_total,
            "candidate_fills": c_total,
            "relative_change": round(rel_change, 4),
            "taker_delta": c_taker - p_taker,
            "maker_delta": c_maker - p_maker,
            "direction": direction,
        },
        "description": (
            f"Strategy is {direction}: "
            f"total fills {p_total} → {c_total} ({rel_change:+.0%}). "
            f"Taker fills {'+' if c_taker >= p_taker else ''}{c_taker - p_taker}, "
            f"maker fills {'+' if c_maker >= p_maker else ''}{c_maker - p_maker}."
        ),
    }


def _attribute_fill_quality(cps: dict, pps: dict) -> dict[str, Any] | None:
    """Did fill quality change?"""
    c_em_fvf = _g(cps, "fill_quality", "mean_fill_vs_fair_emerald")
    p_em_fvf = _g(pps, "fill_quality", "mean_fill_vs_fair_emerald")
    c_tom_fvf = _g(cps, "fill_quality", "mean_fill_vs_fair_tomato")
    p_tom_fvf = _g(pps, "fill_quality", "mean_fill_vs_fair_tomato")

    changes = {}
    magnitude = 0

    if c_em_fvf is not None and p_em_fvf is not None:
        d = c_em_fvf - p_em_fvf
        changes["emerald_fill_vs_fair_delta"] = round(d, 4)
        magnitude = max(magnitude, abs(d))

    if c_tom_fvf is not None and p_tom_fvf is not None:
        d = c_tom_fvf - p_tom_fvf
        changes["tomato_fill_vs_fair_delta"] = round(d, 4)
        magnitude = max(magnitude, abs(d))

    if magnitude < 0.1:
        return None

    return {
        "mechanism": "fill_quality_change",
        "magnitude": magnitude,
        "detail": changes,
        "description": (
            f"Fill quality shifted: "
            + ", ".join(f"{k}={v:+.2f}" for k, v in changes.items())
            + "."
        ),
    }


def _attribute_risk_change(cps: dict, pps: dict) -> dict[str, Any] | None:
    """Did the risk profile change?"""
    c_std = _g(cps, "pnl", "std") or 0
    p_std = _g(pps, "pnl", "std") or 0
    c_dd = _g(cps, "drawdown", "mean_max_drawdown") or 0
    p_dd = _g(pps, "drawdown", "mean_max_drawdown") or 0

    std_delta = c_std - p_std
    dd_delta = c_dd - p_dd  # more negative = worse

    magnitude = max(abs(std_delta), abs(dd_delta))
    if magnitude < 50:  # negligible
        return None

    parts = []
    if abs(std_delta) > 50:
        direction = "higher" if std_delta > 0 else "lower"
        parts.append(f"Std {direction} by {abs(std_delta):.0f}")
    if abs(dd_delta) > 50:
        direction = "worse" if dd_delta < 0 else "better"
        parts.append(f"Drawdown {direction} by {abs(dd_delta):.0f}")

    return {
        "mechanism": "risk_change",
        "magnitude": magnitude,
        "detail": {
            "std_delta": round(std_delta, 1),
            "drawdown_delta": round(dd_delta, 1),
            "parent_std": round(p_std, 1),
            "candidate_std": round(c_std, 1),
        },
        "description": ". ".join(parts) + "." if parts else "Risk profile unchanged.",
    }


def _attribute_sharpe(cps: dict, pps: dict) -> dict[str, Any] | None:
    """Decompose Sharpe change into mean vs std contributions."""
    c_mean = _g(cps, "pnl", "mean") or 0
    p_mean = _g(pps, "pnl", "mean") or 0
    c_std = _g(cps, "pnl", "std") or 1
    p_std = _g(pps, "pnl", "std") or 1
    c_sharpe = _g(cps, "pnl", "sharpe_like") or 0
    p_sharpe = _g(pps, "pnl", "sharpe_like") or 0

    sharpe_delta = c_sharpe - p_sharpe
    if abs(sharpe_delta) < 0.5:
        return None

    # Decompose: Sharpe = mean / std
    # Delta Sharpe ≈ (delta_mean / p_std) - (p_mean * delta_std / p_std^2)
    mean_contribution = (c_mean - p_mean) / p_std if p_std > 0 else 0
    std_contribution = -p_mean * (c_std - p_std) / (p_std ** 2) if p_std > 0 else 0

    if abs(mean_contribution) > abs(std_contribution):
        driver = "higher mean" if mean_contribution > 0 else "lower mean"
    else:
        driver = "lower volatility" if std_contribution > 0 else "higher volatility"

    return {
        "mechanism": "sharpe_decomposition",
        "magnitude": abs(sharpe_delta),
        "detail": {
            "sharpe_delta": round(sharpe_delta, 3),
            "mean_contribution": round(mean_contribution, 3),
            "std_contribution": round(std_contribution, 3),
            "driver": driver,
        },
        "description": (
            f"Sharpe changed by {sharpe_delta:+.2f}: "
            f"driven by {driver} "
            f"(mean contribution={mean_contribution:+.2f}, "
            f"std contribution={std_contribution:+.2f})."
        ),
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _build_summary(
    attributions: list[dict[str, Any]],
    cps: dict,
    pps: dict,
) -> str:
    """Build a plain-English attribution summary."""
    if not attributions:
        return "No significant mechanism changes detected."

    parts = []
    for attr in sorted(attributions, key=lambda a: -abs(a.get("magnitude", 0))):
        parts.append(attr["description"])

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _g(d: dict, *keys: str) -> float | None:
    """Get nested value, returning None for missing/NaN/inf."""
    obj = d
    for k in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(k)
    if obj is None:
        return None
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return float(obj)
    return None
