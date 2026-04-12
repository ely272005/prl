"""Noise and suspicion detection — flags wins that may not be real.

Checks for:
  - gains tiny relative to run variance
  - mean improves but Sharpe degrades
  - preservation constraints broken
  - gains purely from aggression increase
  - packet quality weakened
  - known bad-family patterns
  - calibration profile suspect
"""
from __future__ import annotations

import math
from typing import Any


# Flag thresholds
MIN_SIGNAL_TO_NOISE = 0.30    # delta_mean / parent_std
MAX_SHARPE_REGRESSION = -2.0  # Sharpe can drop at most 2 points while mean improves
AGGRESSION_GAIN_THRESHOLD = 0.25  # >25% more fills and gain → suspect


SUSPECT_FLAGS = (
    "noise_likely",
    "sharpe_mean_divergence",
    "aggression_driven_gain",
    "preservation_violated",
    "packet_quality_weakened",
    "bad_family_pattern",
    "calibration_suspect",
    "single_product_fragile",
)


def detect_suspicions(
    candidate: dict[str, Any],
    parent: dict[str, Any],
    preservation_audit: dict[str, Any] | None = None,
    attribution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all suspicion checks on a candidate.

    Returns a dict with flags, severity, and explanation.
    """
    cps = candidate.get("packet_short", candidate)
    pps = parent.get("packet_short", parent)

    flags: list[dict[str, Any]] = []

    # 1. Noise check: gain tiny relative to variance
    noise_flag = _check_noise(cps, pps)
    if noise_flag:
        flags.append(noise_flag)

    # 2. Sharpe-mean divergence
    diverge_flag = _check_sharpe_mean_divergence(cps, pps)
    if diverge_flag:
        flags.append(diverge_flag)

    # 3. Aggression-driven gain
    agg_flag = _check_aggression_gain(cps, pps, attribution)
    if agg_flag:
        flags.append(agg_flag)

    # 4. Preservation violated
    if preservation_audit and preservation_audit.get("verdict") == "violated":
        flags.append({
            "flag": "preservation_violated",
            "severity": "high",
            "detail": preservation_audit.get("reason", "Constraint violations detected"),
        })

    # 5. Packet quality weakened
    quality_flag = _check_packet_quality(cps, pps)
    if quality_flag:
        flags.append(quality_flag)

    # 6. Single-product fragility
    fragility_flag = _check_single_product_fragility(cps, pps)
    if fragility_flag:
        flags.append(fragility_flag)

    # 7. Calibration suspect
    cal_flag = _check_calibration_suspect(cps)
    if cal_flag:
        flags.append(cal_flag)

    # Overall suspicion level
    if any(f["severity"] == "high" for f in flags):
        suspicion_level = "high"
    elif any(f["severity"] == "medium" for f in flags):
        suspicion_level = "medium"
    elif flags:
        suspicion_level = "low"
    else:
        suspicion_level = "clean"

    return {
        "suspicion_level": suspicion_level,
        "flag_count": len(flags),
        "flags": flags,
        "is_suspicious": suspicion_level in ("high", "medium"),
    }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_noise(cps: dict, pps: dict) -> dict | None:
    """Flag if gain is tiny relative to run variance."""
    c_mean = _g(cps, "pnl", "mean") or 0
    p_mean = _g(pps, "pnl", "mean") or 0
    p_std = _g(pps, "pnl", "std") or 1

    delta = c_mean - p_mean
    if delta <= 0:
        return None  # no gain to check

    signal_to_noise = delta / p_std if p_std > 0 else 0

    if signal_to_noise < MIN_SIGNAL_TO_NOISE:
        return {
            "flag": "noise_likely",
            "severity": "medium",
            "detail": (
                f"Gain ({delta:.0f}) is small relative to parent std ({p_std:.0f}). "
                f"Signal-to-noise ratio: {signal_to_noise:.3f} < {MIN_SIGNAL_TO_NOISE}."
            ),
            "signal_to_noise": round(signal_to_noise, 4),
        }
    return None


def _check_sharpe_mean_divergence(cps: dict, pps: dict) -> dict | None:
    """Flag if mean improves but Sharpe degrades materially."""
    c_mean = _g(cps, "pnl", "mean") or 0
    p_mean = _g(pps, "pnl", "mean") or 0
    c_sharpe = _g(cps, "pnl", "sharpe_like") or 0
    p_sharpe = _g(pps, "pnl", "sharpe_like") or 0

    mean_improved = c_mean > p_mean
    sharpe_degraded = c_sharpe < p_sharpe + MAX_SHARPE_REGRESSION

    if mean_improved and sharpe_degraded:
        return {
            "flag": "sharpe_mean_divergence",
            "severity": "medium",
            "detail": (
                f"Mean improved ({p_mean:.0f} → {c_mean:.0f}) but Sharpe degraded "
                f"({p_sharpe:.2f} → {c_sharpe:.2f}). "
                f"Gain came at the cost of much higher variance."
            ),
        }
    return None


def _check_aggression_gain(
    cps: dict,
    pps: dict,
    attribution: dict[str, Any] | None,
) -> dict | None:
    """Flag if gain is purely from more aggressive trading."""
    c_mean = _g(cps, "pnl", "mean") or 0
    p_mean = _g(pps, "pnl", "mean") or 0

    if c_mean <= p_mean:
        return None  # no gain

    # Check aggressiveness from attribution
    if attribution:
        for attr in attribution.get("attributions", []):
            if attr.get("mechanism") == "aggressiveness_change":
                rel_change = attr.get("detail", {}).get("relative_change", 0)
                if rel_change > AGGRESSION_GAIN_THRESHOLD:
                    return {
                        "flag": "aggression_driven_gain",
                        "severity": "medium",
                        "detail": (
                            f"Gain appears driven by {rel_change:.0%} more fills. "
                            f"This may not represent real alpha."
                        ),
                    }

    # Fallback: check fill counts directly
    c_fills = (_g(cps, "fill_quality", "taker_fill_count") or 0) + \
              (_g(cps, "fill_quality", "maker_fill_count") or 0)
    p_fills = (_g(pps, "fill_quality", "taker_fill_count") or 0) + \
              (_g(pps, "fill_quality", "maker_fill_count") or 0)

    if p_fills > 0 and c_fills > p_fills * (1 + AGGRESSION_GAIN_THRESHOLD):
        c_eff = _g(cps, "efficiency", "pnl_per_fill") or 0
        p_eff = _g(pps, "efficiency", "pnl_per_fill") or 0
        if c_eff < p_eff:
            return {
                "flag": "aggression_driven_gain",
                "severity": "medium",
                "detail": (
                    f"More fills ({p_fills} → {c_fills}) with lower efficiency "
                    f"({p_eff:.2f} → {c_eff:.2f}). "
                    f"Gain is from volume, not quality."
                ),
            }
    return None


def _check_packet_quality(cps: dict, pps: dict) -> dict | None:
    """Flag if packet quality indicators weakened."""
    c_pos_rate = _g(cps, "pnl", "positive_rate") or 0
    p_pos_rate = _g(pps, "pnl", "positive_rate") or 0
    c_p05 = _g(cps, "pnl", "p05") or 0
    p_p05 = _g(pps, "pnl", "p05") or 0
    c_mean = _g(cps, "pnl", "mean") or 0
    p_mean = _g(pps, "pnl", "mean") or 0

    issues = []

    if c_pos_rate < p_pos_rate - 0.10:
        issues.append(f"positive rate dropped ({p_pos_rate:.0%} → {c_pos_rate:.0%})")

    if c_p05 < p_p05 - 500 and c_mean > p_mean:
        issues.append(f"tail risk worsened: p05 dropped from {p_p05:.0f} to {c_p05:.0f}")

    if not issues:
        return None

    return {
        "flag": "packet_quality_weakened",
        "severity": "low",
        "detail": "; ".join(issues),
    }


def _check_single_product_fragility(cps: dict, pps: dict) -> dict | None:
    """Flag if all gains came from a single product while the other regressed."""
    c_em = _g(cps, "per_product", "emerald", "mean") or 0
    p_em = _g(pps, "per_product", "emerald", "mean") or 0
    c_tom = _g(cps, "per_product", "tomato", "mean") or 0
    p_tom = _g(pps, "per_product", "tomato", "mean") or 0

    em_delta = c_em - p_em
    tom_delta = c_tom - p_tom
    total_delta = em_delta + tom_delta

    if total_delta <= 0:
        return None

    # One product improved while the other regressed
    if em_delta > 0 and tom_delta < 0 and abs(tom_delta) > total_delta * 0.2:
        return {
            "flag": "single_product_fragile",
            "severity": "low",
            "detail": (
                f"Gain entirely from EMERALDS ({em_delta:+.0f}) while "
                f"TOMATOES regressed ({tom_delta:+.0f}). Fragile."
            ),
        }
    if tom_delta > 0 and em_delta < 0 and abs(em_delta) > total_delta * 0.2:
        return {
            "flag": "single_product_fragile",
            "severity": "low",
            "detail": (
                f"Gain entirely from TOMATOES ({tom_delta:+.0f}) while "
                f"EMERALDS regressed ({em_delta:+.0f}). Fragile."
            ),
        }
    return None


def _check_calibration_suspect(cps: dict) -> dict | None:
    """Flag if confidence is low or warnings suggest calibration issues."""
    confidence = cps.get("confidence", "LOW")
    warnings = cps.get("warnings", [])

    issues = []
    if confidence == "LOW":
        issues.append("LOW confidence — too few sessions for reliable comparison")

    cal_warnings = [w for w in warnings if "calibrat" in w.lower() or "drift" in w.lower()]
    if cal_warnings:
        issues.extend(cal_warnings)

    if not issues:
        return None

    return {
        "flag": "calibration_suspect",
        "severity": "medium" if confidence == "LOW" else "low",
        "detail": "; ".join(issues),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _g(d: dict, *keys: str) -> float | None:
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
