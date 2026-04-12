"""Preservation-constraint audit — checks whether a candidate respected task rules.

Takes a task's preservation/allowed/forbidden lists and the candidate's packet,
compares against the parent's packet, and flags violations.

This is a data-level audit: it cannot inspect source code changes, but it can
detect whether the outputs violate the intent of the constraints.
"""
from __future__ import annotations

import math
from typing import Any


# Thresholds for "unchanged" — within noise tolerance
_PNL_UNCHANGED_THRESHOLD = 0.05       # 5% relative change considered "unchanged"
_FILL_RATE_UNCHANGED_THRESHOLD = 0.10  # 10% change in fill metrics
_POSITION_UNCHANGED_THRESHOLD = 0.15   # 15% change in position metrics


def audit_preservation(
    task: dict[str, Any],
    candidate: dict[str, Any],
    parent: dict[str, Any],
) -> dict[str, Any]:
    """Audit whether a candidate respected its task constraints.

    Returns a dict with violations, warnings, and a clean/dirty verdict.
    """
    violations: list[dict[str, Any]] = []
    warnings: list[str] = []

    product_scope = task.get("product_scope", [])
    preservation = task.get("preservation", [])
    forbidden = task.get("forbidden_changes", [])
    task_type = task.get("task_type", "")

    cps = candidate.get("packet_short", candidate)
    pps = parent.get("packet_short", parent)

    # 1. Product-scope violations: did non-target products change?
    violations.extend(_check_product_preservation(product_scope, cps, pps))

    # 2. Risk-control violations
    violations.extend(_check_risk_controls(preservation, cps, pps))

    # 3. Maker-structure violations
    if _constraint_mentions(preservation, "maker"):
        violations.extend(_check_maker_structure(cps, pps))

    # 4. Aggressiveness violations
    if _constraint_mentions(forbidden, "aggressiveness") or _constraint_mentions(preservation, "aggressiveness"):
        violations.extend(_check_aggressiveness(cps, pps))

    # 5. Calibration check: should be identical to parent
    if task_type == "calibration_check":
        violations.extend(_check_calibration(cps, pps))

    # 6. Near-parent control: should be nearly identical
    if task_type == "near_parent_control":
        violations.extend(_check_near_parent(cps, pps))

    # Build overall verdict
    critical_violations = [v for v in violations if v["severity"] == "critical"]
    moderate_violations = [v for v in violations if v["severity"] == "moderate"]

    if critical_violations:
        verdict = "violated"
        reason = f"{len(critical_violations)} critical constraint violation(s)"
    elif moderate_violations:
        verdict = "suspect"
        reason = f"{len(moderate_violations)} moderate constraint issue(s)"
    else:
        verdict = "clean"
        reason = "No constraint violations detected"

    return {
        "verdict": verdict,
        "reason": reason,
        "violations": violations,
        "warnings": warnings,
        "critical_count": len(critical_violations),
        "moderate_count": len(moderate_violations),
    }


# ---------------------------------------------------------------------------
# Specific checks
# ---------------------------------------------------------------------------

def _check_product_preservation(
    product_scope: list[str],
    candidate: dict[str, Any],
    parent: dict[str, Any],
) -> list[dict[str, Any]]:
    """Check that non-target products were not materially changed."""
    violations = []
    preserved_products = []

    if product_scope == ["EMERALDS"]:
        preserved_products = [("tomato", "TOMATOES")]
    elif product_scope == ["TOMATOES"]:
        preserved_products = [("emerald", "EMERALDS")]
    # If both products are in scope, nothing to check

    for key, label in preserved_products:
        c_mean = _get_nested(candidate, "per_product", key, "mean")
        p_mean = _get_nested(parent, "per_product", key, "mean")

        if c_mean is not None and p_mean is not None and p_mean != 0:
            rel_change = abs(c_mean - p_mean) / abs(p_mean)
            if rel_change > _PNL_UNCHANGED_THRESHOLD:
                violations.append({
                    "constraint": f"Preserve {label}",
                    "severity": "critical" if rel_change > 0.15 else "moderate",
                    "detail": (
                        f"{label} PnL changed by {rel_change:.1%}: "
                        f"parent={p_mean:.0f}, candidate={c_mean:.0f}"
                    ),
                    "metric": f"{key}_mean",
                    "parent_value": p_mean,
                    "candidate_value": c_mean,
                    "relative_change": round(rel_change, 4),
                })

    return violations


def _check_risk_controls(
    preservation: list[str],
    candidate: dict[str, Any],
    parent: dict[str, Any],
) -> list[dict[str, Any]]:
    """Check that risk controls were not weakened."""
    violations = []

    if not _constraint_mentions(preservation, "position limit"):
        return violations

    # Check drawdown worsened significantly
    c_dd = _get_nested(candidate, "drawdown", "mean_max_drawdown")
    p_dd = _get_nested(parent, "drawdown", "mean_max_drawdown")

    if c_dd is not None and p_dd is not None:
        # Drawdowns are negative; more negative = worse
        if p_dd < 0 and c_dd < p_dd * 1.5:  # 50% worse drawdown
            violations.append({
                "constraint": "Preserve risk controls",
                "severity": "moderate",
                "detail": (
                    f"Drawdown worsened significantly: "
                    f"parent={p_dd:.0f}, candidate={c_dd:.0f}"
                ),
                "metric": "mean_max_drawdown",
                "parent_value": p_dd,
                "candidate_value": c_dd,
                "relative_change": round(abs(c_dd - p_dd) / abs(p_dd), 4) if p_dd != 0 else None,
            })

    return violations


def _check_maker_structure(
    candidate: dict[str, Any],
    parent: dict[str, Any],
) -> list[dict[str, Any]]:
    """Check that maker-heavy structure was preserved."""
    violations = []

    c_passive = _get_nested(candidate, "fill_quality", "passive_fill_rate")
    p_passive = _get_nested(parent, "fill_quality", "passive_fill_rate")

    if c_passive is not None and p_passive is not None and p_passive > 0:
        drop = p_passive - c_passive
        if drop > _FILL_RATE_UNCHANGED_THRESHOLD:
            violations.append({
                "constraint": "Preserve maker structure",
                "severity": "critical" if drop > 0.2 else "moderate",
                "detail": (
                    f"Passive fill rate dropped from {p_passive:.3f} to {c_passive:.3f} "
                    f"(drop={drop:.3f})"
                ),
                "metric": "passive_fill_rate",
                "parent_value": p_passive,
                "candidate_value": c_passive,
                "relative_change": round(drop / p_passive, 4),
            })

    return violations


def _check_aggressiveness(
    candidate: dict[str, Any],
    parent: dict[str, Any],
) -> list[dict[str, Any]]:
    """Check that overall aggressiveness did not increase globally."""
    violations = []

    # Total fill count as aggressiveness proxy
    c_fills = (
        _get_nested(candidate, "fill_quality", "taker_fill_count") or 0
    ) + (
        _get_nested(candidate, "fill_quality", "maker_fill_count") or 0
    )
    p_fills = (
        _get_nested(parent, "fill_quality", "taker_fill_count") or 0
    ) + (
        _get_nested(parent, "fill_quality", "maker_fill_count") or 0
    )

    if p_fills > 0 and c_fills > p_fills * 1.3:  # 30% more fills
        violations.append({
            "constraint": "Do not increase aggressiveness globally",
            "severity": "moderate",
            "detail": (
                f"Total fills increased by {(c_fills - p_fills) / p_fills:.0%}: "
                f"parent={p_fills}, candidate={c_fills}"
            ),
            "metric": "total_fills",
            "parent_value": p_fills,
            "candidate_value": c_fills,
            "relative_change": round((c_fills - p_fills) / p_fills, 4),
        })

    return violations


def _check_calibration(
    candidate: dict[str, Any],
    parent: dict[str, Any],
) -> list[dict[str, Any]]:
    """Calibration check: candidate should reproduce parent exactly."""
    violations = []
    c_mean = _get_nested(candidate, "pnl", "mean") or 0
    p_mean = _get_nested(parent, "pnl", "mean") or 0

    if p_mean != 0:
        rel_change = abs(c_mean - p_mean) / abs(p_mean)
        if rel_change > 0.10:  # More than 10% drift
            violations.append({
                "constraint": "Calibration: reproduce parent",
                "severity": "critical",
                "detail": (
                    f"Calibration drift: parent={p_mean:.0f}, "
                    f"candidate={c_mean:.0f} ({rel_change:.1%} change)"
                ),
                "metric": "pnl_mean",
                "parent_value": p_mean,
                "candidate_value": c_mean,
                "relative_change": round(rel_change, 4),
            })

    return violations


def _check_near_parent(
    candidate: dict[str, Any],
    parent: dict[str, Any],
) -> list[dict[str, Any]]:
    """Near-parent control: should be within 2% of parent."""
    violations = []
    c_mean = _get_nested(candidate, "pnl", "mean") or 0
    p_mean = _get_nested(parent, "pnl", "mean") or 0

    if p_mean != 0:
        rel_change = abs(c_mean - p_mean) / abs(p_mean)
        if rel_change > 0.05:  # More than 5% is suspicious for a control
            violations.append({
                "constraint": "Near-parent: minimal functional change",
                "severity": "moderate" if rel_change < 0.10 else "critical",
                "detail": (
                    f"Near-parent control deviated: parent={p_mean:.0f}, "
                    f"candidate={c_mean:.0f} ({rel_change:.1%} change). "
                    f"High noise floor or functional change leaked in."
                ),
                "metric": "pnl_mean",
                "parent_value": p_mean,
                "candidate_value": c_mean,
                "relative_change": round(rel_change, 4),
            })

    return violations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_nested(d: dict[str, Any], *keys: str) -> float | None:
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


def _constraint_mentions(constraints: list[str], keyword: str) -> bool:
    kw = keyword.lower()
    return any(kw in c.lower() for c in constraints)
