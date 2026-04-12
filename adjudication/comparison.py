"""Parent and frontier comparison engine.

Computes structured deltas between a candidate and:
  1. its exact parent,
  2. the best in its family,
  3. the current frontier set,
  4. calibration anchors.

Every comparison is explicit: metric, candidate value, reference value,
delta, relative change, and a plain-English summary.
"""
from __future__ import annotations

import math
from typing import Any


# Metrics to compare, with (key_path, higher_is_better, format)
COMPARISON_METRICS = [
    ("pnl_mean",           ("pnl", "mean"),            True,  ".0f"),
    ("pnl_std",            ("pnl", "std"),              False, ".0f"),
    ("sharpe",             ("pnl", "sharpe_like"),      True,  ".2f"),
    ("p05",                ("pnl", "p05"),              True,  ".0f"),
    ("p50",                ("pnl", "p50"),              True,  ".0f"),
    ("p95",                ("pnl", "p95"),              True,  ".0f"),
    ("positive_rate",      ("pnl", "positive_rate"),    True,  ".1%"),
    ("emerald_mean",       ("per_product", "emerald", "mean"), True,  ".0f"),
    ("tomato_mean",        ("per_product", "tomato", "mean"),  True,  ".0f"),
    ("emerald_sharpe",     ("per_product", "emerald", "sharpe_like"), True, ".2f"),
    ("tomato_sharpe",      ("per_product", "tomato", "sharpe_like"),  True, ".2f"),
    ("passive_fill_rate",  ("fill_quality", "passive_fill_rate"), True, ".3f"),
    ("pnl_per_fill",       ("efficiency", "pnl_per_fill"), True, ".2f"),
    ("mean_max_drawdown",  ("drawdown", "mean_max_drawdown"), False, ".0f"),
]


def _extract(packet: dict[str, Any], key_path: tuple[str, ...]) -> float | None:
    """Extract a value from a nested packet dict."""
    ps = packet.get("packet_short", packet)
    obj = ps
    for k in key_path:
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


def compare_pair(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    reference_label: str = "parent",
) -> dict[str, Any]:
    """Compare a candidate packet against a reference packet.

    Returns a dict with per-metric deltas and a summary.
    """
    deltas = []
    improvements = 0
    regressions = 0

    for name, key_path, higher_better, fmt in COMPARISON_METRICS:
        cval = _extract(candidate, key_path)
        rval = _extract(reference, key_path)

        if cval is None or rval is None:
            deltas.append({
                "metric": name,
                "candidate": cval,
                "reference": rval,
                "delta": None,
                "relative": None,
                "improved": None,
            })
            continue

        delta = cval - rval
        relative = delta / abs(rval) if rval != 0 else (1.0 if delta > 0 else -1.0 if delta < 0 else 0.0)

        improved = (delta > 0) if higher_better else (delta < 0)
        if abs(delta) < 1e-9:
            improved = None  # no change

        if improved is True:
            improvements += 1
        elif improved is False:
            regressions += 1

        deltas.append({
            "metric": name,
            "candidate": round(cval, 6),
            "reference": round(rval, 6),
            "delta": round(delta, 6),
            "relative": round(relative, 6),
            "improved": improved,
        })

    return {
        "reference_label": reference_label,
        "reference_id": _get_id(reference),
        "candidate_id": _get_id(candidate),
        "deltas": deltas,
        "improvements": improvements,
        "regressions": regressions,
        "net_direction": _net_direction(improvements, regressions),
    }


def compare_to_frontier(
    candidate: dict[str, Any],
    frontier: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare candidate to the best values across the frontier set.

    Returns per-metric comparison against the frontier best.
    """
    if not frontier:
        return {
            "reference_label": "frontier",
            "frontier_size": 0,
            "deltas": [],
            "beats_frontier_on": [],
            "below_frontier_on": [],
        }

    beats = []
    below = []
    deltas = []

    for name, key_path, higher_better, fmt in COMPARISON_METRICS:
        cval = _extract(candidate, key_path)
        frontier_vals = [_extract(p, key_path) for p in frontier]
        frontier_vals = [v for v in frontier_vals if v is not None]

        if cval is None or not frontier_vals:
            deltas.append({"metric": name, "candidate": cval, "frontier_best": None, "delta": None})
            continue

        if higher_better:
            frontier_best = max(frontier_vals)
            improved = cval > frontier_best
        else:
            frontier_best = min(frontier_vals)
            improved = cval < frontier_best

        delta = cval - frontier_best
        if improved:
            beats.append(name)
        elif abs(delta) > 1e-9:
            below.append(name)

        deltas.append({
            "metric": name,
            "candidate": round(cval, 6),
            "frontier_best": round(frontier_best, 6),
            "delta": round(delta, 6),
            "improved": improved,
        })

    return {
        "reference_label": "frontier",
        "frontier_size": len(frontier),
        "deltas": deltas,
        "beats_frontier_on": beats,
        "below_frontier_on": below,
    }


def compare_to_family(
    candidate: dict[str, Any],
    family_peers: list[dict[str, Any]],
    family_name: str,
) -> dict[str, Any]:
    """Compare candidate to family peers.

    Returns comparison against the best in the family.
    """
    if not family_peers:
        return {
            "reference_label": f"family:{family_name}",
            "family_size": 0,
            "deltas": [],
        }

    # Find best peer by Sharpe
    best_peer = max(
        family_peers,
        key=lambda p: _extract(p, ("pnl", "sharpe_like")) or 0,
    )
    result = compare_pair(candidate, best_peer, reference_label=f"best_in_{family_name}")
    result["family_size"] = len(family_peers)
    return result


def summarize_comparison(
    vs_parent: dict[str, Any],
    vs_frontier: dict[str, Any],
    vs_family: dict[str, Any] | None = None,
) -> str:
    """Build a one-paragraph English summary of all comparisons."""
    parts = []

    # vs parent
    parent_pnl = _find_delta(vs_parent, "pnl_mean")
    parent_sharpe = _find_delta(vs_parent, "sharpe")
    if parent_pnl and parent_sharpe:
        pnl_d = parent_pnl["delta"] or 0
        sharpe_d = parent_sharpe["delta"] or 0
        direction = "improves on" if pnl_d > 0 and sharpe_d > 0 else \
                    "regresses from" if pnl_d < 0 and sharpe_d < 0 else \
                    "is mixed vs"
        parts.append(
            f"Candidate {direction} parent: "
            f"PnL {'+' if pnl_d >= 0 else ''}{pnl_d:.0f}, "
            f"Sharpe {'+' if sharpe_d >= 0 else ''}{sharpe_d:.2f}."
        )

    # vs frontier
    if vs_frontier.get("beats_frontier_on"):
        parts.append(
            f"Beats frontier on: {', '.join(vs_frontier['beats_frontier_on'])}."
        )
    elif vs_frontier.get("frontier_size", 0) > 0:
        parts.append("Does not beat frontier on any metric.")

    # vs family
    if vs_family and vs_family.get("deltas"):
        family_pnl = _find_delta(vs_family, "pnl_mean")
        if family_pnl and family_pnl["delta"] is not None:
            d = family_pnl["delta"]
            label = vs_family.get("reference_label", "family")
            parts.append(
                f"vs {label}: PnL {'+' if d >= 0 else ''}{d:.0f}."
            )

    return " ".join(parts) if parts else "Insufficient data for comparison."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_id(packet: dict[str, Any]) -> str:
    ps = packet.get("packet_short", packet)
    return (
        packet.get("_case_id")
        or packet.get("case_id")
        or ps.get("candidate_id", "?")
    )


def _net_direction(improvements: int, regressions: int) -> str:
    if improvements > regressions + 2:
        return "strong_improvement"
    if improvements > regressions:
        return "mild_improvement"
    if regressions > improvements + 2:
        return "strong_regression"
    if regressions > improvements:
        return "mild_regression"
    return "mixed"


def _find_delta(comparison: dict[str, Any], metric_name: str) -> dict | None:
    for d in comparison.get("deltas", []):
        if d["metric"] == metric_name:
            return d
    return None
