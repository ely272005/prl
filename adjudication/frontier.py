"""Frontier update logic — manages the set of best candidates by role.

Roles:
  - top_mean: highest mean PnL
  - top_sharpe: highest Sharpe-like
  - best_calibrated: best packet quality (highest positive rate + lowest drawdown ratio)
  - best_maker_heavy: best among maker-heavy family
  - best_active_tomatoes: best tomato-focused or high-tomato contributor
  - best_control_anchor: most reliable calibration control

Rules:
  - add challengers that beat role-specific thresholds
  - retire dominated candidates (worse on all key metrics)
  - preserve calibration anchors (never remove without replacement)
  - maintain diversity: max 2 candidates per family in frontier
"""
from __future__ import annotations

import math
from typing import Any

from adjudication.comparison import _extract


FRONTIER_ROLES = (
    "top_mean",
    "top_sharpe",
    "best_calibrated",
    "best_maker_heavy",
    "best_active_tomatoes",
    "best_control_anchor",
)

MAX_PER_FAMILY = 2
MAX_FRONTIER_SIZE = 12


def compute_frontier_updates(
    current_frontier: list[dict[str, Any]],
    candidate_verdicts: list[dict[str, Any]],
    all_packets: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute frontier updates based on adjudication verdicts.

    Parameters
    ----------
    current_frontier : list[dict]
        Current frontier packets (with _case_id/_family or packet_short).
    candidate_verdicts : list[dict]
        Verdicts from adjudicate_candidate().
    all_packets : list[dict]
        All available packets for dominated-check.

    Returns
    -------
    dict with:
        - additions: candidates to add
        - retirements: candidates to remove
        - role_assignments: role → candidate mapping
        - frontier_after: the updated frontier
        - changes_summary: human-readable summary
    """
    additions: list[dict[str, Any]] = []
    retirements: list[dict[str, Any]] = []
    reasons: list[str] = []

    # Identify challengers
    challengers = [
        v for v in candidate_verdicts
        if v.get("verdict") in ("frontier_challenger", "escalate")
        and not v.get("suspicion", {}).get("is_suspicious", False)
    ]

    # Build working frontier (copy)
    frontier = list(current_frontier)

    for challenger in challengers:
        cid = challenger.get("candidate_id", "?")
        c_packet = _find_packet(cid, all_packets)
        if c_packet is None:
            continue

        # Check if challenger dominates any frontier member
        dominated = _find_dominated(c_packet, frontier)
        for dom in dominated:
            dom_id = _get_id(dom)
            # Don't retire the only calibration anchor
            if _is_sole_calibration_anchor(dom, frontier):
                reasons.append(f"Kept {dom_id} as sole calibration anchor despite being dominated.")
                continue

            retirements.append({
                "candidate_id": dom_id,
                "reason": f"Dominated by {cid} on all key metrics",
                "family": dom.get("_family", "unknown"),
            })
            frontier = [f for f in frontier if _get_id(f) != dom_id]

        # Check family diversity constraint
        c_family = challenger.get("family", "unknown")
        family_count = sum(1 for f in frontier if _get_family(f) == c_family)

        if family_count >= MAX_PER_FAMILY:
            # Replace worst in family if challenger is better
            worst = _find_worst_in_family(c_family, frontier)
            if worst and _is_better_overall(c_packet, worst):
                worst_id = _get_id(worst)
                retirements.append({
                    "candidate_id": worst_id,
                    "reason": f"Replaced by better {c_family} candidate {cid}",
                    "family": c_family,
                })
                frontier = [f for f in frontier if _get_id(f) != worst_id]
            else:
                reasons.append(f"Skipped {cid}: family {c_family} already at max ({MAX_PER_FAMILY}).")
                continue

        # Check frontier size
        if len(frontier) >= MAX_FRONTIER_SIZE:
            worst = _find_weakest_overall(frontier)
            if worst and _is_better_overall(c_packet, worst):
                worst_id = _get_id(worst)
                retirements.append({
                    "candidate_id": worst_id,
                    "reason": f"Frontier full; replaced by stronger {cid}",
                    "family": _get_family(worst),
                })
                frontier = [f for f in frontier if _get_id(f) != worst_id]
            else:
                reasons.append(f"Skipped {cid}: frontier full and not stronger than weakest.")
                continue

        additions.append({
            "candidate_id": cid,
            "verdict": challenger.get("verdict"),
            "pnl_mean": challenger.get("pnl_mean"),
            "sharpe": challenger.get("sharpe"),
            "family": c_family,
            "reason": challenger.get("reason", ""),
        })
        frontier.append(c_packet)

    # Assign roles
    role_assignments = _assign_roles(frontier)

    return {
        "additions": additions,
        "retirements": retirements,
        "role_assignments": role_assignments,
        "frontier_size_before": len(current_frontier),
        "frontier_size_after": len(frontier),
        "frontier_after": [_get_id(f) for f in frontier],
        "changes_summary": _build_summary(additions, retirements, reasons),
        "notes": reasons,
    }


# ---------------------------------------------------------------------------
# Role assignment
# ---------------------------------------------------------------------------

def _assign_roles(frontier: list[dict[str, Any]]) -> dict[str, str]:
    """Assign frontier roles to candidates."""
    roles: dict[str, str] = {}

    if not frontier:
        return roles

    # top_mean
    best_mean = max(frontier, key=lambda p: _extract(p, ("pnl", "mean")) or -1e9)
    roles["top_mean"] = _get_id(best_mean)

    # top_sharpe
    best_sharpe = max(frontier, key=lambda p: _extract(p, ("pnl", "sharpe_like")) or -1e9)
    roles["top_sharpe"] = _get_id(best_sharpe)

    # best_calibrated (highest positive_rate with lowest drawdown ratio)
    def calibration_score(p: dict) -> float:
        pos_rate = _extract(p, ("pnl", "positive_rate")) or 0
        mean = _extract(p, ("pnl", "mean")) or 0
        dd = abs(_extract(p, ("drawdown", "mean_max_drawdown")) or 999)
        dd_ratio = dd / mean if mean > 0 else 999
        return pos_rate - dd_ratio  # higher is better
    best_cal = max(frontier, key=calibration_score)
    roles["best_calibrated"] = _get_id(best_cal)

    # best_maker_heavy
    maker_heavy = [p for p in frontier if "maker" in _get_family(p).lower()]
    if maker_heavy:
        best_mh = max(maker_heavy, key=lambda p: _extract(p, ("pnl", "sharpe_like")) or -1e9)
        roles["best_maker_heavy"] = _get_id(best_mh)

    # best_active_tomatoes (highest tomato PnL share)
    def tomato_score(p: dict) -> float:
        return _extract(p, ("per_product", "tomato", "mean")) or -1e9
    best_tom = max(frontier, key=tomato_score)
    tom_mean = _extract(best_tom, ("per_product", "tomato", "mean")) or 0
    if tom_mean > 0:
        roles["best_active_tomatoes"] = _get_id(best_tom)

    # best_control_anchor (highest confidence, most stable)
    promoted = [p for p in frontier if (p.get("packet_short", p).get("promote", {}).get("recommended", False))]
    if promoted:
        best_anchor = max(promoted, key=lambda p: (_extract(p, ("pnl", "positive_rate")) or 0))
        roles["best_control_anchor"] = _get_id(best_anchor)

    return roles


# ---------------------------------------------------------------------------
# Dominated / comparison helpers
# ---------------------------------------------------------------------------

def _find_dominated(candidate: dict, frontier: list[dict]) -> list[dict]:
    """Find frontier members dominated by the candidate on all key metrics."""
    dominated = []
    for member in frontier:
        if _dominates(candidate, member):
            dominated.append(member)
    return dominated


def _dominates(a: dict, b: dict) -> bool:
    """Does A dominate B? (A is better or equal on all key metrics, strictly better on at least one.)"""
    metrics = [
        (("pnl", "mean"), True),
        (("pnl", "sharpe_like"), True),
        (("pnl", "p05"), True),
    ]

    strictly_better = False
    for key_path, higher_better in metrics:
        a_val = _extract(a, key_path) or 0
        b_val = _extract(b, key_path) or 0

        if higher_better:
            if a_val < b_val:
                return False
            if a_val > b_val:
                strictly_better = True
        else:
            if a_val > b_val:
                return False
            if a_val < b_val:
                strictly_better = True

    return strictly_better


def _is_better_overall(a: dict, b: dict) -> bool:
    """Is A better than B on a composite score?"""
    a_score = (_extract(a, ("pnl", "sharpe_like")) or 0) + (_extract(a, ("pnl", "mean")) or 0) / 1000
    b_score = (_extract(b, ("pnl", "sharpe_like")) or 0) + (_extract(b, ("pnl", "mean")) or 0) / 1000
    return a_score > b_score


def _find_worst_in_family(family: str, frontier: list[dict]) -> dict | None:
    members = [p for p in frontier if _get_family(p) == family]
    if not members:
        return None
    return min(members, key=lambda p: (_extract(p, ("pnl", "sharpe_like")) or 0))


def _find_weakest_overall(frontier: list[dict]) -> dict | None:
    if not frontier:
        return None
    return min(frontier, key=lambda p: (
        (_extract(p, ("pnl", "sharpe_like")) or 0) + (_extract(p, ("pnl", "mean")) or 0) / 1000
    ))


def _is_sole_calibration_anchor(member: dict, frontier: list[dict]) -> bool:
    """Is this the only promoted member in the frontier?"""
    promoted = [
        p for p in frontier
        if p.get("packet_short", p).get("promote", {}).get("recommended", False)
    ]
    if len(promoted) <= 1:
        mid = _get_id(member)
        return any(_get_id(p) == mid for p in promoted)
    return False


def _find_packet(cid: str, all_packets: list[dict]) -> dict | None:
    for p in all_packets:
        if _get_id(p) == cid:
            return p
    return None


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


def _get_family(packet: dict[str, Any]) -> str:
    return packet.get("_family", packet.get("family", "unknown"))


def _build_summary(
    additions: list[dict],
    retirements: list[dict],
    notes: list[str],
) -> str:
    parts = []
    if additions:
        parts.append(f"Added {len(additions)}: {', '.join(a['candidate_id'] for a in additions)}.")
    if retirements:
        parts.append(f"Retired {len(retirements)}: {', '.join(r['candidate_id'] for r in retirements)}.")
    if notes:
        parts.extend(notes)
    return " ".join(parts) if parts else "No frontier changes."
