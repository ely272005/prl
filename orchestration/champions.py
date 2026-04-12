"""Champion table management — formal record of the best candidates by role.

Roles:
  - overall_champion: best composite (Sharpe + mean/1000)
  - best_sharpe: highest Sharpe-like
  - best_mean: highest mean PnL
  - best_calibrated: highest (positive_rate - drawdown_ratio)
  - best_maker_heavy: best Sharpe among maker-heavy family
  - best_active_tomatoes: best tomato PnL contributor
  - best_anchor: most reliable calibration reference (promoted, highest positive_rate)

Each champion entry tracks status: active, retired, superseded, preserved.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any


CHAMPION_ROLES = (
    "overall_champion",
    "best_sharpe",
    "best_mean",
    "best_calibrated",
    "best_maker_heavy",
    "best_active_tomatoes",
    "best_anchor",
)

CHAMPION_STATUSES = ("active", "retired", "superseded", "preserved")


def build_champion_table(
    frontier_packets: list[dict[str, Any]],
    frontier_updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a champion table from frontier packets.

    Parameters
    ----------
    frontier_packets : list[dict]
        Packets currently on the frontier.
    frontier_updates : dict, optional
        Latest frontier update for role_assignments.
    """
    champions: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()

    if not frontier_packets:
        return {"champions": [], "updated_at": now}

    # Assign each role to the best candidate
    assigned = _assign_all_roles(frontier_packets)

    for role, packet in assigned.items():
        metrics = _extract_metrics(packet)
        champions.append({
            "role": role,
            "candidate_id": _get_id(packet),
            "family": _get_family(packet),
            **metrics,
            "status": "active",
            "since": now,
            "superseded_by": None,
            "retirement_reason": None,
            "notes": "",
        })

    return {"champions": champions, "updated_at": now}


def update_champion_table(
    current_table: dict[str, Any],
    new_frontier_packets: list[dict[str, Any]],
) -> dict[str, Any]:
    """Update champion table with new frontier data.

    Existing champions that are no longer best become superseded.
    New best candidates become active.
    """
    now = datetime.now(timezone.utc).isoformat()
    old_champions = {c["role"]: c for c in current_table.get("champions", [])}
    new_assigned = _assign_all_roles(new_frontier_packets)

    updated: list[dict[str, Any]] = []

    for role in CHAMPION_ROLES:
        old = old_champions.get(role)
        new_packet = new_assigned.get(role)

        if new_packet is None:
            # No candidate for this role
            if old and old["status"] == "active":
                old = dict(old)
                old["status"] = "retired"
                old["retirement_reason"] = "No candidate available for role."
                updated.append(old)
            continue

        new_id = _get_id(new_packet)
        new_metrics = _extract_metrics(new_packet)

        if old and old["candidate_id"] == new_id:
            # Same champion — update metrics, keep status
            entry = dict(old)
            entry.update(new_metrics)
            updated.append(entry)
        else:
            # New champion
            if old and old["status"] == "active":
                # Supersede old
                old_entry = dict(old)
                old_entry["status"] = "superseded"
                old_entry["superseded_by"] = new_id
                updated.append(old_entry)

            updated.append({
                "role": role,
                "candidate_id": new_id,
                "family": _get_family(new_packet),
                **new_metrics,
                "status": "active",
                "since": now,
                "superseded_by": None,
                "retirement_reason": None,
                "notes": "",
            })

    return {"champions": updated, "updated_at": now}


def promote_champion(
    table: dict[str, Any],
    candidate_id: str,
    role: str,
    metrics: dict[str, Any] | None = None,
    family: str = "",
) -> dict[str, Any]:
    """Manually promote a candidate to a champion role."""
    now = datetime.now(timezone.utc).isoformat()
    champions = list(table.get("champions", []))

    # Supersede existing active holder of this role
    for i, c in enumerate(champions):
        if c["role"] == role and c["status"] == "active":
            champions[i] = dict(c)
            champions[i]["status"] = "superseded"
            champions[i]["superseded_by"] = candidate_id

    entry = {
        "role": role,
        "candidate_id": candidate_id,
        "family": family,
        **(metrics or {}),
        "status": "active",
        "since": now,
        "superseded_by": None,
        "retirement_reason": None,
        "notes": "",
    }
    champions.append(entry)
    return {"champions": champions, "updated_at": now}


def retire_champion(
    table: dict[str, Any],
    candidate_id: str,
    reason: str = "",
) -> dict[str, Any]:
    """Retire a champion entry."""
    champions = list(table.get("champions", []))
    for i, c in enumerate(champions):
        if c["candidate_id"] == candidate_id and c["status"] == "active":
            champions[i] = dict(c)
            champions[i]["status"] = "retired"
            champions[i]["retirement_reason"] = reason or "Manually retired."
    return {"champions": champions, "updated_at": table.get("updated_at", "")}


def preserve_champion(
    table: dict[str, Any],
    candidate_id: str,
    reason: str = "",
) -> dict[str, Any]:
    """Mark a champion as preserved (kept for calibration/anchor reasons)."""
    champions = list(table.get("champions", []))
    for i, c in enumerate(champions):
        if c["candidate_id"] == candidate_id:
            champions[i] = dict(c)
            champions[i]["status"] = "preserved"
            champions[i]["notes"] = reason or "Preserved for calibration reference."
    return {"champions": champions, "updated_at": table.get("updated_at", "")}


def get_active_champions(table: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only active champion entries."""
    return [c for c in table.get("champions", []) if c["status"] == "active"]


# ---------------------------------------------------------------------------
# Role assignment
# ---------------------------------------------------------------------------

def _assign_all_roles(packets: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Assign champion roles to packets."""
    if not packets:
        return {}

    roles: dict[str, dict] = {}

    # overall_champion: best composite score
    roles["overall_champion"] = max(packets, key=_composite_score)

    # best_sharpe
    roles["best_sharpe"] = max(packets, key=lambda p: _val(p, "sharpe_like") or -1e9)

    # best_mean
    roles["best_mean"] = max(packets, key=lambda p: _val(p, "mean") or -1e9)

    # best_calibrated: highest (positive_rate - drawdown_ratio)
    roles["best_calibrated"] = max(packets, key=_calibration_score)

    # best_maker_heavy: best Sharpe among maker-heavy family
    maker = [p for p in packets if "maker" in _get_family(p).lower()]
    if maker:
        roles["best_maker_heavy"] = max(
            maker, key=lambda p: _val(p, "sharpe_like") or -1e9,
        )

    # best_active_tomatoes: highest tomato PnL
    best_tom = max(packets, key=lambda p: _tom_mean(p))
    if _tom_mean(best_tom) > 0:
        roles["best_active_tomatoes"] = best_tom

    # best_anchor: most reliable promoted candidate
    promoted = [
        p for p in packets
        if p.get("packet_short", p).get("promote", {}).get("recommended", False)
    ]
    if promoted:
        roles["best_anchor"] = max(
            promoted, key=lambda p: _val(p, "positive_rate") or 0,
        )

    return roles


def _composite_score(p: dict) -> float:
    sharpe = _val(p, "sharpe_like") or 0
    mean = _val(p, "mean") or 0
    return sharpe + mean / 1000


def _calibration_score(p: dict) -> float:
    pos_rate = _val(p, "positive_rate") or 0
    mean = _val(p, "mean") or 0
    dd = abs(_nested(p, "drawdown", "mean_max_drawdown") or 999)
    dd_ratio = dd / mean if mean > 0 else 999
    return pos_rate - dd_ratio


def _tom_mean(p: dict) -> float:
    return _nested(p, "per_product", "tomato", "mean") or 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _val(packet: dict, field: str) -> float | None:
    """Extract a PnL-level field from a packet."""
    ps = packet.get("packet_short", packet)
    pnl = ps.get("pnl", {})
    v = pnl.get(field)
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return float(v)


def _nested(packet: dict, *keys: str) -> float | None:
    ps = packet.get("packet_short", packet)
    obj = ps
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


def _get_id(packet: dict) -> str:
    ps = packet.get("packet_short", packet)
    return (
        packet.get("_case_id")
        or packet.get("case_id")
        or ps.get("candidate_id", "?")
    )


def _get_family(packet: dict) -> str:
    return packet.get("_family", packet.get("family", "unknown"))


def _extract_metrics(packet: dict) -> dict[str, Any]:
    """Extract flat metrics dict from a packet."""
    return {
        "pnl_mean": round(_val(packet, "mean") or 0, 1),
        "sharpe": round(_val(packet, "sharpe_like") or 0, 2),
        "positive_rate": round(_val(packet, "positive_rate") or 0, 4),
        "p05": round(_val(packet, "p05") or 0, 1),
        "confidence": packet.get("packet_short", packet).get("confidence", "LOW"),
    }
