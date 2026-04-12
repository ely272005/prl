"""Candidate-level verdict engine — assigns structured verdicts to experiment results.

For each candidate, produces a full verdict report including:
  - structured comparison deltas
  - preservation audit
  - mechanism attribution
  - suspicion flags
  - a final verdict label with explicit reason

Verdict labels:
  - frontier_challenger: beats frontier on key metrics, clean packet
  - escalate: meaningful improvement over parent, worth more sessions
  - keep: minor improvement or neutral, informative but not a breakthrough
  - control_success: control task reproduced parent as expected
  - control_failure: control task deviated from parent unexpectedly
  - reject: worse than parent or too noisy/suspect to trust
  - suspect_simulator_gain: gain exists but likely not real alpha
"""
from __future__ import annotations

from typing import Any

from adjudication.comparison import (
    compare_pair,
    compare_to_frontier,
    compare_to_family,
    summarize_comparison,
    _extract,
)
from adjudication.preservation import audit_preservation
from adjudication.attribution import attribute_mechanism
from adjudication.suspicion import detect_suspicions


VERDICT_LABELS = (
    "frontier_challenger",
    "escalate",
    "keep",
    "control_success",
    "control_failure",
    "reject",
    "suspect_simulator_gain",
)


def adjudicate_candidate(
    candidate: dict[str, Any],
    parent: dict[str, Any],
    task: dict[str, Any],
    frontier: list[dict[str, Any]] | None = None,
    family_peers: list[dict[str, Any]] | None = None,
    family_name: str | None = None,
) -> dict[str, Any]:
    """Produce a full structured verdict for a candidate.

    Parameters
    ----------
    candidate : dict
        Candidate packet (packet_short at top level or nested).
    parent : dict
        Parent packet for comparison.
    task : dict
        The StrategyTask dict that generated this candidate.
    frontier : list[dict], optional
        Current frontier set for frontier comparison.
    family_peers : list[dict], optional
        Packets from the same family.
    family_name : str, optional
        Family label for the candidate.
    """
    cps = candidate.get("packet_short", candidate)
    task_type = task.get("task_type", "exploit")

    # 1. Comparison engine
    vs_parent = compare_pair(candidate, parent, "parent")
    vs_frontier = compare_to_frontier(candidate, frontier or [])
    vs_family = compare_to_family(
        candidate, family_peers or [], family_name or "unknown"
    ) if family_peers else None

    comparison_summary = summarize_comparison(vs_parent, vs_frontier, vs_family)

    # 2. Preservation audit
    preservation = audit_preservation(task, candidate, parent)

    # 3. Mechanism attribution
    attribution = attribute_mechanism(candidate, parent)

    # 4. Suspicion detection
    suspicion = detect_suspicions(candidate, parent, preservation, attribution)

    # 5. Extract key metrics
    pnl_mean = _extract(candidate, ("pnl", "mean")) or 0
    sharpe = _extract(candidate, ("pnl", "sharpe_like")) or 0
    p_pnl_mean = _extract(parent, ("pnl", "mean")) or 0
    p_sharpe = _extract(parent, ("pnl", "sharpe_like")) or 0
    confidence = cps.get("confidence", "LOW")
    promote = cps.get("promote", {}).get("recommended", False)
    kill = cps.get("kill", {}).get("recommended", False)

    pnl_delta = pnl_mean - p_pnl_mean
    sharpe_delta = sharpe - p_sharpe

    # 6. Assign verdict
    verdict, reason = _assign_verdict(
        task_type=task_type,
        pnl_delta=pnl_delta,
        sharpe_delta=sharpe_delta,
        pnl_mean=pnl_mean,
        sharpe=sharpe,
        confidence=confidence,
        promote=promote,
        kill=kill,
        preservation=preservation,
        suspicion=suspicion,
        vs_parent=vs_parent,
        vs_frontier=vs_frontier,
    )

    # 7. Compute per-product contribution
    c_em = _extract(candidate, ("per_product", "emerald", "mean")) or 0
    c_tom = _extract(candidate, ("per_product", "tomato", "mean")) or 0
    p_em = _extract(parent, ("per_product", "emerald", "mean")) or 0
    p_tom = _extract(parent, ("per_product", "tomato", "mean")) or 0

    # 8. Build verdict report
    return {
        "candidate_id": _get_id(candidate),
        "task_id": task.get("task_id", "?"),
        "parent_id": task.get("parent_id", _get_id(parent)),
        "family": family_name or task.get("parent_family", "unknown"),
        "batch_id": task.get("batch_id", "?"),
        "source_hypothesis": task.get("source_card_id", "?"),
        "task_type": task_type,

        # Metrics
        "pnl_mean": round(pnl_mean, 1),
        "pnl_std": round(_extract(candidate, ("pnl", "std")) or 0, 1),
        "sharpe": round(sharpe, 2),
        "p05": round(_extract(candidate, ("pnl", "p05")) or 0, 1),
        "p50": round(_extract(candidate, ("pnl", "p50")) or 0, 1),
        "p95": round(_extract(candidate, ("pnl", "p95")) or 0, 1),
        "positive_rate": round(_extract(candidate, ("pnl", "positive_rate")) or 0, 4),
        "confidence": confidence,
        "promote_recommended": promote,
        "kill_recommended": kill,

        # Per-product
        "emerald_mean": round(c_em, 1),
        "tomato_mean": round(c_tom, 1),
        "emerald_delta": round(c_em - p_em, 1),
        "tomato_delta": round(c_tom - p_tom, 1),

        # Deltas vs parent
        "pnl_delta": round(pnl_delta, 1),
        "sharpe_delta": round(sharpe_delta, 2),

        # Comparison results
        "vs_parent": vs_parent,
        "vs_frontier": vs_frontier,
        "vs_family": vs_family,
        "comparison_summary": comparison_summary,

        # Audit results
        "preservation_audit": preservation,
        "attribution": attribution,
        "suspicion": suspicion,

        # Verdict
        "verdict": verdict,
        "reason": reason,
        "mechanism_interpretation": attribution.get("summary", ""),
        "transfer_risk": _assess_transfer_risk(cps, suspicion, attribution),
        "recommended_next_action": _recommend_next(verdict, task, attribution, suspicion),
    }


# ---------------------------------------------------------------------------
# Verdict assignment logic
# ---------------------------------------------------------------------------

def _assign_verdict(
    task_type: str,
    pnl_delta: float,
    sharpe_delta: float,
    pnl_mean: float,
    sharpe: float,
    confidence: str,
    promote: bool,
    kill: bool,
    preservation: dict,
    suspicion: dict,
    vs_parent: dict,
    vs_frontier: dict,
) -> tuple[str, str]:
    """Assign a verdict label and reason."""

    # Control tasks have their own logic
    if task_type == "calibration_check":
        if preservation.get("verdict") == "clean":
            return "control_success", "Calibration reproduced parent within tolerance."
        return "control_failure", f"Calibration deviated: {preservation.get('reason', '?')}."

    if task_type == "near_parent_control":
        if preservation.get("verdict") in ("clean", "suspect"):
            return "control_success", "Near-parent control within expected noise range."
        return "control_failure", f"Near-parent control deviated: {preservation.get('reason', '?')}."

    # Preservation violation → reject or suspect
    if preservation.get("verdict") == "violated":
        return "reject", (
            f"Preservation constraints violated: {preservation.get('reason')}. "
            f"Result is not a valid test of the stated hypothesis."
        )

    # High suspicion → suspect_simulator_gain
    if suspicion.get("suspicion_level") == "high":
        flags = [f["flag"] for f in suspicion.get("flags", [])]
        return "suspect_simulator_gain", (
            f"Suspicious gain: {', '.join(flags)}. "
            f"Likely not real alpha."
        )

    # Kill recommended → reject
    if kill:
        return "reject", (
            f"Packet recommends kill. "
            f"PnL={pnl_mean:.0f}, Sharpe={sharpe:.2f}, confidence={confidence}."
        )

    # Worse than parent on both mean and Sharpe → reject
    if pnl_delta < 0 and sharpe_delta < 0:
        return "reject", (
            f"Worse than parent on both PnL ({pnl_delta:+.0f}) "
            f"and Sharpe ({sharpe_delta:+.2f})."
        )

    # Beats frontier on key metrics → frontier_challenger
    beats = vs_frontier.get("beats_frontier_on", [])
    if "pnl_mean" in beats and "sharpe" in beats and not suspicion.get("is_suspicious"):
        return "frontier_challenger", (
            f"Beats frontier on PnL and Sharpe. "
            f"PnL={pnl_mean:.0f} ({pnl_delta:+.0f} vs parent), "
            f"Sharpe={sharpe:.2f} ({sharpe_delta:+.2f})."
        )

    # Promote recommended + better than parent → escalate
    if promote and pnl_delta > 0:
        return "escalate", (
            f"Promote-worthy improvement: "
            f"PnL={pnl_mean:.0f} ({pnl_delta:+.0f}), "
            f"Sharpe={sharpe:.2f} ({sharpe_delta:+.2f}), "
            f"confidence={confidence}."
        )

    # Better Sharpe with non-negative PnL delta → escalate
    if sharpe_delta > 1.0 and pnl_delta >= 0:
        return "escalate", (
            f"Material Sharpe improvement ({sharpe_delta:+.2f}) "
            f"with stable or better PnL ({pnl_delta:+.0f})."
        )

    # Better mean but Sharpe flat or slightly worse → keep with caution
    if pnl_delta > 0 and sharpe_delta > -1.0:
        if suspicion.get("suspicion_level") == "medium":
            return "suspect_simulator_gain", (
                f"PnL improved ({pnl_delta:+.0f}) but suspicion flags present: "
                f"{', '.join(f['flag'] for f in suspicion.get('flags', []))}."
            )
        return "keep", (
            f"Modest improvement: PnL {pnl_delta:+.0f}, Sharpe {sharpe_delta:+.2f}. "
            f"Informative but not a breakthrough."
        )

    # Mixed results
    if pnl_delta > 0 and sharpe_delta < -1.0:
        return "keep", (
            f"PnL improved ({pnl_delta:+.0f}) but Sharpe degraded ({sharpe_delta:+.2f}). "
            f"Higher mean came at the cost of more variance."
        )

    if pnl_delta < 0 and sharpe_delta > 0:
        return "keep", (
            f"Sharpe improved ({sharpe_delta:+.2f}) but PnL dropped ({pnl_delta:+.0f}). "
            f"Cleaner but less profitable."
        )

    # Default: keep as informative
    return "keep", (
        f"Mixed or neutral result: PnL {pnl_delta:+.0f}, Sharpe {sharpe_delta:+.2f}. "
        f"No clear improvement or regression."
    )


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


def _assess_transfer_risk(
    cps: dict[str, Any],
    suspicion: dict[str, Any],
    attribution: dict[str, Any],
) -> str:
    """Assess risk that gains won't transfer to official server."""
    risks = []

    if suspicion.get("is_suspicious"):
        risks.append("suspicious gain pattern")

    dominant = attribution.get("dominant_mechanism", "")
    if dominant == "aggressiveness_change":
        risks.append("gain from aggression may not transfer")

    confidence = cps.get("confidence", "LOW")
    if confidence == "LOW":
        risks.append("LOW confidence — insufficient sessions")

    sim_note = cps.get("external_validity_note", "")
    if "cannot guarantee" in sim_note.lower():
        risks.append("simulator fidelity uncertain")

    if not risks:
        return "Low transfer risk — clean gain pattern."
    return "Transfer risk: " + "; ".join(risks) + "."


def _recommend_next(
    verdict: str,
    task: dict[str, Any],
    attribution: dict[str, Any],
    suspicion: dict[str, Any],
) -> str:
    """Generate a recommended next action based on the verdict."""
    task_type = task.get("task_type", "")

    if verdict == "frontier_challenger":
        return (
            f"Run 50+ sessions for high-confidence confirmation. "
            f"If confirmed, promote to official testing set."
        )

    if verdict == "escalate":
        dominant = attribution.get("dominant_mechanism", "")
        return (
            f"Run more sessions to increase confidence. "
            f"Explore further along dominant mechanism ({dominant})."
        )

    if verdict == "keep":
        return (
            f"Informative result. Extract learnings for next batch. "
            f"No further runs needed for this specific candidate."
        )

    if verdict == "reject":
        card_id = task.get("source_card_id", "?")
        return (
            f"Hypothesis {card_id} may be falsified in this direction. "
            f"Consider alternative mechanisms or different parent."
        )

    if verdict == "suspect_simulator_gain":
        return (
            f"Do not promote. Investigate whether gain is real "
            f"by running with different seeds or more sessions."
        )

    if verdict == "control_success":
        return "Control validates experiment design. Proceed with comparisons."

    if verdict == "control_failure":
        return (
            "Control failed — noise floor is high. "
            "Treat all gains in this batch with extra skepticism."
        )

    return "No specific recommendation."
