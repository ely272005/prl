"""Verdict-to-action routing — explicit rules for what happens to each candidate.

Maps Phase 5 verdict labels to campaign actions:
  frontier_challenger → official_gate shortlist
  escalate            → confirmation campaign (official eligible if low transfer risk)
  keep                → archive into family memory
  reject              → dead-branch record
  suspect_simulator_gain → calibration only, not official queue
  control_success     → no action
  control_failure     → calibration campaign

Every routing decision is inspectable: candidate_id, verdict, action, reason.
"""
from __future__ import annotations

from typing import Any


ROUTING_ACTIONS = (
    "official_gate_shortlist",  # Eligible for official testing
    "confirmation_campaign",    # Needs more local sessions
    "calibration_only",         # Suspicious — calibration campaign only
    "archive",                  # Informative but not actionable
    "dead_branch",              # Failed — record and avoid
    "calibration_campaign",     # Control failed — investigate noise
    "no_action",                # Nothing to do
)

ROUTING_RULES: dict[str, dict[str, Any]] = {
    "frontier_challenger": {
        "action": "official_gate_shortlist",
        "campaign_type": "official_gate",
        "priority": "high",
        "official_eligible": True,
        "reason": "Frontier challenger — eligible for official testing after confirmation.",
    },
    "escalate": {
        "action": "confirmation_campaign",
        "campaign_type": "confirmation",
        "priority": "high",
        "official_eligible": False,  # refined by transfer risk
        "reason": "Meaningful improvement — needs confirmation run before official consideration.",
    },
    "keep": {
        "action": "archive",
        "campaign_type": None,
        "priority": "low",
        "official_eligible": False,
        "reason": "Informative but not actionable — archive for family memory.",
    },
    "reject": {
        "action": "dead_branch",
        "campaign_type": None,
        "priority": "low",
        "official_eligible": False,
        "reason": "Rejected — record as dead direction.",
    },
    "suspect_simulator_gain": {
        "action": "calibration_only",
        "campaign_type": "calibration",
        "priority": "medium",
        "official_eligible": False,
        "reason": "Suspicious gain — calibration campaign only, not official queue.",
    },
    "control_success": {
        "action": "no_action",
        "campaign_type": None,
        "priority": "low",
        "official_eligible": False,
        "reason": "Control passed — experiment design validated.",
    },
    "control_failure": {
        "action": "calibration_campaign",
        "campaign_type": "calibration",
        "priority": "high",
        "official_eligible": False,
        "reason": "Control failed — noise floor investigation needed.",
    },
}

_DEFAULT_RULE: dict[str, Any] = {
    "action": "archive",
    "campaign_type": None,
    "priority": "low",
    "official_eligible": False,
    "reason": "Unknown verdict — default to archive.",
}


def route_candidates(
    candidate_verdicts: list[dict[str, Any]],
    hypothesis_verdicts: list[dict[str, Any]] | None = None,
    learnings: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Route all candidates to campaign actions.

    Returns a list of routing decisions, one per candidate.
    """
    hyp_map: dict[str, dict] = {}
    if hypothesis_verdicts:
        for hv in hypothesis_verdicts:
            tid = hv.get("task_id", "")
            if tid:
                hyp_map[tid] = hv

    decisions = []
    for cv in candidate_verdicts:
        hv = hyp_map.get(cv.get("task_id", ""))
        decision = route_candidate(cv, hv, learnings)
        decisions.append(decision)

    return decisions


def route_candidate(
    candidate_verdict: dict[str, Any],
    hypothesis_verdict: dict[str, Any] | None = None,
    learnings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Route a single candidate verdict to a campaign action."""
    verdict_label = candidate_verdict.get("verdict", "?")
    rule = ROUTING_RULES.get(verdict_label, _DEFAULT_RULE)

    decision = {
        "candidate_id": candidate_verdict.get("candidate_id", "?"),
        "verdict": verdict_label,
        "action": rule["action"],
        "campaign_type": rule["campaign_type"],
        "priority": rule["priority"],
        "official_eligible": rule["official_eligible"],
        "reason": rule["reason"],
        "family": candidate_verdict.get("family", "unknown"),
        "parent_id": candidate_verdict.get("parent_id", "?"),
        "pnl_delta": candidate_verdict.get("pnl_delta", 0),
        "sharpe_delta": candidate_verdict.get("sharpe_delta", 0),
    }

    # Refine escalate based on transfer risk
    if verdict_label == "escalate":
        transfer = candidate_verdict.get("transfer_risk", "")
        if "Low transfer risk" in transfer or "low" in transfer.lower().split("—")[0]:
            decision["official_eligible"] = True
            decision["action"] = "official_gate_shortlist"
            decision["reason"] = (
                "Escalated with low transfer risk — eligible for official queue "
                "after confirmation."
            )

    # Add hypothesis context if available
    if hypothesis_verdict:
        decision["hypothesis_outcome"] = hypothesis_verdict.get("outcome", "?")
        decision["hypothesis_id"] = hypothesis_verdict.get("hypothesis_id", "?")
    else:
        decision["hypothesis_outcome"] = None
        decision["hypothesis_id"] = None

    # Check dead zones
    if learnings:
        dead_ids = {
            dz.get("hypothesis_id", "")
            for dz in learnings.get("dead_zones", [])
        }
        hyp_id = candidate_verdict.get("source_hypothesis", "")
        if hyp_id in dead_ids:
            decision["dead_zone_warning"] = (
                f"Hypothesis {hyp_id} is in a dead zone — do not invest further."
            )

    return decision


def summarize_routing(decisions: list[dict[str, Any]]) -> str:
    """Produce a human-readable routing summary."""
    from collections import Counter
    action_counts = Counter(d["action"] for d in decisions)
    official = [d for d in decisions if d.get("official_eligible")]
    dead = [d for d in decisions if d.get("dead_zone_warning")]

    lines = ["## Routing Summary", ""]
    for action, count in action_counts.most_common():
        lines.append(f"- **{action}**: {count} candidate(s)")
    lines.append("")

    if official:
        lines.append(f"**Official-eligible**: {len(official)} candidate(s)")
        for d in official:
            lines.append(f"  - {d['candidate_id']} ({d['verdict']})")
        lines.append("")

    if dead:
        lines.append(f"**Dead zone warnings**: {len(dead)}")
        for d in dead:
            lines.append(f"  - {d['candidate_id']}: {d['dead_zone_warning']}")

    return "\n".join(lines)
