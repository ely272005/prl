"""Run-plan generator — converts Phase 5 outputs into a concrete experiment plan.

For each campaign, the run plan specifies:
  - how many candidates to generate
  - what role each candidate serves
  - which parents are allowed
  - which mechanisms are being attacked
  - which constraints must be preserved
  - why the campaign exists
  - what counts as success / failure
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from orchestration.campaigns import create_campaigns_from_actions
from orchestration.routing import route_candidates
from orchestration.redundancy import check_all_redundancy, filter_redundant_campaigns
from orchestration.allocation import allocate_budget


# Candidate roles within a campaign
CANDIDATE_ROLES = (
    "challenger",           # Primary experimental candidate
    "near_twin_control",    # Near-parent control for noise measurement
    "calibration_anchor",   # Exact-parent reproduction
    "exploration_probe",    # Broad-search variant
    "mechanism_variant",    # Same mechanism, different parameter
    "champion_defender",    # Defending current champion
)


def build_run_plan(
    next_actions: list[dict[str, Any]],
    learnings: dict[str, Any],
    frontier_updates: dict[str, Any],
    candidate_verdicts: list[dict[str, Any]],
    hypothesis_verdicts: list[dict[str, Any]] | None = None,
    frontier_packets: list[dict[str, Any]] | None = None,
    budget: dict[str, int] | None = None,
    exploit_ratio: float = 0.70,
) -> dict[str, Any]:
    """Build a full run plan from Phase 5 outputs.

    This is the main entry point for the orchestrator.

    Parameters
    ----------
    next_actions : list[dict]
        From Phase 5 recommend_next_actions().
    learnings : dict
        From Phase 5 extract_batch_learnings().
    frontier_updates : dict
        From Phase 5 compute_frontier_updates().
    candidate_verdicts : list[dict]
        From Phase 5 adjudicate_candidate().
    hypothesis_verdicts : list[dict], optional
        From Phase 5 adjudicate_hypothesis().
    frontier_packets : list[dict], optional
        Current frontier packets for redundancy checks.
    budget : dict, optional
        Budget limits.
    exploit_ratio : float
        Exploit/explore ratio (default 0.70).
    """
    now = datetime.now(timezone.utc).isoformat()

    # 1. Route candidates
    routing_decisions = route_candidates(
        candidate_verdicts, hypothesis_verdicts, learnings,
    )

    # 2. Create campaigns from next actions
    campaigns = create_campaigns_from_actions(
        next_actions, learnings, frontier_updates, candidate_verdicts,
    )

    # 3. Build frontier verdict stubs for redundancy check
    frontier_verdicts = _build_frontier_stubs(frontier_packets or [])

    # 4. Check redundancy
    dead_zones = learnings.get("dead_zones", [])
    redundancy = check_all_redundancy(
        campaigns, candidate_verdicts, frontier_verdicts, dead_zones,
    )

    # 5. Filter redundant campaigns
    campaigns = filter_redundant_campaigns(campaigns, redundancy)

    # 6. Allocate budget
    allocation = allocate_budget(campaigns, budget, exploit_ratio)
    campaigns = allocation["campaigns"]

    # 7. Assign candidate roles within each campaign
    for camp in campaigns:
        camp["planned_roles"] = _assign_candidate_roles(camp)

    # 8. Collect skipped actions
    skipped = [
        a for a in next_actions
        if a.get("action_type") in ("stop_exploring", "no_action")
    ]

    return {
        "plan_id": f"RP-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}",
        "created_at": now,
        "budget": allocation["budget"],
        "exploit_ratio": exploit_ratio,
        "exploit_budget": allocation["exploit_budget"],
        "explore_budget": allocation["explore_budget"],
        "total_allocated": allocation["total_allocated"],
        "campaigns": campaigns,
        "routing_decisions": routing_decisions,
        "redundancy": redundancy,
        "skipped_actions": skipped,
        "overflow_notes": allocation.get("overflow_notes", []),
        "summary": _build_summary(campaigns, routing_decisions, redundancy, allocation),
    }


def _assign_candidate_roles(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    """Assign roles to planned candidate slots for a campaign."""
    ct = campaign.get("campaign_type", "exploration")
    n = campaign.get("allocated_candidates", 0)
    if n <= 0:
        return []

    roles: list[dict[str, Any]] = []

    if ct == "confirmation":
        # Primary challenger + control
        roles.append({"role": "challenger", "notes": "Primary confirmation candidate."})
        if n >= 2:
            roles.append({"role": "calibration_anchor", "notes": "Parent reproduction for noise baseline."})
        for i in range(n - 2):
            roles.append({"role": "mechanism_variant", "notes": f"Variant #{i + 1} for robustness check."})

    elif ct == "exploration":
        # Multiple probes
        for i in range(min(n, max(1, n - 1))):
            roles.append({"role": "exploration_probe", "notes": f"Probe #{i + 1}."})
        if n >= 3:
            roles.append({"role": "near_twin_control", "notes": "Control for noise measurement."})

    elif ct == "official_gate":
        roles.append({"role": "challenger", "notes": "Primary official candidate."})
        if n >= 2:
            roles.append({"role": "calibration_anchor", "notes": "Control for official comparison."})

    elif ct == "calibration":
        for i in range(n):
            roles.append({"role": "calibration_anchor", "notes": f"Calibration candidate #{i + 1}."})

    elif ct == "champion_defense":
        roles.append({"role": "champion_defender", "notes": "Defender of current champion."})
        if n >= 2:
            roles.append({"role": "near_twin_control", "notes": "Near-twin for regression check."})

    elif ct == "control_batch":
        for i in range(n):
            roles.append({"role": "near_twin_control", "notes": f"Control #{i + 1}."})

    else:
        for i in range(n):
            roles.append({"role": "challenger", "notes": f"Candidate #{i + 1}."})

    return roles[:n]


def _build_frontier_stubs(frontier_packets: list[dict]) -> list[dict]:
    """Build minimal verdict-like dicts from frontier packets for redundancy checks."""
    stubs = []
    for p in frontier_packets:
        ps = p.get("packet_short", p)
        pnl = ps.get("pnl", {})
        pp = ps.get("per_product", {})
        stubs.append({
            "candidate_id": p.get("_case_id", p.get("case_id", ps.get("candidate_id", "?"))),
            "pnl_mean": pnl.get("mean", 0),
            "sharpe": pnl.get("sharpe_like", 0),
            "positive_rate": pnl.get("positive_rate", 0),
            "emerald_mean": pp.get("emerald", {}).get("mean", 0),
            "tomato_mean": pp.get("tomato", {}).get("mean", 0),
        })
    return stubs


def _build_summary(
    campaigns: list[dict],
    routing: list[dict],
    redundancy: dict,
    allocation: dict,
) -> str:
    """Build a human-readable run plan summary."""
    active = [c for c in campaigns if c.get("allocated_candidates", 0) > 0]
    deferred = [c for c in campaigns if c.get("allocated_candidates", 0) == 0]
    official_eligible = [r for r in routing if r.get("official_eligible")]

    parts = [
        f"{len(active)} active campaign(s), {len(deferred)} deferred.",
        f"Total candidates: {allocation.get('total_allocated', 0)}/{allocation.get('total_local', 0)}.",
        f"Exploit/explore ratio: {allocation.get('exploit_ratio', 0):.0%}.",
        f"Official-eligible: {len(official_eligible)} candidate(s).",
    ]

    issues = redundancy.get("total_issues", 0)
    if issues:
        parts.append(f"Redundancy issues: {issues}.")

    return " ".join(parts)
