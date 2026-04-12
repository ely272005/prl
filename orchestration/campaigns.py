"""Campaign abstraction — coherent research pushes with clear objectives.

A campaign represents a focused research effort such as:
  - confirming a frontier challenger with more sessions
  - exploring a validated mechanism on different parents
  - preparing candidates for official testing
  - validating simulator calibration
  - defending the current champion against regression

Each campaign has a type, objective, budget, success/failure criteria,
and explicit links to the learnings that motivated it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


CAMPAIGN_TYPES = (
    "exploration",        # Probing new mechanisms or families
    "confirmation",       # Confirming a promising result with more sessions
    "official_gate",      # Preparing candidates for official testing
    "calibration",        # Validating simulator or noise floor
    "champion_defense",   # Protecting the best current structure
    "control_batch",      # Running controls to validate experiment design
)

CAMPAIGN_STATUSES = (
    "planned",
    "active",
    "completed",
    "abandoned",
)

CAMPAIGN_PRIORITIES = ("high", "medium", "low")

# How each campaign type should behave operationally
CAMPAIGN_TYPE_PROPERTIES = {
    "exploration": {
        "novelty_tolerance": "high",
        "scope": "broad",
        "default_budget": {"max_candidates": 6, "max_sessions_per_candidate": 20},
        "exploit_explore": "explore",
    },
    "confirmation": {
        "novelty_tolerance": "low",
        "scope": "narrow",
        "default_budget": {"max_candidates": 3, "max_sessions_per_candidate": 50},
        "exploit_explore": "exploit",
    },
    "official_gate": {
        "novelty_tolerance": "none",
        "scope": "very_narrow",
        "default_budget": {"max_candidates": 2, "max_sessions_per_candidate": 50},
        "exploit_explore": "exploit",
    },
    "calibration": {
        "novelty_tolerance": "none",
        "scope": "narrow",
        "default_budget": {"max_candidates": 4, "max_sessions_per_candidate": 30},
        "exploit_explore": "exploit",
    },
    "champion_defense": {
        "novelty_tolerance": "none",
        "scope": "very_narrow",
        "default_budget": {"max_candidates": 2, "max_sessions_per_candidate": 30},
        "exploit_explore": "exploit",
    },
    "control_batch": {
        "novelty_tolerance": "none",
        "scope": "narrow",
        "default_budget": {"max_candidates": 3, "max_sessions_per_candidate": 20},
        "exploit_explore": "exploit",
    },
}

_counter = 0


def _next_campaign_id() -> str:
    global _counter
    _counter += 1
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"CAM-{ts}-{_counter:03d}"


def reset_counter() -> None:
    """Reset counter (for tests)."""
    global _counter
    _counter = 0


def create_campaign(
    title: str,
    campaign_type: str,
    objective: str,
    priority: str = "medium",
    family: str = "",
    target_mechanism: str = "",
    product_scope: list[str] | None = None,
    allowed_parents: list[str] | None = None,
    forbidden_directions: list[str] | None = None,
    preservation_constraints: list[str] | None = None,
    success_criteria: str = "",
    failure_criteria: str = "",
    budget: dict[str, int] | None = None,
    exploit_explore: str | None = None,
    source_learnings: list[str] | None = None,
    source_next_actions: list[dict] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Create a new campaign dict."""
    props = CAMPAIGN_TYPE_PROPERTIES.get(campaign_type, {})
    return {
        "campaign_id": _next_campaign_id(),
        "title": title,
        "campaign_type": campaign_type,
        "objective": objective,
        "status": "planned",
        "priority": priority,
        "family": family,
        "target_mechanism": target_mechanism,
        "product_scope": product_scope or [],
        "allowed_parents": allowed_parents or [],
        "forbidden_directions": forbidden_directions or [],
        "preservation_constraints": preservation_constraints or [],
        "success_criteria": success_criteria,
        "failure_criteria": failure_criteria,
        "budget": budget or dict(props.get("default_budget", {"max_candidates": 5, "max_sessions_per_candidate": 20})),
        "exploit_explore": exploit_explore or props.get("exploit_explore", "exploit"),
        "novelty_tolerance": props.get("novelty_tolerance", "low"),
        "scope": props.get("scope", "narrow"),
        "source_learnings": source_learnings or [],
        "source_next_actions": source_next_actions or [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "candidates_run": [],
        "verdicts": [],
        "learnings_out": [],
        "next_actions_out": [],
        "notes": notes,
        "was_worth_budget": None,
    }


def create_campaigns_from_actions(
    next_actions: list[dict[str, Any]],
    learnings: dict[str, Any],
    frontier_updates: dict[str, Any],
    candidate_verdicts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create campaigns from Phase 5 next-action recommendations.

    Groups related actions into coherent campaigns, assigns types,
    and populates constraints from learnings.
    """
    campaigns: list[dict[str, Any]] = []
    dead_zones = {dz.get("hypothesis_id", "") for dz in learnings.get("dead_zones", [])}
    dead_reasons = {
        dz.get("hypothesis_id", ""): dz.get("reason", "")
        for dz in learnings.get("dead_zones", [])
    }

    # Classify actions
    confirm_actions: list[dict] = []
    explore_actions: list[dict] = []
    calibration_actions: list[dict] = []
    product_gate_actions: list[dict] = []
    try_parent_actions: list[dict] = []

    for action in next_actions:
        at = action.get("action_type", "")
        if at == "confirm_challenger":
            confirm_actions.append(action)
        elif at == "explore_further":
            explore_actions.append(action)
        elif at == "investigate_noise":
            calibration_actions.append(action)
        elif at == "product_gate":
            product_gate_actions.append(action)
        elif at == "try_different_parent":
            try_parent_actions.append(action)
        # stop_exploring, promote_to_official, refine_preservation, no_action → no campaign

    all_dead = list(dead_zones)

    # 1. Confirmation campaigns — one per frontier challenger
    for action in confirm_actions:
        target = action.get("target", "?")
        cv = _find_verdict(target, candidate_verdicts)
        family = cv.get("family", "unknown") if cv else "unknown"
        mechanism = cv.get("attribution", {}).get("dominant_mechanism", "") if cv else ""
        parent_id = cv.get("parent_id", "") if cv else ""

        campaigns.append(create_campaign(
            title=f"Confirm {target} ({family})",
            campaign_type="confirmation",
            objective=f"Run 50+ sessions on {target} to confirm frontier challenge.",
            priority="high",
            family=family,
            target_mechanism=mechanism,
            allowed_parents=[parent_id] if parent_id else [],
            forbidden_directions=all_dead,
            success_criteria=(
                f"Candidate maintains PnL and Sharpe advantage over parent with HIGH confidence."
            ),
            failure_criteria=(
                f"Candidate reverts to parent range or suspicion flags emerge."
            ),
            source_next_actions=[action],
        ))

    # 2. Exploration campaigns — one per validated mechanism
    for action in explore_actions:
        target = action.get("target", "?")
        if target in dead_zones:
            continue

        campaigns.append(create_campaign(
            title=f"Explore {target}",
            campaign_type="exploration",
            objective=f"Explore validated mechanism {target} with parameter variations.",
            priority="high",
            target_mechanism=target,
            forbidden_directions=all_dead,
            success_criteria="Find parameter optimum or discover mechanism limits.",
            failure_criteria="No improvement over existing best; mechanism saturated.",
            source_next_actions=[action],
        ))

    # 3. Calibration campaign — if control failures detected
    if calibration_actions:
        campaigns.append(create_campaign(
            title="Calibration validation",
            campaign_type="calibration",
            objective="Validate simulator noise floor after control failures.",
            priority="high",
            forbidden_directions=all_dead,
            success_criteria="Controls reproduce parent within 5% tolerance.",
            failure_criteria="Controls still deviate; noise floor too high for reliable testing.",
            source_next_actions=calibration_actions,
        ))

    # 4. Product-gated campaigns
    for action in product_gate_actions:
        target = action.get("target", "?")
        if target in dead_zones:
            continue

        campaigns.append(create_campaign(
            title=f"Product-gate {target}",
            campaign_type="exploration",
            objective=f"Re-run {target} with product-gated application.",
            priority="medium",
            target_mechanism=target,
            forbidden_directions=all_dead,
            success_criteria="Mechanism helps target product without hurting the other.",
            failure_criteria="Product gating does not isolate the effect.",
            source_next_actions=[action],
        ))

    # 5. Cross-parent exploration
    for action in try_parent_actions:
        target = action.get("target", "?")
        if target in dead_zones:
            continue

        campaigns.append(create_campaign(
            title=f"Cross-parent test: {target}",
            campaign_type="exploration",
            objective=f"Test hypothesis {target} on a different family base.",
            priority="low",
            target_mechanism=target,
            forbidden_directions=all_dead,
            success_criteria="Mechanism works on a different parent base.",
            failure_criteria="Mechanism is parent-specific; no transfer.",
            source_next_actions=[action],
        ))

    # 6. Champion defense — if frontier changed
    if frontier_updates.get("additions"):
        new_ids = [a.get("candidate_id", "") for a in frontier_updates["additions"]]
        campaigns.append(create_campaign(
            title="Champion defense",
            campaign_type="champion_defense",
            objective=(
                f"Verify current champion still holds after frontier additions: "
                f"{', '.join(new_ids)}."
            ),
            priority="medium",
            success_criteria="Champion performance stable under recalibration.",
            failure_criteria="Champion regressed; update champion table.",
        ))

    return campaigns


def _find_verdict(candidate_id: str, verdicts: list[dict]) -> dict | None:
    """Find a verdict by candidate_id (exact or prefix match)."""
    for v in verdicts:
        vid = v.get("candidate_id", "")
        if vid == candidate_id or vid.startswith(candidate_id) or candidate_id.startswith(vid):
            return v
    return None
