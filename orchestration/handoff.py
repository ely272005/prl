"""Prosperity-GPT handoff packaging — disciplined briefs for strategy generation.

Produces a clean artifact for each campaign containing:
  - campaign objective
  - parent strategy IDs
  - source learnings
  - exact constraints
  - exact target mechanism
  - what to preserve / what not to touch
  - what counts as success
  - what prior hypotheses failed
  - what role each requested candidate should play

The point is to make Prosperity GPT receive high-quality, disciplined briefs,
not random context dumps.
"""
from __future__ import annotations

from typing import Any


def build_campaign_handoff(
    campaign: dict[str, Any],
    learnings: dict[str, Any],
    champions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a Prosperity GPT handoff artifact for one campaign.

    Returns a dict with structured fields and a rendered markdown brief.
    """
    dead_zones = learnings.get("dead_zones", [])
    validated = learnings.get("validated_mechanisms", [])
    falsified = learnings.get("falsified_mechanisms", [])
    family_lessons = learnings.get("family_lessons", {})

    family = campaign.get("family", "")
    relevant_family_lessons = family_lessons.get(family, [])

    handoff = {
        "campaign_id": campaign.get("campaign_id", "?"),
        "campaign_type": campaign.get("campaign_type", "?"),
        "title": campaign.get("title", "?"),
        "objective": campaign.get("objective", "?"),
        "target_mechanism": campaign.get("target_mechanism", ""),
        "product_scope": campaign.get("product_scope", []),
        "allowed_parents": campaign.get("allowed_parents", []),
        "forbidden_directions": campaign.get("forbidden_directions", []),
        "preservation_constraints": campaign.get("preservation_constraints", []),
        "success_criteria": campaign.get("success_criteria", ""),
        "failure_criteria": campaign.get("failure_criteria", ""),
        "budget": campaign.get("budget", {}),
        "allocated_candidates": campaign.get("allocated_candidates", 0),
        "planned_roles": campaign.get("planned_roles", []),
        "validated_mechanisms": [v.get("hypothesis_id", "?") for v in validated],
        "falsified_mechanisms": [f.get("hypothesis_id", "?") for f in falsified],
        "dead_zones": [dz.get("hypothesis_id", "?") for dz in dead_zones],
        "family_lessons": relevant_family_lessons,
        "champion_context": _champion_context(champions),
        "brief_markdown": "",  # filled below
    }

    handoff["brief_markdown"] = render_handoff_markdown(handoff)
    return handoff


def build_all_handoffs(
    campaigns: list[dict[str, Any]],
    learnings: dict[str, Any],
    champions: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build handoff artifacts for all active campaigns."""
    handoffs = []
    for camp in campaigns:
        if camp.get("allocated_candidates", 0) > 0:
            handoffs.append(build_campaign_handoff(camp, learnings, champions))
    return handoffs


def render_handoff_markdown(handoff: dict[str, Any]) -> str:
    """Render a Prosperity GPT handoff brief as markdown."""
    lines = [
        f"# Campaign Brief: {handoff.get('title', '?')}",
        "",
        f"**Campaign ID:** {handoff.get('campaign_id', '?')}",
        f"**Type:** {handoff.get('campaign_type', '?')}",
        "",
    ]

    # Objective
    lines.append("## Objective")
    lines.append("")
    lines.append(handoff.get("objective", "?"))
    lines.append("")

    # Target mechanism
    if handoff.get("target_mechanism"):
        lines.append(f"**Target mechanism:** {handoff['target_mechanism']}")
        lines.append("")

    # Product scope
    if handoff.get("product_scope"):
        lines.append(f"**Product scope:** {', '.join(handoff['product_scope'])}")
        lines.append("")

    # Parents
    if handoff.get("allowed_parents"):
        lines.append("## Allowed Parents")
        lines.append("")
        for pid in handoff["allowed_parents"]:
            lines.append(f"- `{pid}`")
        lines.append("")

    # Constraints
    lines.append("## Constraints")
    lines.append("")

    preservation = handoff.get("preservation_constraints", [])
    if preservation:
        lines.append("**Preserve:**")
        for c in preservation:
            lines.append(f"- {c}")
        lines.append("")

    forbidden = handoff.get("forbidden_directions", [])
    if forbidden:
        lines.append("**Do NOT touch:**")
        for f in forbidden:
            lines.append(f"- {f}")
        lines.append("")

    # Dead zones
    dead = handoff.get("dead_zones", [])
    if dead:
        lines.append("## Dead Zones (already falsified)")
        lines.append("")
        for dz in dead:
            lines.append(f"- `{dz}` — do not explore this direction.")
        lines.append("")

    # Validated mechanisms
    validated = handoff.get("validated_mechanisms", [])
    if validated:
        lines.append("## Validated Mechanisms (proven to work)")
        lines.append("")
        for vm in validated:
            lines.append(f"- `{vm}`")
        lines.append("")

    # Family lessons
    family_lessons = handoff.get("family_lessons", [])
    if family_lessons:
        lines.append("## Family Lessons")
        lines.append("")
        for lesson in family_lessons[:5]:
            lines.append(f"- {lesson}")
        lines.append("")

    # Champion context
    champ_ctx = handoff.get("champion_context", "")
    if champ_ctx:
        lines.append("## Current Champion")
        lines.append("")
        lines.append(champ_ctx)
        lines.append("")

    # Success / failure criteria
    lines.append("## Success Criteria")
    lines.append("")
    lines.append(handoff.get("success_criteria", "Not specified."))
    lines.append("")

    lines.append("## Failure Criteria")
    lines.append("")
    lines.append(handoff.get("failure_criteria", "Not specified."))
    lines.append("")

    # Requested candidates
    roles = handoff.get("planned_roles", [])
    n = handoff.get("allocated_candidates", 0)
    if roles:
        lines.append(f"## Requested Candidates ({n})")
        lines.append("")
        for i, role in enumerate(roles):
            lines.append(f"{i + 1}. **{role.get('role', '?')}** — {role.get('notes', '')}")
        lines.append("")

    # Falsified (for reference)
    falsified = handoff.get("falsified_mechanisms", [])
    if falsified:
        lines.append("## Previously Falsified")
        lines.append("")
        for fm in falsified:
            lines.append(f"- `{fm}`")
        lines.append("")

    return "\n".join(lines)


def _champion_context(champions: dict[str, Any] | None) -> str:
    if not champions:
        return ""

    active = [c for c in champions.get("champions", []) if c.get("status") == "active"]
    if not active:
        return "No active champion."

    parts = []
    for c in active:
        parts.append(
            f"- **{c.get('role', '?')}**: {c.get('candidate_id', '?')} "
            f"(PnL={c.get('pnl_mean', 0):.0f}, Sharpe={c.get('sharpe', 0):.2f})"
        )
    return "\n".join(parts)
