"""Campaign history — persistent record of what was tried, what worked, what didn't.

Stores per-campaign:
  - what it tried
  - what candidates were run
  - what the verdicts were
  - what learnings came out
  - what next actions followed
  - whether the campaign was worth the budget

This is critical for cumulative progress — without it the loop forgets.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_history(path: Path) -> dict[str, Any]:
    """Load campaign history from a JSON file."""
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return {"campaigns": [], "updated_at": None}


def save_history(history: dict[str, Any], path: Path) -> None:
    """Save campaign history to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    history["updated_at"] = datetime.now(timezone.utc).isoformat()
    with path.open("w") as f:
        json.dump(history, f, indent=2, default=str)


def record_campaign_result(
    history: dict[str, Any],
    campaign: dict[str, Any],
    candidate_verdicts: list[dict[str, Any]] | None = None,
    hypothesis_verdicts: list[dict[str, Any]] | None = None,
    learnings_out: list[str] | None = None,
    next_actions_out: list[str] | None = None,
    was_worth_budget: bool | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Record a completed campaign into history.

    Returns the updated history dict.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Summarize verdicts
    verdict_summary: list[dict[str, Any]] = []
    if candidate_verdicts:
        for cv in candidate_verdicts:
            verdict_summary.append({
                "candidate_id": cv.get("candidate_id", "?"),
                "verdict": cv.get("verdict", "?"),
                "pnl_delta": cv.get("pnl_delta", 0),
                "sharpe_delta": cv.get("sharpe_delta", 0),
            })

    hypothesis_summary: list[dict[str, Any]] = []
    if hypothesis_verdicts:
        for hv in hypothesis_verdicts:
            hypothesis_summary.append({
                "hypothesis_id": hv.get("hypothesis_id", "?"),
                "outcome": hv.get("outcome", "?"),
                "reason": hv.get("reason", ""),
            })

    record = {
        "campaign_id": campaign.get("campaign_id", "?"),
        "title": campaign.get("title", "?"),
        "campaign_type": campaign.get("campaign_type", "?"),
        "objective": campaign.get("objective", "?"),
        "family": campaign.get("family", ""),
        "target_mechanism": campaign.get("target_mechanism", ""),
        "exploit_explore": campaign.get("exploit_explore", "?"),
        "budget_used": campaign.get("allocated_candidates", 0),
        "created_at": campaign.get("created_at", "?"),
        "completed_at": now,
        "candidate_verdicts": verdict_summary,
        "hypothesis_verdicts": hypothesis_summary,
        "learnings_out": learnings_out or [],
        "next_actions_out": next_actions_out or [],
        "was_worth_budget": was_worth_budget,
        "notes": notes,
    }

    campaigns = list(history.get("campaigns", []))
    campaigns.append(record)

    return {"campaigns": campaigns, "updated_at": now}


def summarize_recent(
    history: dict[str, Any],
    n: int = 5,
) -> str:
    """Summarize the N most recent campaigns."""
    campaigns = history.get("campaigns", [])
    recent = campaigns[-n:]

    if not recent:
        return "No campaign history."

    lines = [f"## Recent Campaigns ({len(recent)})", ""]

    for rec in reversed(recent):
        title = rec.get("title", "?")
        ct = rec.get("campaign_type", "?")
        worth = rec.get("was_worth_budget")
        worth_str = {True: "yes", False: "no", None: "?"}[worth]

        verdicts = rec.get("candidate_verdicts", [])
        v_summary = ", ".join(
            f"{v['candidate_id'][:8]}={v['verdict']}"
            for v in verdicts[:3]
        )

        lines.append(f"### {title}")
        lines.append(f"Type: {ct} | Budget: {rec.get('budget_used', '?')} | Worth it: {worth_str}")
        if v_summary:
            lines.append(f"Verdicts: {v_summary}")
        lines.append("")

    return "\n".join(lines)


def campaign_stats(history: dict[str, Any]) -> dict[str, Any]:
    """Compute aggregate stats from campaign history."""
    campaigns = history.get("campaigns", [])
    if not campaigns:
        return {"total": 0}

    total = len(campaigns)
    by_type: dict[str, int] = {}
    worth_yes = 0
    worth_no = 0
    total_budget = 0

    for rec in campaigns:
        ct = rec.get("campaign_type", "?")
        by_type[ct] = by_type.get(ct, 0) + 1
        total_budget += rec.get("budget_used", 0)
        worth = rec.get("was_worth_budget")
        if worth is True:
            worth_yes += 1
        elif worth is False:
            worth_no += 1

    return {
        "total": total,
        "by_type": by_type,
        "total_budget_used": total_budget,
        "worth_budget_yes": worth_yes,
        "worth_budget_no": worth_no,
        "worth_budget_unknown": total - worth_yes - worth_no,
    }
