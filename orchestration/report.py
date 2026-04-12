"""Report generation for orchestration — JSON, markdown, and handoff artifacts.

Produces:
  - campaigns.json / .md
  - run_plan.json / .md
  - official_queue.json / .md
  - champions.json / .md
  - campaign_history.json / .md
  - routing_decisions.json / .md
  - prosperity_handoff/ directory with per-campaign .md files
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestration.official_queue import generate_official_memo
from orchestration.handoff import build_all_handoffs, render_handoff_markdown
from orchestration.history import summarize_recent


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_sanitize(data), f, indent=2, default=str)


def _write_md(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(text)


def write_all_reports(
    output_dir: Path,
    run_plan: dict[str, Any],
    official_queue: list[dict[str, Any]],
    champions: dict[str, Any],
    routing_decisions: list[dict[str, Any]],
    campaign_history: dict[str, Any],
    handoffs: list[dict[str, Any]],
    learnings: dict[str, Any] | None = None,
) -> list[Path]:
    """Write all orchestration reports. Returns paths written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    ts = datetime.now(timezone.utc).isoformat()

    # 1. Campaigns
    campaigns = run_plan.get("campaigns", [])
    path = output_dir / "campaigns.json"
    _write_json({"generated_at": ts, "campaigns": campaigns}, path)
    written.append(path)

    path = output_dir / "campaigns.md"
    _write_md(_render_campaigns_md(campaigns), path)
    written.append(path)

    # 2. Run plan
    path = output_dir / "run_plan.json"
    _write_json({"generated_at": ts, **run_plan}, path)
    written.append(path)

    path = output_dir / "run_plan.md"
    _write_md(_render_run_plan_md(run_plan), path)
    written.append(path)

    # 3. Official queue
    path = output_dir / "official_queue.json"
    _write_json({"generated_at": ts, "queue": official_queue}, path)
    written.append(path)

    path = output_dir / "official_queue.md"
    _write_md(generate_official_memo(official_queue, champions), path)
    written.append(path)

    # 4. Champions
    path = output_dir / "champions.json"
    _write_json({"generated_at": ts, **champions}, path)
    written.append(path)

    path = output_dir / "champions.md"
    _write_md(_render_champions_md(champions), path)
    written.append(path)

    # 5. Campaign history
    path = output_dir / "campaign_history.json"
    _write_json({"generated_at": ts, **campaign_history}, path)
    written.append(path)

    path = output_dir / "campaign_history.md"
    _write_md(summarize_recent(campaign_history, n=10), path)
    written.append(path)

    # 6. Routing decisions
    path = output_dir / "routing_decisions.json"
    _write_json({"generated_at": ts, "decisions": routing_decisions}, path)
    written.append(path)

    path = output_dir / "routing_decisions.md"
    _write_md(_render_routing_md(routing_decisions), path)
    written.append(path)

    # 7. Prosperity handoff
    handoff_dir = output_dir / "prosperity_handoff"
    handoff_dir.mkdir(parents=True, exist_ok=True)
    for ho in handoffs:
        cid = ho.get("campaign_id", "unknown").replace("-", "_")
        path = handoff_dir / f"{cid}_brief.md"
        _write_md(ho.get("brief_markdown", ""), path)
        written.append(path)

    return written


# ---------------------------------------------------------------------------
# Markdown renderers
# ---------------------------------------------------------------------------

def _render_campaigns_md(campaigns: list[dict]) -> str:
    lines = [
        "# Campaigns",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]

    if not campaigns:
        lines.append("No campaigns planned.")
        return "\n".join(lines)

    lines.append("| ID | Title | Type | Priority | E/E | Budget | Status |")
    lines.append("|----|-------|------|----------|-----|--------|--------|")
    for c in campaigns:
        lines.append(
            f"| {c.get('campaign_id', '?')} "
            f"| {c.get('title', '?')[:35]} "
            f"| {c.get('campaign_type', '?')} "
            f"| {c.get('priority', '?')} "
            f"| {c.get('exploit_explore', '?')[:3]} "
            f"| {c.get('allocated_candidates', '?')} "
            f"| {c.get('status', '?')} |"
        )
    lines.append("")

    for c in campaigns:
        lines.append(f"### {c.get('campaign_id', '?')} — {c.get('title', '?')}")
        lines.append("")
        lines.append(f"**Type:** {c.get('campaign_type', '?')} | **Priority:** {c.get('priority', '?')}")
        lines.append(f"**Objective:** {c.get('objective', '?')}")
        lines.append(f"**Success:** {c.get('success_criteria', '?')}")
        lines.append(f"**Failure:** {c.get('failure_criteria', '?')}")
        lines.append(f"**Budget:** {c.get('allocated_candidates', 0)} candidates")
        lines.append("")

        roles = c.get("planned_roles", [])
        if roles:
            lines.append("**Planned roles:**")
            for r in roles:
                lines.append(f"  - {r.get('role', '?')}: {r.get('notes', '')}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _render_run_plan_md(plan: dict) -> str:
    lines = [
        "# Run Plan",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"**Plan ID:** {plan.get('plan_id', '?')}",
        "",
        f"**Budget:** {plan.get('total_allocated', 0)}/{plan.get('budget', {}).get('total_local_candidates', '?')} local candidates, "
        f"{plan.get('budget', {}).get('total_official_tests', '?')} official tests",
        "",
        f"**Exploit/Explore:** {plan.get('exploit_ratio', 0):.0%} exploit / {1 - plan.get('exploit_ratio', 0):.0%} explore",
        "",
    ]

    lines.append(f"**Summary:** {plan.get('summary', '?')}")
    lines.append("")

    # Redundancy
    redundancy = plan.get("redundancy", {})
    issues = redundancy.get("total_issues", 0)
    if issues:
        lines.append(f"## Redundancy Issues ({issues})")
        lines.append("")
        for rec in redundancy.get("recommendations", []):
            lines.append(f"- **{rec['action']}** → {rec['target']}: {rec['reason']}")
        lines.append("")

    # Overflow
    overflow = plan.get("overflow_notes", [])
    if overflow:
        lines.append("## Budget Notes")
        for note in overflow:
            lines.append(f"- {note}")
        lines.append("")

    # Skipped
    skipped = plan.get("skipped_actions", [])
    if skipped:
        lines.append(f"## Skipped Actions ({len(skipped)})")
        for s in skipped:
            lines.append(f"- {s.get('action_type', '?')}: {s.get('detail', '')[:70]}")
        lines.append("")

    return "\n".join(lines)


def _render_champions_md(champions: dict) -> str:
    lines = [
        "# Champion Table",
        "",
        f"Updated: {champions.get('updated_at', '?')}",
        "",
    ]

    entries = champions.get("champions", [])
    if not entries:
        lines.append("No champions registered.")
        return "\n".join(lines)

    # Active
    active = [c for c in entries if c.get("status") == "active"]
    if active:
        lines.append("## Active Champions")
        lines.append("")
        lines.append("| Role | Candidate | Family | PnL | Sharpe | Pos Rate | Confidence |")
        lines.append("|------|-----------|--------|-----|--------|----------|------------|")
        for c in active:
            lines.append(
                f"| {c.get('role', '?')} "
                f"| {c.get('candidate_id', '?')[:14]} "
                f"| {c.get('family', '?')} "
                f"| {c.get('pnl_mean', 0):.0f} "
                f"| {c.get('sharpe', 0):.2f} "
                f"| {c.get('positive_rate', 0):.0%} "
                f"| {c.get('confidence', '?')} |"
            )
        lines.append("")

    # Other statuses
    for status in ("superseded", "retired", "preserved"):
        group = [c for c in entries if c.get("status") == status]
        if group:
            lines.append(f"## {status.title()}")
            lines.append("")
            for c in group:
                reason = c.get("retirement_reason") or c.get("notes") or ""
                sup = c.get("superseded_by", "")
                extra = f" → {sup}" if sup else ""
                extra += f" ({reason})" if reason else ""
                lines.append(f"- {c.get('candidate_id', '?')} ({c.get('role', '?')}){extra}")
            lines.append("")

    return "\n".join(lines)


def _render_routing_md(decisions: list[dict]) -> str:
    lines = [
        "# Routing Decisions",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]

    if not decisions:
        lines.append("No routing decisions.")
        return "\n".join(lines)

    lines.append("| Candidate | Verdict | Action | Priority | Official | Reason |")
    lines.append("|-----------|---------|--------|----------|----------|--------|")
    for d in decisions:
        official = "yes" if d.get("official_eligible") else "no"
        lines.append(
            f"| {d.get('candidate_id', '?')[:14]} "
            f"| {d.get('verdict', '?')} "
            f"| {d.get('action', '?')} "
            f"| {d.get('priority', '?')} "
            f"| {official} "
            f"| {d.get('reason', '?')[:50]} |"
        )
    lines.append("")

    # Dead zone warnings
    dead_warnings = [d for d in decisions if d.get("dead_zone_warning")]
    if dead_warnings:
        lines.append("## Dead Zone Warnings")
        lines.append("")
        for d in dead_warnings:
            lines.append(f"- **{d['candidate_id']}**: {d['dead_zone_warning']}")
        lines.append("")

    return "\n".join(lines)
