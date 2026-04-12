"""Report generation for adjudication — JSON, markdown, and structured outputs.

Produces:
  - candidate_verdicts.json / .md
  - hypothesis_verdicts.json / .md
  - frontier_updates.json / .md
  - batch_learnings.json / .md
  - next_actions.json / .md
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adjudication.next_actions import format_gpt_summary


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


# ---------------------------------------------------------------------------
# Write all reports
# ---------------------------------------------------------------------------

def write_all_reports(
    output_dir: Path,
    candidate_verdicts: list[dict[str, Any]],
    hypothesis_verdicts: list[dict[str, Any]],
    frontier_updates: dict[str, Any],
    learnings: dict[str, Any],
    next_actions: list[dict[str, Any]],
) -> list[Path]:
    """Write all adjudication reports to output_dir. Returns paths written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    ts = datetime.now(timezone.utc).isoformat()

    # 1. Candidate verdicts
    path = output_dir / "candidate_verdicts.json"
    _write_json({"generated_at": ts, "verdicts": candidate_verdicts}, path)
    written.append(path)

    path = output_dir / "candidate_verdicts.md"
    _write_md(_render_candidate_verdicts_md(candidate_verdicts), path)
    written.append(path)

    # 2. Hypothesis verdicts
    path = output_dir / "hypothesis_verdicts.json"
    _write_json({"generated_at": ts, "verdicts": hypothesis_verdicts}, path)
    written.append(path)

    path = output_dir / "hypothesis_verdicts.md"
    _write_md(_render_hypothesis_verdicts_md(hypothesis_verdicts), path)
    written.append(path)

    # 3. Frontier updates
    path = output_dir / "frontier_updates.json"
    _write_json({"generated_at": ts, **frontier_updates}, path)
    written.append(path)

    path = output_dir / "frontier_updates.md"
    _write_md(_render_frontier_updates_md(frontier_updates), path)
    written.append(path)

    # 4. Batch learnings
    path = output_dir / "batch_learnings.json"
    _write_json({"generated_at": ts, **learnings}, path)
    written.append(path)

    path = output_dir / "batch_learnings.md"
    _write_md(_render_learnings_md(learnings), path)
    written.append(path)

    # 5. Next actions
    path = output_dir / "next_actions.json"
    _write_json({"generated_at": ts, "actions": next_actions}, path)
    written.append(path)

    path = output_dir / "next_actions.md"
    _write_md(format_gpt_summary(next_actions, learnings), path)
    written.append(path)

    return written


# ---------------------------------------------------------------------------
# Markdown renderers
# ---------------------------------------------------------------------------

def _render_candidate_verdicts_md(verdicts: list[dict]) -> str:
    lines = [
        "# Candidate Verdicts",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| ID | Task | Parent | Verdict | PnL Delta | Sharpe Delta | Confidence |")
    lines.append("|----|------|--------|---------|-----------|-------------|------------|")
    for v in verdicts:
        lines.append(
            f"| {v.get('candidate_id', '?')[:12]} "
            f"| {v.get('task_id', '?')} "
            f"| {v.get('parent_id', '?')[:12]} "
            f"| **{v.get('verdict', '?')}** "
            f"| {v.get('pnl_delta', 0):+.0f} "
            f"| {v.get('sharpe_delta', 0):+.2f} "
            f"| {v.get('confidence', '?')} |"
        )
    lines.append("")

    # Detailed verdicts
    lines.append("## Details")
    lines.append("")
    for v in verdicts:
        verdict = v.get("verdict", "?")
        icon = _verdict_icon(verdict)
        lines.append(f"### {icon} {v.get('candidate_id', '?')[:12]} — {verdict}")
        lines.append("")
        lines.append(f"**Task:** {v.get('task_id', '?')} | **Parent:** {v.get('parent_id', '?')}")
        lines.append(f"**Family:** {v.get('family', '?')}")
        lines.append("")

        lines.append(f"**PnL:** {v.get('pnl_mean', 0):.0f} (delta {v.get('pnl_delta', 0):+.0f})")
        lines.append(f"**Sharpe:** {v.get('sharpe', 0):.2f} (delta {v.get('sharpe_delta', 0):+.2f})")
        lines.append(f"**EMERALDS:** {v.get('emerald_mean', 0):.0f} (delta {v.get('emerald_delta', 0):+.0f})")
        lines.append(f"**TOMATOES:** {v.get('tomato_mean', 0):.0f} (delta {v.get('tomato_delta', 0):+.0f})")
        lines.append("")

        lines.append(f"**Reason:** {v.get('reason', '?')}")
        lines.append("")
        lines.append(f"**Mechanism:** {v.get('mechanism_interpretation', '?')}")
        lines.append("")
        lines.append(f"**Transfer risk:** {v.get('transfer_risk', '?')}")
        lines.append("")
        lines.append(f"**Next action:** {v.get('recommended_next_action', '?')}")
        lines.append("")

        # Suspicion flags
        suspicion = v.get("suspicion", {})
        if suspicion.get("flags"):
            lines.append("**Suspicion flags:**")
            for f in suspicion["flags"]:
                lines.append(f"  - {f.get('flag', '?')}: {f.get('detail', '')}")
            lines.append("")

        # Preservation
        pres = v.get("preservation_audit", {})
        if pres.get("violations"):
            lines.append("**Preservation violations:**")
            for viol in pres["violations"]:
                lines.append(f"  - [{viol.get('severity', '?')}] {viol.get('detail', '')}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _render_hypothesis_verdicts_md(verdicts: list[dict]) -> str:
    lines = [
        "# Hypothesis Verdicts",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]

    lines.append("| Hypothesis | Task | Outcome | Mean | Sharpe | Detail |")
    lines.append("|------------|------|---------|------|--------|--------|")
    for v in verdicts:
        outcome = v.get("outcome", "?")
        icon = _outcome_icon(outcome)
        lines.append(
            f"| {v.get('hypothesis_id', '?')} "
            f"| {v.get('task_id', '?')} "
            f"| {icon} {outcome} "
            f"| {'Y' if v.get('mean_helped') else 'N'} "
            f"| {'Y' if v.get('sharpe_helped') else 'N'} "
            f"| {v.get('reason', '?')[:60]} |"
        )
    lines.append("")

    # Lessons
    for v in verdicts:
        lessons = v.get("lessons", [])
        if lessons:
            lines.append(f"### {v.get('hypothesis_id', '?')} — Lessons")
            for lesson in lessons:
                lines.append(f"- {lesson}")
            lines.append("")

    return "\n".join(lines)


def _render_frontier_updates_md(updates: dict) -> str:
    lines = [
        "# Frontier Updates",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"Frontier: {updates.get('frontier_size_before', '?')} → {updates.get('frontier_size_after', '?')}",
        "",
    ]

    additions = updates.get("additions", [])
    if additions:
        lines.append("## Additions")
        for a in additions:
            lines.append(
                f"- **{a.get('candidate_id', '?')}** ({a.get('family', '?')}): "
                f"PnL={a.get('pnl_mean', 0):.0f}, Sharpe={a.get('sharpe', 0):.2f}. "
                f"{a.get('reason', '')}"
            )
        lines.append("")

    retirements = updates.get("retirements", [])
    if retirements:
        lines.append("## Retirements")
        for r in retirements:
            lines.append(f"- **{r.get('candidate_id', '?')}** ({r.get('family', '?')}): {r.get('reason', '')}")
        lines.append("")

    roles = updates.get("role_assignments", {})
    if roles:
        lines.append("## Role Assignments")
        for role, cid in roles.items():
            lines.append(f"- **{role}**: {cid}")
        lines.append("")

    lines.append(f"**Summary:** {updates.get('changes_summary', 'No changes.')}")
    return "\n".join(lines)


def _render_learnings_md(learnings: dict) -> str:
    lines = [
        "# Batch Learnings",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"**{learnings.get('summary', '')}**",
        "",
    ]

    for section, title in [
        ("validated_mechanisms", "Validated Mechanisms"),
        ("falsified_mechanisms", "Falsified Mechanisms"),
        ("suspicious_directions", "Suspicious Directions"),
        ("promising_zones", "Promising Zones"),
        ("dead_zones", "Dead Zones"),
    ]:
        items = learnings.get(section, [])
        if items:
            lines.append(f"## {title}")
            for item in items:
                if isinstance(item, dict):
                    hyp_id = item.get("hypothesis_id", item.get("candidate_id", "?"))
                    lines.append(f"- **{hyp_id}**: {item.get('reason', '')}")
                    for lesson in item.get("lessons", []):
                        lines.append(f"  - {lesson}")
                else:
                    lines.append(f"- {item}")
            lines.append("")

    family_lessons = learnings.get("family_lessons", {})
    if family_lessons:
        lines.append("## Family Lessons")
        for family, lessons in family_lessons.items():
            lines.append(f"### {family}")
            for lesson in lessons:
                lines.append(f"- {lesson}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verdict_icon(verdict: str) -> str:
    return {
        "frontier_challenger": "[FC]",
        "escalate": "[UP]",
        "keep": "[OK]",
        "control_success": "[CS]",
        "control_failure": "[CF]",
        "reject": "[XX]",
        "suspect_simulator_gain": "[??]",
    }.get(verdict, "[--]")


def _outcome_icon(outcome: str) -> str:
    return {
        "validated": "[V]",
        "partially_validated": "[P]",
        "falsified": "[F]",
        "inconclusive": "[?]",
        "informative_failure": "[I]",
        "not_applicable": "[-]",
    }.get(outcome, "[--]")
