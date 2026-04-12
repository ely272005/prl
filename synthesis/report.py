"""Synthesis report generation — JSON, markdown, and brief outputs.

Produces three artifact types:
  1. JSON report:    machine-readable synthesis output
  2. Markdown report: human-readable summary of tasks, parents, batches
  3. Brief files:     individual prompt briefs ready for Prosperity GPT
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from synthesis.task import StrategyTask, ExperimentBatch
from synthesis.briefs import render_brief, render_control_brief, briefs_to_dict


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


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def build_json_report(
    batch: ExperimentBatch,
    cards: list[dict[str, Any]] | None = None,
    parent_scores: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build structured JSON report from an experiment batch."""
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "batch": _sanitize(batch.to_dict()),
        "summary": {
            "batch_id": batch.batch_id,
            "mode": batch.mode,
            "task_count": len(batch.tasks),
            "control_count": len(batch.controls),
            "overlap_warnings": batch.overlap_warnings,
        },
    }

    if cards:
        report["source_cards"] = _sanitize(cards)

    if parent_scores:
        report["parent_scores"] = _sanitize(parent_scores)

    # Task breakdown
    by_type: dict[str, int] = {}
    by_priority: dict[str, int] = {}
    by_product: dict[str, int] = {}
    for t in batch.tasks:
        by_type[t.task_type] = by_type.get(t.task_type, 0) + 1
        by_priority[t.priority] = by_priority.get(t.priority, 0) + 1
        for p in t.product_scope:
            by_product[p] = by_product.get(p, 0) + 1

    report["summary"]["by_type"] = by_type
    report["summary"]["by_priority"] = by_priority
    report["summary"]["by_product"] = by_product

    # Brief texts
    report["briefs"] = briefs_to_dict(
        batch.tasks, cards, batch.controls,
    )

    return report


def write_json_report(
    batch: ExperimentBatch,
    output_path: Path,
    cards: list[dict[str, Any]] | None = None,
    parent_scores: list[dict[str, Any]] | None = None,
) -> None:
    report = build_json_report(batch, cards, parent_scores)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

_PRIORITY_ICON = {
    "critical": "[!!]",
    "high": "[!]",
    "medium": "[-]",
    "low": "[.]",
}


def build_markdown_report(
    batch: ExperimentBatch,
    cards: list[dict[str, Any]] | None = None,
) -> str:
    lines: list[str] = []
    lines.append("# Synthesis Report — Strategy Tasks")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"Batch: **{batch.batch_id}** — {batch.title}")
    lines.append(f"Mode: {batch.mode}")
    lines.append("")

    # Summary
    lines.append(f"## Summary")
    lines.append("")
    lines.append(f"- **Exploit/Defend tasks:** {len(batch.tasks)}")
    lines.append(f"- **Controls:** {len(batch.controls)}")
    lines.append(f"- **Rationale:** {batch.rationale}")
    lines.append("")

    # Diversity
    if batch.diversity_notes:
        lines.append("### Diversity")
        lines.append("```")
        lines.append(batch.diversity_notes)
        lines.append("```")
        lines.append("")

    # Overlap warnings
    if batch.overlap_warnings:
        lines.append("### Overlap Warnings")
        for w in batch.overlap_warnings:
            lines.append(f"- {w}")
        lines.append("")

    # Task table
    lines.append("## Tasks")
    lines.append("")
    lines.append("| ID | Title | Type | Priority | Product | Parent |")
    lines.append("|----|-------|------|----------|---------|--------|")
    for t in batch.tasks:
        icon = _PRIORITY_ICON.get(t.priority, "[-]")
        products = ", ".join(t.product_scope)
        lines.append(
            f"| {t.task_id} | {t.title[:50]} | {t.task_type} | "
            f"{icon} {t.priority} | {products} | {t.parent_id} |"
        )
    lines.append("")

    # Task details
    lines.append("## Task Details")
    lines.append("")
    for t in batch.tasks:
        lines.append(f"### {t.task_id}: {t.title}")
        lines.append("")
        lines.append(f"**Type:** {t.task_type} | **Priority:** {t.priority} | "
                      f"**Confidence:** {t.confidence}")
        lines.append(f"**Source card:** {t.source_card_id} — {t.source_card_title}")
        lines.append(f"**Parent:** {t.parent_id} ({t.parent_family})")
        lines.append("")

        lines.append(f"**Objective:** {t.exploit_objective}")
        lines.append("")
        lines.append(f"**Mechanism:** {t.expected_mechanism}")
        lines.append("")
        lines.append(f"**Risk:** {t.main_risk}")
        lines.append("")

        lines.append("**Allowed changes:**")
        for c in t.allowed_changes:
            lines.append(f"  - {c}")
        lines.append("**Forbidden changes:**")
        for c in t.forbidden_changes:
            lines.append(f"  - {c}")
        lines.append("**Preservation:**")
        for c in t.preservation:
            lines.append(f"  - {c}")
        lines.append("")

        lines.append(f"**Success:** {t.success_metric} — {t.success_threshold}")
        lines.append("")

        if t.warnings:
            for w in t.warnings:
                lines.append(f"> Warning: {w}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Controls
    if batch.controls:
        lines.append("## Controls")
        lines.append("")
        for t in batch.controls:
            lines.append(f"### {t.task_id}: {t.title}")
            lines.append(f"**Type:** {t.task_type} | **Parent:** {t.parent_id}")
            lines.append(f"**Success:** {t.success_threshold}")
            lines.append("")
            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def write_markdown_report(
    batch: ExperimentBatch,
    output_path: Path,
    cards: list[dict[str, Any]] | None = None,
) -> None:
    md = build_markdown_report(batch, cards)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write(md)


# ---------------------------------------------------------------------------
# Brief file writer — one file per task
# ---------------------------------------------------------------------------

def write_brief_files(
    batch: ExperimentBatch,
    output_dir: Path,
    cards: list[dict[str, Any]] | None = None,
) -> list[Path]:
    """Write individual brief files for each task in the batch.

    Returns list of paths written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for i, task in enumerate(batch.tasks):
        card = cards[i] if cards and i < len(cards) else None
        brief_text = render_brief(task, card)
        path = output_dir / f"{task.task_id}_brief.md"
        with path.open("w") as f:
            f.write(brief_text)
        written.append(path)

    for task in batch.controls:
        brief_text = render_control_brief(task)
        path = output_dir / f"{task.task_id}_brief.md"
        with path.open("w") as f:
            f.write(brief_text)
        written.append(path)

    return written
