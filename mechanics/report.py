"""Report generation — JSON and markdown outputs from probe results."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mechanics.probe_spec import ProbeResult


def _sanitize(obj: Any) -> Any:
    """Make an object JSON-serializable (handle NaN, inf)."""
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
    results: list[ProbeResult],
    dataset_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a structured JSON report from probe results."""
    # Group by family
    by_family: dict[str, list[dict]] = {}
    for r in results:
        by_family.setdefault(r.family, []).append(_sanitize(r.to_dict()))

    verdict_counts = {"supported": 0, "refuted": 0, "inconclusive": 0, "insufficient_data": 0}
    for r in results:
        verdict_counts[r.verdict] = verdict_counts.get(r.verdict, 0) + 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_summary or {},
        "summary": {
            "total_probes_run": len(results),
            "verdict_counts": verdict_counts,
        },
        "results_by_family": by_family,
    }


def write_json_report(
    results: list[ProbeResult],
    output_path: Path,
    dataset_summary: dict[str, Any] | None = None,
) -> None:
    """Write JSON report to file."""
    report = build_json_report(results, dataset_summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

_VERDICT_ICON = {
    "supported": "[+]",
    "refuted": "[-]",
    "inconclusive": "[?]",
    "insufficient_data": "[!]",
}


def build_markdown_report(
    results: list[ProbeResult],
    dataset_summary: dict[str, Any] | None = None,
) -> str:
    """Build a readable markdown report from probe results."""
    lines: list[str] = []
    lines.append("# Mechanics Probe Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    if dataset_summary:
        lines.append("## Dataset")
        lines.append(f"- Sources: {dataset_summary.get('dataset_label', 'unknown')}")
        lines.append(f"- Sessions: {dataset_summary.get('total_sessions', '?')}")
        lines.append("")

    # Verdict summary
    verdict_counts = {"supported": 0, "refuted": 0, "inconclusive": 0, "insufficient_data": 0}
    for r in results:
        verdict_counts[r.verdict] = verdict_counts.get(r.verdict, 0) + 1

    lines.append("## Verdict Summary")
    lines.append("")
    lines.append(f"| Verdict | Count |")
    lines.append(f"|---------|-------|")
    for v, c in verdict_counts.items():
        lines.append(f"| {v} | {c} |")
    lines.append("")

    # Quick findings (only supported/refuted)
    actionable = [r for r in results if r.verdict in ("supported", "refuted")]
    if actionable:
        lines.append("## Key Findings")
        lines.append("")
        for r in actionable:
            icon = _VERDICT_ICON[r.verdict]
            lines.append(f"- {icon} **{r.product} / {r.title}**: {r.detail}")
        lines.append("")

    # Full results by family
    by_family: dict[str, list[ProbeResult]] = {}
    for r in results:
        by_family.setdefault(r.family, []).append(r)

    for family in sorted(by_family):
        lines.append(f"## Family: {family}")
        lines.append("")

        for r in by_family[family]:
            icon = _VERDICT_ICON[r.verdict]
            lines.append(f"### {icon} {r.probe_id} — {r.product}")
            lines.append("")
            lines.append(f"**{r.title}**")
            lines.append("")
            lines.append(f"Hypothesis: *{r.hypothesis}*")
            lines.append("")
            lines.append(f"Verdict: **{r.verdict}** (confidence: {r.confidence})")
            lines.append("")
            lines.append(f"{r.detail}")
            lines.append("")

            # Sample size
            if r.sample_size:
                parts = [f"{k}={v}" for k, v in r.sample_size.items()]
                lines.append(f"Sample: {', '.join(parts)}")
                lines.append("")

            # Key metrics (compact)
            if r.metrics:
                lines.append("Key metrics:")
                lines.append("")
                for k, v in r.metrics.items():
                    if isinstance(v, dict) and len(str(v)) > 200:
                        # Large nested dict — just show keys
                        lines.append(f"- {k}: ({len(v)} entries)")
                    else:
                        lines.append(f"- {k}: {_format_metric(v)}")
                lines.append("")

            if r.warnings:
                for w in r.warnings:
                    lines.append(f"> Warning: {w}")
                lines.append("")

            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def _format_metric(v: Any) -> str:
    """Format a metric value for display."""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return "N/A"
        return f"{v:.4f}"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, dict):
        # Compact dict display
        parts = []
        for k2, v2 in v.items():
            if isinstance(v2, dict) and "mean" in v2:
                mean = v2["mean"]
                count = v2.get("count", "?")
                if isinstance(mean, float) and not math.isnan(mean):
                    parts.append(f"{k2}: mean={mean:.3f} (n={count})")
                else:
                    parts.append(f"{k2}: N/A (n={count})")
            elif isinstance(v2, float) and not math.isnan(v2):
                parts.append(f"{k2}={v2:.4f}")
            else:
                parts.append(f"{k2}={v2}")
        return "{" + ", ".join(parts) + "}"
    return str(v)


def write_markdown_report(
    results: list[ProbeResult],
    output_path: Path,
    dataset_summary: dict[str, Any] | None = None,
) -> None:
    """Write markdown report to file."""
    md = build_markdown_report(results, dataset_summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write(md)
