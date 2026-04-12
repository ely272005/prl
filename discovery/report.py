"""Discovery report generation — JSON and markdown outputs from alpha cards."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from discovery.alpha_card import AlphaCard, _sanitize


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def build_json_report(
    cards: list[AlphaCard],
    summary: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build structured JSON report from alpha cards."""
    by_category: dict[str, list[dict]] = {}
    for card in cards:
        by_category.setdefault(card.category, []).append(card.to_dict())

    confidence_counts = {"high": 0, "medium": 0, "low": 0}
    category_counts: dict[str, int] = {}
    for card in cards:
        confidence_counts[card.confidence] = confidence_counts.get(card.confidence, 0) + 1
        category_counts[card.category] = category_counts.get(card.category, 0) + 1

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_cards": len(cards),
            "by_confidence": confidence_counts,
            "by_category": category_counts,
            **(summary or {}),
        },
        "alpha_cards": [card.to_dict() for card in cards],
        "cards_by_category": _sanitize(by_category),
    }

    # Include comparison stats if available
    if comparison:
        report["comparison"] = _sanitize({
            "packet_count": comparison.get("packet_count", 0),
            "winner_count": comparison.get("winner_count", 0),
            "loser_count": comparison.get("loser_count", 0),
            "split_method": comparison.get("split_method", "unknown"),
            "family_comparison": comparison.get("family_comparison", {}),
        })

    # Classification: strong vs noise
    strong = [c for c in cards if c.confidence in ("high", "medium") and c.strength > 2.0]
    speculative = [c for c in cards if c not in strong]
    report["classification"] = {
        "strong_patterns": [c.card_id for c in strong],
        "speculative_patterns": [c.card_id for c in speculative],
    }

    return report


def write_json_report(
    cards: list[AlphaCard],
    output_path: Path,
    summary: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
) -> None:
    report = build_json_report(cards, summary, comparison)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

_CONFIDENCE_ICON = {"high": "[H]", "medium": "[M]", "low": "[L]"}
_CATEGORY_ICON = {
    "regime_edge": ">>",
    "role_mismatch": "<>",
    "bot_weakness": "!!",
    "danger_refinement": "**",
    "winner_trait": "^^",
    "inventory_exploit": "~~",
}


def build_markdown_report(
    cards: list[AlphaCard],
    summary: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
) -> str:
    lines: list[str] = []
    lines.append("# Discovery Report — Alpha Cards")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    if summary:
        lines.append("## Data Sources")
        lines.append(f"- Sessions: {summary.get('session_count', '?')}")
        if summary.get("packet_count"):
            lines.append(f"- Packets: {summary.get('packet_count', '?')} "
                         f"({summary.get('winner_count', '?')} promoted, "
                         f"{summary.get('loser_count', '?')} rejected)")
        if summary.get("probe_result_count"):
            lines.append(f"- Probe results: {summary.get('probe_result_count', 0)}")
        lines.append("")

    # Summary stats
    lines.append(f"## Summary: {len(cards)} Alpha Cards Generated")
    lines.append("")

    strong = [c for c in cards if c.confidence in ("high", "medium") and c.strength > 2.0]
    speculative = [c for c in cards if c not in strong]

    if strong:
        lines.append(f"**Strong patterns ({len(strong)}):** "
                      + ", ".join(c.card_id for c in strong))
    if speculative:
        lines.append(f"**Speculative ({len(speculative)}):** "
                      + ", ".join(c.card_id for c in speculative))
    lines.append("")

    by_cat: dict[str, int] = {}
    for c in cards:
        by_cat[c.category] = by_cat.get(c.category, 0) + 1
    lines.append("| Category | Count |")
    lines.append("|----------|-------|")
    for cat, cnt in sorted(by_cat.items()):
        lines.append(f"| {cat} | {cnt} |")
    lines.append("")

    # Top alpha cards with full detail
    lines.append("## Alpha Cards (ranked by strength)")
    lines.append("")

    for i, card in enumerate(cards):
        conf_icon = _CONFIDENCE_ICON.get(card.confidence, "[?]")
        cat_icon = _CATEGORY_ICON.get(card.category, "--")

        lines.append(f"### {cat_icon} {card.card_id}: {card.title}")
        lines.append("")
        lines.append(f"**Category:** {card.category} | "
                      f"**Confidence:** {card.confidence} {conf_icon} | "
                      f"**Products:** {', '.join(card.products)}")
        lines.append("")

        lines.append(f"**OBSERVED FACT:** {card.observed_fact}")
        lines.append("")
        lines.append(f"**INTERPRETATION:** {card.interpretation}")
        lines.append("")
        lines.append(f"**SUGGESTED EXPLOIT:** {card.suggested_exploit}")
        lines.append("")

        # Evidence
        if card.evidence:
            lines.append("Evidence:")
            for k, v in card.evidence.items():
                lines.append(f"  - {k}: {_fmt(v)}")
            lines.append("")

        # Sample size
        if card.sample_size:
            parts = [f"{k}={v}" for k, v in card.sample_size.items()]
            lines.append(f"Sample: {', '.join(parts)}")
            lines.append("")

        if card.candidate_strategy_style:
            lines.append(f"Strategy style: *{card.candidate_strategy_style}*")
        if card.recommended_experiment:
            lines.append(f"Next experiment: *{card.recommended_experiment}*")
        lines.append("")

        if card.warnings:
            for w in card.warnings:
                lines.append(f"> Warning: {w}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Recommended experiments
    experiments = [(c.card_id, c.recommended_experiment) for c in cards if c.recommended_experiment]
    if experiments:
        lines.append("## Recommended Next Experiments")
        lines.append("")
        for cid, exp in experiments:
            lines.append(f"- **{cid}**: {exp}")
        lines.append("")

    # Explicit warnings
    lines.append("## Things to Explicitly Avoid")
    lines.append("")
    danger_cards = [c for c in cards if c.category == "danger_refinement"]
    if danger_cards:
        for c in danger_cards:
            lines.append(f"- **{c.card_id}**: {c.observed_fact}")
    else:
        lines.append("- No danger refinement cards generated from this dataset.")
    lines.append("")

    # Comparison summary
    if comparison:
        family_cmp = comparison.get("family_comparison", {})
        if family_cmp:
            lines.append("## Family Performance Summary")
            lines.append("")
            lines.append("| Family | Count | Promoted | Mean PnL | Mean Sharpe |")
            lines.append("|--------|-------|----------|----------|-------------|")
            for fam, stats in sorted(family_cmp.items(), key=lambda x: -x[1].get("pnl_mean", 0)):
                pnl = stats.get("pnl_mean", 0)
                sharpe = stats.get("sharpe_mean", 0)
                lines.append(
                    f"| {fam} | {stats.get('count', 0)} | "
                    f"{stats.get('promoted', 0)} | "
                    f"{pnl:.0f} | {sharpe:.1f} |"
                )
            lines.append("")

    return "\n".join(lines)


def write_markdown_report(
    cards: list[AlphaCard],
    output_path: Path,
    summary: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
) -> None:
    md = build_markdown_report(cards, summary, comparison)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write(md)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return "N/A"
        return f"{v:.4f}"
    if isinstance(v, dict):
        parts = []
        for k2, v2 in v.items():
            if isinstance(v2, float) and not (math.isnan(v2) or math.isinf(v2)):
                parts.append(f"{k2}={v2:.3f}")
            else:
                parts.append(f"{k2}={v2}")
        return "{" + ", ".join(parts) + "}"
    return str(v)
