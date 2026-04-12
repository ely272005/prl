"""Prompt brief generator — renders strategy tasks into Prosperity GPT briefs.

Each brief is a structured text block designed to be pasted directly into
Prosperity GPT for constrained strategy generation. The format is tight,
self-contained, and optimized for giving the LLM just enough context to
make targeted changes without drifting.
"""
from __future__ import annotations

from typing import Any

from synthesis.task import StrategyTask


# ---------------------------------------------------------------------------
# Brief template sections
# ---------------------------------------------------------------------------

_BRIEF_TEMPLATE = """\
## Strategy Generation Brief: {task_id}

**Title:** {title}
**Type:** {task_type}
**Priority:** {priority} | **Confidence:** {confidence}

---

### Objective
{exploit_objective}

### Evidence
{evidence_block}

### Parent Strategy
- **Parent ID:** {parent_id}
- **Parent Family:** {parent_family}
- **Why this parent:** {parent_rationale}
- **Parent PnL:** {parent_pnl}

### Scope
- **Products:** {product_scope}
- **Target Regime:** {regime_str}

### Allowed Changes
{allowed_block}

### Forbidden Changes
{forbidden_block}

### Preservation Constraints
{preservation_block}

### Expected Mechanism
{expected_mechanism}

### Main Risk
{main_risk}

### Success Criteria
{eval_block}
- **Primary metric:** {success_metric}
- **Threshold:** {success_threshold}

### Warnings
{warnings_block}

### Requested Output
Generate a modified `strategy.py` that:
1. Starts from the parent strategy ({parent_id}) as the base
2. Makes ONLY the allowed changes described above
3. Gates all changes on the target regime: {regime_str}
4. Preserves all listed constraints
5. Includes a comment block at the top explaining what was changed and why

Return the complete `strategy.py` file ready to backtest.
"""


def _format_evidence(card: dict[str, Any]) -> str:
    """Format the alpha card's evidence into readable lines."""
    evidence = card.get("evidence", {})
    if not evidence:
        return "No quantitative evidence provided."

    lines = []
    for key, value in evidence.items():
        if isinstance(value, float):
            lines.append(f"- **{key}:** {value:.4f}")
        elif isinstance(value, int):
            lines.append(f"- **{key}:** {value}")
        else:
            lines.append(f"- **{key}:** {value}")

    # Add the card's fact/interpretation/exploit layers
    fact = card.get("observed_fact", "")
    interpretation = card.get("interpretation", "")
    exploit = card.get("suggested_exploit", "")

    if fact:
        lines.append(f"\n**Observed Fact:** {fact}")
    if interpretation:
        lines.append(f"**Interpretation:** {interpretation}")
    if exploit:
        lines.append(f"**Suggested Exploit:** {exploit}")

    return "\n".join(lines)


def _format_list(items: list[str], bullet: str = "-") -> str:
    """Format a list of strings as bullet points."""
    if not items:
        return f"{bullet} None"
    return "\n".join(f"{bullet} {item}" for item in items)


def _format_regime(regime: dict[str, Any]) -> str:
    """Format a regime dict into a readable string."""
    if not regime:
        return "General (no specific regime)"
    parts = []
    for k, v in regime.items():
        if k in ("product", "role", "comparison", "probe_id", "probe_family", "metric"):
            continue
        parts.append(f"{k}={v}")
    return ", ".join(parts) if parts else "General"


def render_brief(
    task: StrategyTask,
    card: dict[str, Any] | None = None,
) -> str:
    """Render a single strategy task into a Prosperity GPT prompt brief.

    Parameters
    ----------
    task : StrategyTask
        The strategy task to render.
    card : dict, optional
        The original alpha card for richer evidence display.
        If None, evidence block is built from the task itself.
    """
    evidence_block = _format_evidence(card) if card else (
        f"- **Objective:** {task.exploit_objective}\n"
        f"- **Mechanism:** {task.expected_mechanism}"
    )

    return _BRIEF_TEMPLATE.format(
        task_id=task.task_id,
        title=task.title,
        task_type=task.task_type,
        priority=task.priority,
        confidence=task.confidence,
        exploit_objective=task.exploit_objective,
        evidence_block=evidence_block,
        parent_id=task.parent_id,
        parent_family=task.parent_family,
        parent_rationale=task.parent_rationale,
        parent_pnl=task.success_threshold,
        product_scope=", ".join(task.product_scope),
        regime_str=_format_regime(task.regime_targeted),
        allowed_block=_format_list(task.allowed_changes),
        forbidden_block=_format_list(task.forbidden_changes),
        preservation_block=_format_list(task.preservation),
        expected_mechanism=task.expected_mechanism,
        main_risk=task.main_risk,
        eval_block=_format_list(task.evaluation_criteria),
        success_metric=task.success_metric,
        success_threshold=task.success_threshold,
        warnings_block=_format_list(task.warnings) if task.warnings else "- None",
    )


def render_control_brief(task: StrategyTask) -> str:
    """Render a control task into a simpler brief.

    Control tasks (calibration_check, near_parent_control) need less
    structure since they don't involve alpha cards.
    """
    lines = [
        f"## Control Brief: {task.task_id}",
        f"",
        f"**Title:** {task.title}",
        f"**Type:** {task.task_type}",
        f"**Priority:** {task.priority}",
        f"",
        f"### Objective",
        f"{task.exploit_objective}",
        f"",
        f"### Parent Strategy",
        f"- **Parent ID:** {task.parent_id}",
        f"- **Parent Family:** {task.parent_family}",
        f"",
        f"### Allowed Changes",
        _format_list(task.allowed_changes),
        f"",
        f"### Forbidden Changes",
        _format_list(task.forbidden_changes),
        f"",
        f"### Success Criteria",
        _format_list(task.evaluation_criteria),
        f"- **Threshold:** {task.success_threshold}",
        f"",
    ]

    if task.task_type == "calibration_check":
        lines.extend([
            "### Requested Output",
            f"Return the EXACT parent strategy ({task.parent_id}) with NO changes.",
            "This is a calibration baseline — any modification invalidates the control.",
        ])
    else:
        lines.extend([
            "### Requested Output",
            f"Return the parent strategy ({task.parent_id}) with ONE cosmetic-only change:",
            "- Rename an internal variable, reorder imports, or add a comment.",
            "- Do NOT change any functional logic, parameters, or thresholds.",
            "This is a noise-estimation control — deviation from parent PnL indicates noise floor.",
        ])

    return "\n".join(lines)


def render_batch_briefs(
    tasks: list[StrategyTask],
    cards: list[dict[str, Any]] | None = None,
    controls: list[StrategyTask] | None = None,
    batch_title: str = "Experiment Batch",
) -> str:
    """Render all briefs for an experiment batch into a single document.

    Parameters
    ----------
    tasks : list[StrategyTask]
        Exploit / defend tasks in the batch.
    cards : list[dict], optional
        Original alpha cards, matched by index to tasks.
    controls : list[StrategyTask], optional
        Control tasks for the batch.
    batch_title : str
        Title for the batch header.
    """
    sections = [
        f"# {batch_title}",
        f"",
        f"Total tasks: {len(tasks)} exploit/defend + {len(controls or [])} controls",
        f"",
        "---",
        "",
    ]

    # Exploit / defend briefs
    for i, task in enumerate(tasks):
        card = cards[i] if cards and i < len(cards) else None
        sections.append(render_brief(task, card))
        sections.append("\n---\n")

    # Control briefs
    if controls:
        sections.append("# Control Tasks\n")
        for task in controls:
            sections.append(render_control_brief(task))
            sections.append("\n---\n")

    return "\n".join(sections)


def briefs_to_dict(
    tasks: list[StrategyTask],
    cards: list[dict[str, Any]] | None = None,
    controls: list[StrategyTask] | None = None,
) -> list[dict[str, Any]]:
    """Convert tasks to a list of brief dicts for JSON serialization."""
    result = []

    for i, task in enumerate(tasks):
        card = cards[i] if cards and i < len(cards) else None
        result.append({
            "task_id": task.task_id,
            "title": task.title,
            "task_type": task.task_type,
            "priority": task.priority,
            "confidence": task.confidence,
            "brief_text": render_brief(task, card),
            "is_control": False,
        })

    for task in (controls or []):
        result.append({
            "task_id": task.task_id,
            "title": task.title,
            "task_type": task.task_type,
            "priority": task.priority,
            "confidence": task.confidence,
            "brief_text": render_control_brief(task),
            "is_control": True,
        })

    return result
