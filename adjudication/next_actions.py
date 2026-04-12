"""Next-action recommendation layer — proposes what to do after adjudication.

Produces specific, actionable recommendations, not vague suggestions.

Examples of good output:
  - "Run 50+ sessions on mh07 to confirm frontier challenge."
  - "Attack TOMATOES activity between 73474 and ah03 with tighter preservation."
  - "Stop exploring aggressive join_inside=2 on 73474-like bases."
  - "Promote mh07 and ah03 to official testing set."
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


ACTION_TYPES = (
    "confirm_challenger",      # Run more sessions to confirm a frontier challenge
    "explore_further",         # Explore a validated mechanism more deeply
    "promote_to_official",     # Move candidate to official testing
    "stop_exploring",          # Direction is dead, stop wasting runs
    "refine_preservation",     # Tighten constraints for cleaner tests
    "try_different_parent",    # Same hypothesis, different base strategy
    "product_gate",            # Apply change to one product only
    "investigate_noise",       # Control failed; investigate noise floor
    "no_action",               # Batch produced no actionable signal
)


def recommend_next_actions(
    candidate_verdicts: list[dict[str, Any]],
    hypothesis_verdicts: list[dict[str, Any]],
    learnings: dict[str, Any],
    frontier_updates: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate specific next-action recommendations.

    Returns a list of action dicts, each with:
      - action_type
      - priority: high / medium / low
      - target: what to act on
      - detail: specific instruction
      - rationale: why
    """
    actions: list[dict[str, Any]] = []

    # 1. Confirm frontier challengers
    for cv in candidate_verdicts:
        if cv.get("verdict") == "frontier_challenger":
            actions.append({
                "action_type": "confirm_challenger",
                "priority": "high",
                "target": cv.get("candidate_id", "?"),
                "detail": (
                    f"Run 50+ sessions on {cv.get('candidate_id', '?')} "
                    f"to confirm frontier challenge. "
                    f"Current: PnL={cv.get('pnl_mean', 0):.0f}, "
                    f"Sharpe={cv.get('sharpe', 0):.2f}."
                ),
                "rationale": cv.get("reason", ""),
            })

    # 2. Explore validated mechanisms further
    for hv in hypothesis_verdicts:
        if hv.get("outcome") == "validated":
            actions.append({
                "action_type": "explore_further",
                "priority": "high",
                "target": hv.get("hypothesis_id", "?"),
                "detail": (
                    f"Mechanism '{hv.get('hypothesis_title', '?')}' validated. "
                    f"Generate follow-up batch varying one parameter at a time "
                    f"to find the optimum."
                ),
                "rationale": hv.get("reason", ""),
            })

    # 3. Promote candidates that are ready
    for add in frontier_updates.get("additions", []):
        if add.get("verdict") == "frontier_challenger":
            actions.append({
                "action_type": "promote_to_official",
                "priority": "medium",
                "target": add.get("candidate_id", "?"),
                "detail": (
                    f"Promote {add.get('candidate_id', '?')} to official testing set "
                    f"after high-confidence confirmation."
                ),
                "rationale": add.get("reason", ""),
            })

    # 4. Stop exploring dead zones
    for dz in learnings.get("dead_zones", []):
        actions.append({
            "action_type": "stop_exploring",
            "priority": "medium",
            "target": dz.get("hypothesis_id", "?"),
            "detail": (
                f"Stop exploring {dz.get('hypothesis_id', '?')}: "
                f"failed {dz.get('failure_count', 0)} times. "
                f"Direction is exhausted."
            ),
            "rationale": dz.get("reason", ""),
        })

    # 5. Product-gating recommendations from partial validations
    for hv in hypothesis_verdicts:
        if hv.get("outcome") == "partially_validated" and hv.get("single_product_only"):
            actions.append({
                "action_type": "product_gate",
                "priority": "medium",
                "target": hv.get("hypothesis_id", "?"),
                "detail": (
                    f"Hypothesis '{hv.get('hypothesis_title', '?')}' helped one product "
                    f"but hurt the other. Re-run with product-gated application."
                ),
                "rationale": hv.get("reason", ""),
            })

    # 6. Try different parent for inconclusive hypotheses
    inconclusive = [hv for hv in hypothesis_verdicts if hv.get("outcome") == "inconclusive"]
    if inconclusive:
        # Group by hypothesis_id
        by_hyp: dict[str, list] = defaultdict(list)
        for hv in inconclusive:
            by_hyp[hv["hypothesis_id"]].append(hv)

        for hyp_id, hvs in by_hyp.items():
            if len(hvs) == 1:  # Only tried once
                actions.append({
                    "action_type": "try_different_parent",
                    "priority": "low",
                    "target": hyp_id,
                    "detail": (
                        f"Hypothesis {hyp_id} was inconclusive on one parent. "
                        f"Try a different family base to see if the mechanism works elsewhere."
                    ),
                    "rationale": hvs[0].get("reason", ""),
                })

    # 7. Preservation refinement
    for cv in candidate_verdicts:
        pres = cv.get("preservation_audit", {})
        if pres.get("verdict") == "suspect" and cv.get("verdict") != "reject":
            actions.append({
                "action_type": "refine_preservation",
                "priority": "low",
                "target": cv.get("task_id", "?"),
                "detail": (
                    f"Task {cv.get('task_id', '?')} had suspect preservation "
                    f"({pres.get('reason', '?')}). "
                    f"Tighten constraints in next batch."
                ),
                "rationale": pres.get("reason", ""),
            })

    # 8. Control failure → investigate noise
    control_failures = [
        cv for cv in candidate_verdicts if cv.get("verdict") == "control_failure"
    ]
    if control_failures:
        actions.append({
            "action_type": "investigate_noise",
            "priority": "high",
            "target": "batch_controls",
            "detail": (
                f"{len(control_failures)} control(s) failed. "
                f"Noise floor is high. Run additional calibration checks "
                f"before trusting any gains from this batch."
            ),
            "rationale": "Control tasks deviated from parent unexpectedly.",
        })

    # 9. If nothing actionable, say so
    if not actions:
        actions.append({
            "action_type": "no_action",
            "priority": "low",
            "target": "batch",
            "detail": "Batch produced no actionable signal. Consider broader hypothesis generation.",
            "rationale": "No frontier challengers, no validated mechanisms, no clear next step.",
        })

    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    actions.sort(key=lambda a: priority_order.get(a["priority"], 9))

    return actions


def format_gpt_summary(
    actions: list[dict[str, Any]],
    learnings: dict[str, Any],
) -> str:
    """Format next actions and learnings into a Prosperity-GPT-ready summary.

    This is meant to be pasted into Prosperity GPT to guide the next generation round.
    """
    lines = [
        "## Experiment Results Summary",
        "",
    ]

    # Validated mechanisms
    validated = learnings.get("validated_mechanisms", [])
    if validated:
        lines.append("### Validated Mechanisms (keep exploring)")
        for v in validated:
            lines.append(f"- **{v['hypothesis_id']}**: {v['reason']}")
        lines.append("")

    # Falsified mechanisms
    falsified = learnings.get("falsified_mechanisms", [])
    if falsified:
        lines.append("### Falsified Mechanisms (stop exploring)")
        for f in falsified:
            lines.append(f"- **{f['hypothesis_id']}**: {f['reason']}")
        lines.append("")

    # Dead zones
    dead = learnings.get("dead_zones", [])
    if dead:
        lines.append("### Dead Zones (exhausted directions)")
        for d in dead:
            lines.append(f"- **{d['hypothesis_id']}**: {d['reason']}")
        lines.append("")

    # Family lessons
    family_lessons = learnings.get("family_lessons", {})
    if family_lessons:
        lines.append("### Family Lessons")
        for family, lessons in family_lessons.items():
            lines.append(f"**{family}:**")
            for lesson in lessons[:3]:
                lines.append(f"  - {lesson}")
        lines.append("")

    # Next actions
    lines.append("### Recommended Next Actions")
    for action in actions:
        if action["action_type"] == "no_action":
            lines.append(f"- {action['detail']}")
            continue
        prio_icon = {"high": "[!!]", "medium": "[!]", "low": "[-]"}.get(action["priority"], "[-]")
        lines.append(f"- {prio_icon} **{action['action_type']}** → {action['detail']}")
    lines.append("")

    return "\n".join(lines)
