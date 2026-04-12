"""Learning write-back — extracts structured learnings from adjudicated batches.

Learnings attach to:
  - source alpha cards (hypothesis outcomes)
  - tasks (execution results)
  - parent strategies (what works on this base)
  - families (family-level patterns)

Categories:
  - validated_mechanisms: things that worked
  - falsified_mechanisms: things that didn't work
  - suspicious_directions: gains that look fake
  - promising_zones: parameter/regime areas worth exploring further
  - dead_zones: parameter/regime areas that are exhausted
  - family_lessons: what works/fails for each family
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any


def extract_batch_learnings(
    candidate_verdicts: list[dict[str, Any]],
    hypothesis_verdicts: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Extract structured learnings from an adjudicated batch.

    Returns a dict with categorized learnings.
    """
    validated: list[dict[str, Any]] = []
    falsified: list[dict[str, Any]] = []
    suspicious: list[dict[str, Any]] = []
    promising: list[dict[str, Any]] = []
    dead: list[dict[str, Any]] = []
    family_lessons: dict[str, list[str]] = defaultdict(list)
    card_lessons: dict[str, list[str]] = defaultdict(list)
    parent_lessons: dict[str, list[str]] = defaultdict(list)

    # Process hypothesis verdicts
    for hv in hypothesis_verdicts:
        outcome = hv.get("outcome", "inconclusive")
        hypothesis_id = hv.get("hypothesis_id", "?")
        task_id = hv.get("task_id", "?")
        title = hv.get("hypothesis_title", "?")
        reason = hv.get("reason", "")
        lessons = hv.get("lessons", [])

        entry = {
            "hypothesis_id": hypothesis_id,
            "task_id": task_id,
            "title": title,
            "reason": reason,
            "lessons": lessons,
        }

        if outcome == "validated":
            validated.append(entry)
            card_lessons[hypothesis_id].append(f"VALIDATED: {reason}")
        elif outcome == "falsified":
            falsified.append(entry)
            card_lessons[hypothesis_id].append(f"FALSIFIED: {reason}")
        elif outcome == "informative_failure":
            falsified.append(entry)
            card_lessons[hypothesis_id].append(f"INFORMATIVE FAILURE: {reason}")
        elif outcome == "partially_validated":
            promising.append(entry)
            card_lessons[hypothesis_id].append(f"PARTIAL: {reason}")

    # Process candidate verdicts for family/parent patterns
    for cv in candidate_verdicts:
        verdict = cv.get("verdict", "?")
        family = cv.get("family", "unknown")
        parent_id = cv.get("parent_id", "?")
        task_id = cv.get("task_id", "?")
        pnl_delta = cv.get("pnl_delta", 0)
        sharpe_delta = cv.get("sharpe_delta", 0)
        mechanism = cv.get("attribution", {}).get("dominant_mechanism", "?")

        # Family-level lessons
        if verdict in ("frontier_challenger", "escalate"):
            family_lessons[family].append(
                f"[{task_id}] Improvement via {mechanism}: PnL {pnl_delta:+.0f}, Sharpe {sharpe_delta:+.2f}"
            )
        elif verdict == "reject":
            family_lessons[family].append(
                f"[{task_id}] Rejected: PnL {pnl_delta:+.0f}, Sharpe {sharpe_delta:+.2f}"
            )

        # Parent-level lessons
        if verdict in ("frontier_challenger", "escalate"):
            parent_lessons[parent_id].append(
                f"[{task_id}] {mechanism} worked: PnL {pnl_delta:+.0f}, Sharpe {sharpe_delta:+.2f}"
            )
        elif verdict == "reject":
            parent_lessons[parent_id].append(
                f"[{task_id}] Failed on {parent_id}: PnL {pnl_delta:+.0f}"
            )

        # Suspicious
        if verdict == "suspect_simulator_gain":
            suspicious.append({
                "candidate_id": cv.get("candidate_id", "?"),
                "task_id": task_id,
                "family": family,
                "reason": cv.get("reason", ""),
                "flags": [f["flag"] for f in cv.get("suspicion", {}).get("flags", [])],
            })

    # Identify dead zones: hypotheses that failed across multiple parents
    falsified_by_card = defaultdict(list)
    for f in falsified:
        falsified_by_card[f["hypothesis_id"]].append(f)

    for card_id, failures in falsified_by_card.items():
        if len(failures) >= 2:
            dead.append({
                "hypothesis_id": card_id,
                "failure_count": len(failures),
                "reason": f"Hypothesis {card_id} failed {len(failures)} times across different parents. Direction likely exhausted.",
            })

    # Identify promising zones: partially validated hypotheses
    for p in promising:
        if p.get("lessons"):
            for lesson in p["lessons"]:
                if "product-gated" in lesson.lower() or "profitable" in lesson.lower():
                    # Already captured
                    pass

    return {
        "validated_mechanisms": validated,
        "falsified_mechanisms": falsified,
        "suspicious_directions": suspicious,
        "promising_zones": promising,
        "dead_zones": dead,
        "family_lessons": dict(family_lessons),
        "card_lessons": dict(card_lessons),
        "parent_lessons": dict(parent_lessons),
        "summary": _build_summary(validated, falsified, suspicious, promising, dead),
    }


def _build_summary(
    validated: list,
    falsified: list,
    suspicious: list,
    promising: list,
    dead: list,
) -> str:
    parts = [
        f"Validated: {len(validated)} mechanisms.",
        f"Falsified: {len(falsified)} mechanisms.",
        f"Suspicious: {len(suspicious)} candidates.",
        f"Promising zones: {len(promising)}.",
        f"Dead zones: {len(dead)}.",
    ]
    return " ".join(parts)
