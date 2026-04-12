"""StrategyTask — structured artifact representing a targeted generation task.

Each task is tied to a concrete alpha card (or labeled as a control),
specifies exactly what can change and what must be preserved,
and carries enough context for downstream prompt generation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


TASK_TYPES = (
    "exploit",              # Attack a specific alpha card opportunity
    "defend",               # Protect against a danger refinement card
    "near_parent_control",  # Minimal-change baseline for comparison
    "mechanism_isolation",  # Change one thing to isolate an effect
    "calibration_check",    # Re-run parent to verify baseline
)

PRIORITY_LEVELS = ("critical", "high", "medium", "low")


@dataclass
class StrategyTask:
    """A concrete, constrained strategy generation task."""

    task_id: str
    title: str
    task_type: str                          # One of TASK_TYPES

    # Provenance
    source_card_id: str                     # Alpha card ID, or "control" / "baseline"
    source_card_title: str

    # Scope
    product_scope: list[str]                # Which products this task modifies
    regime_targeted: dict[str, Any]         # The specific regime being attacked

    # Objective
    exploit_objective: str                  # What this task is trying to achieve
    expected_mechanism: str                 # How the exploit should work mechanically
    main_risk: str                          # What could go wrong

    # Parent strategy
    parent_id: str                          # Case ID of the parent strategy
    parent_family: str                      # Family label of the parent
    parent_rationale: str                   # Why this parent was chosen

    # Constraints
    preservation: list[str]                 # What must NOT change
    allowed_changes: list[str]              # What CAN change
    forbidden_changes: list[str]            # What must NOT be touched

    # Evaluation
    evaluation_criteria: list[str]          # How to judge success
    success_metric: str                     # Primary metric to check
    success_threshold: str                  # Concrete threshold

    # Quality
    confidence: str                         # high / medium / low
    priority: str                           # critical / high / medium / low
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "task_type": self.task_type,
            "source_card_id": self.source_card_id,
            "source_card_title": self.source_card_title,
            "product_scope": self.product_scope,
            "regime_targeted": _sanitize(self.regime_targeted),
            "exploit_objective": self.exploit_objective,
            "expected_mechanism": self.expected_mechanism,
            "main_risk": self.main_risk,
            "parent_id": self.parent_id,
            "parent_family": self.parent_family,
            "parent_rationale": self.parent_rationale,
            "preservation": self.preservation,
            "allowed_changes": self.allowed_changes,
            "forbidden_changes": self.forbidden_changes,
            "evaluation_criteria": self.evaluation_criteria,
            "success_metric": self.success_metric,
            "success_threshold": self.success_threshold,
            "confidence": self.confidence,
            "priority": self.priority,
            "warnings": self.warnings,
        }


@dataclass
class ExperimentBatch:
    """A coherent set of tasks meant to be run together."""

    batch_id: str
    title: str
    rationale: str
    mode: str                               # e.g. "top_priority", "single_parent", "product_focus"
    tasks: list[StrategyTask]
    controls: list[StrategyTask]            # Control tasks for comparison
    diversity_notes: str
    overlap_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "title": self.title,
            "rationale": self.rationale,
            "mode": self.mode,
            "task_count": len(self.tasks),
            "control_count": len(self.controls),
            "tasks": [t.to_dict() for t in self.tasks],
            "controls": [t.to_dict() for t in self.controls],
            "diversity_notes": self.diversity_notes,
            "overlap_warnings": self.overlap_warnings,
        }


class TaskCounter:
    """Assigns sequential task IDs."""

    def __init__(self, prefix: str = "T") -> None:
        self._prefix = prefix
        self._count = 0

    def next_id(self) -> str:
        self._count += 1
        return f"{self._prefix}{self._count:03d}"


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
