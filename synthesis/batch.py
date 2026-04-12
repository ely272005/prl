"""Batch construction — groups strategy tasks into coherent experiment batches.

Supports multiple batching modes:
  - top_priority:   take the N highest-priority tasks
  - single_parent:  all tasks derived from one parent
  - product_focus:  all tasks targeting a specific product
  - balanced:       diverse mix across products, categories, and parents
  - control_attack: pair every exploit with its controls

Each batch includes controls, diversity notes, and overlap warnings.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from synthesis.task import StrategyTask, ExperimentBatch, TaskCounter
from synthesis.converter import generate_control_tasks


BATCH_MODES = (
    "top_priority",
    "single_parent",
    "product_focus",
    "balanced",
    "control_attack",
)


def _detect_overlaps(tasks: list[StrategyTask]) -> list[str]:
    """Detect tasks that overlap in scope and could interfere."""
    warnings = []

    # Check for multiple tasks targeting the same regime on the same product
    seen: dict[str, list[str]] = {}
    for t in tasks:
        regime_key = _regime_key(t.regime_targeted)
        for prod in t.product_scope:
            key = f"{prod}:{regime_key}"
            seen.setdefault(key, []).append(t.task_id)

    for key, task_ids in seen.items():
        if len(task_ids) > 1:
            warnings.append(
                f"Tasks {', '.join(task_ids)} overlap on {key} — "
                f"results may interfere"
            )

    # Check for conflicting directions (exploit vs defend on same scope)
    exploit_scopes: dict[str, list[str]] = {}
    defend_scopes: dict[str, list[str]] = {}
    for t in tasks:
        scope = f"{','.join(t.product_scope)}:{_regime_key(t.regime_targeted)}"
        if t.task_type == "exploit":
            exploit_scopes.setdefault(scope, []).append(t.task_id)
        elif t.task_type == "defend":
            defend_scopes.setdefault(scope, []).append(t.task_id)

    for scope in set(exploit_scopes) & set(defend_scopes):
        warnings.append(
            f"Conflicting exploit ({', '.join(exploit_scopes[scope])}) and "
            f"defend ({', '.join(defend_scopes[scope])}) tasks on {scope}"
        )

    return warnings


def _diversity_notes(tasks: list[StrategyTask]) -> str:
    """Summarize diversity of the batch."""
    if not tasks:
        return "Empty batch."

    categories = Counter(t.task_type for t in tasks)
    products = Counter(p for t in tasks for p in t.product_scope)
    parents = Counter(t.parent_id for t in tasks)

    lines = [
        f"Task types: {dict(categories)}",
        f"Products targeted: {dict(products)}",
        f"Parent strategies used: {len(parents)} unique ({dict(parents)})",
    ]

    # Concentration warnings
    total = len(tasks)
    for parent, count in parents.items():
        if count / total > 0.5 and total > 2:
            lines.append(f"NOTE: {parent} dominates ({count}/{total} tasks)")

    return "\n".join(lines)


def _regime_key(regime: dict[str, Any]) -> str:
    """Stable string key for a regime dict."""
    parts = []
    for k in sorted(regime.keys()):
        if k in ("product", "role", "comparison", "probe_id", "probe_family", "metric"):
            continue
        parts.append(f"{k}={regime[k]}")
    return "|".join(parts) if parts else "general"


# ---------------------------------------------------------------------------
# Batch construction modes
# ---------------------------------------------------------------------------

def build_top_priority_batch(
    tasks: list[StrategyTask],
    parents: list[dict[str, Any]],
    max_tasks: int = 8,
    batch_id: str = "B001",
) -> ExperimentBatch:
    """Build a batch from the top-priority tasks."""
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    sorted_tasks = sorted(
        tasks,
        key=lambda t: (priority_order.get(t.priority, 9), t.task_id),
    )
    selected = sorted_tasks[:max_tasks]

    controls = _collect_controls(selected, parents)

    return ExperimentBatch(
        batch_id=batch_id,
        title=f"Top {len(selected)} priority tasks",
        rationale="Selected highest-priority tasks across all categories.",
        mode="top_priority",
        tasks=selected,
        controls=controls,
        diversity_notes=_diversity_notes(selected),
        overlap_warnings=_detect_overlaps(selected),
    )


def build_single_parent_batch(
    tasks: list[StrategyTask],
    parents: list[dict[str, Any]],
    parent_id: str,
    batch_id: str = "B001",
) -> ExperimentBatch:
    """Build a batch of all tasks for a specific parent."""
    selected = [t for t in tasks if t.parent_id == parent_id]

    controls = _collect_controls_for_parent(parent_id, parents)

    return ExperimentBatch(
        batch_id=batch_id,
        title=f"All tasks for parent {parent_id}",
        rationale=f"Focused experiment: all cards applied to parent {parent_id}.",
        mode="single_parent",
        tasks=selected,
        controls=controls,
        diversity_notes=_diversity_notes(selected),
        overlap_warnings=_detect_overlaps(selected),
    )


def build_product_focus_batch(
    tasks: list[StrategyTask],
    parents: list[dict[str, Any]],
    product: str,
    max_tasks: int = 8,
    batch_id: str = "B001",
) -> ExperimentBatch:
    """Build a batch focused on a single product."""
    selected = [t for t in tasks if product in t.product_scope][:max_tasks]

    controls = _collect_controls(selected, parents)

    return ExperimentBatch(
        batch_id=batch_id,
        title=f"{product} focused batch",
        rationale=f"All tasks targeting {product} to study product-specific effects.",
        mode="product_focus",
        tasks=selected,
        controls=controls,
        diversity_notes=_diversity_notes(selected),
        overlap_warnings=_detect_overlaps(selected),
    )


def build_balanced_batch(
    tasks: list[StrategyTask],
    parents: list[dict[str, Any]],
    max_tasks: int = 8,
    batch_id: str = "B001",
) -> ExperimentBatch:
    """Build a balanced batch with diversity across categories and products."""
    # Greedy selection: pick tasks to maximize diversity
    selected: list[StrategyTask] = []
    used_categories: Counter[str] = Counter()
    used_products: Counter[str] = Counter()
    used_parents: Counter[str] = Counter()

    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    sorted_tasks = sorted(
        tasks,
        key=lambda t: (priority_order.get(t.priority, 9), t.task_id),
    )

    for task in sorted_tasks:
        if len(selected) >= max_tasks:
            break

        # Diversity penalty: prefer underrepresented slots
        cat_count = used_categories.get(task.task_type, 0)
        prod_count = max(
            (used_products.get(p, 0) for p in task.product_scope), default=0
        )
        parent_count = used_parents.get(task.parent_id, 0)

        # Skip if too concentrated (unless few tasks exist)
        max_allowed = max(2, max_tasks // 3)
        if cat_count >= max_allowed and len(sorted_tasks) > max_tasks:
            continue
        if parent_count >= max_allowed and len(sorted_tasks) > max_tasks:
            continue

        selected.append(task)
        used_categories[task.task_type] += 1
        for p in task.product_scope:
            used_products[p] += 1
        used_parents[task.parent_id] += 1

    controls = _collect_controls(selected, parents)

    return ExperimentBatch(
        batch_id=batch_id,
        title=f"Balanced batch ({len(selected)} tasks)",
        rationale="Diverse selection across categories, products, and parents.",
        mode="balanced",
        tasks=selected,
        controls=controls,
        diversity_notes=_diversity_notes(selected),
        overlap_warnings=_detect_overlaps(selected),
    )


def build_control_attack_batch(
    tasks: list[StrategyTask],
    parents: list[dict[str, Any]],
    max_tasks: int = 4,
    batch_id: str = "B001",
) -> ExperimentBatch:
    """Build a batch pairing each exploit with its controls.

    Picks the top N exploit tasks and generates 2 controls per
    unique parent used, creating a rigorous experiment structure.
    """
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    exploit_tasks = sorted(
        [t for t in tasks if t.task_type in ("exploit", "defend")],
        key=lambda t: (priority_order.get(t.priority, 9), t.task_id),
    )[:max_tasks]

    controls = _collect_controls(exploit_tasks, parents)

    return ExperimentBatch(
        batch_id=batch_id,
        title=f"Control-attack batch ({len(exploit_tasks)} tasks + {len(controls)} controls)",
        rationale=(
            "Each exploit/defend task is paired with controls for the same parent. "
            "Compare task PnL against control PnL to measure genuine signal."
        ),
        mode="control_attack",
        tasks=exploit_tasks,
        controls=controls,
        diversity_notes=_diversity_notes(exploit_tasks),
        overlap_warnings=_detect_overlaps(exploit_tasks),
    )


# ---------------------------------------------------------------------------
# Control generation helpers
# ---------------------------------------------------------------------------

def _collect_controls(
    tasks: list[StrategyTask],
    parents: list[dict[str, Any]],
) -> list[StrategyTask]:
    """Generate controls for all unique parents used in the task list."""
    counter = TaskCounter(prefix="C")
    parent_ids_used = {t.parent_id for t in tasks}

    controls = []
    for parent in parents:
        pid = parent.get("_case_id", parent.get("case_id", "?"))
        if pid in parent_ids_used:
            controls.extend(generate_control_tasks(parent, counter))

    return controls


def _collect_controls_for_parent(
    parent_id: str,
    parents: list[dict[str, Any]],
) -> list[StrategyTask]:
    """Generate controls for a specific parent."""
    counter = TaskCounter(prefix="C")
    for parent in parents:
        pid = parent.get("_case_id", parent.get("case_id", "?"))
        if pid == parent_id:
            return generate_control_tasks(parent, counter)
    return []


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

BATCH_BUILDERS = {
    "top_priority": build_top_priority_batch,
    "single_parent": build_single_parent_batch,
    "product_focus": build_product_focus_batch,
    "balanced": build_balanced_batch,
    "control_attack": build_control_attack_batch,
}


def build_batch(
    mode: str,
    tasks: list[StrategyTask],
    parents: list[dict[str, Any]],
    batch_id: str = "B001",
    **kwargs: Any,
) -> ExperimentBatch:
    """Build an experiment batch using the specified mode.

    Parameters
    ----------
    mode : str
        One of BATCH_MODES.
    tasks : list[StrategyTask]
        All available tasks.
    parents : list[dict]
        All parent strategies (needed for control generation).
    batch_id : str
        Identifier for the batch.
    **kwargs
        Mode-specific arguments (parent_id, product, max_tasks).
    """
    if mode not in BATCH_BUILDERS:
        raise ValueError(f"Unknown batch mode: {mode}. Use one of {BATCH_MODES}")

    builder = BATCH_BUILDERS[mode]

    # Build the call args — each builder has a slightly different signature
    if mode == "single_parent":
        parent_id = kwargs.get("parent_id")
        if not parent_id:
            raise ValueError("single_parent mode requires parent_id")
        return builder(tasks, parents, parent_id=parent_id, batch_id=batch_id)
    elif mode == "product_focus":
        product = kwargs.get("product")
        if not product:
            raise ValueError("product_focus mode requires product")
        return builder(
            tasks, parents,
            product=product,
            max_tasks=kwargs.get("max_tasks", 8),
            batch_id=batch_id,
        )
    else:
        return builder(
            tasks, parents,
            max_tasks=kwargs.get("max_tasks", 8),
            batch_id=batch_id,
        )
