"""Alpha-card-to-task converter — turns alpha cards into constrained strategy tasks.

Each card category maps to a specific conversion strategy that determines:
  - what the task objective is
  - what preservation constraints apply
  - what changes are allowed / forbidden
  - what the evaluation criteria are
"""
from __future__ import annotations

from typing import Any

from synthesis.task import StrategyTask, TaskCounter
from synthesis.parents import select_parent


# ---------------------------------------------------------------------------
# Preservation constraint templates
# ---------------------------------------------------------------------------

# Product-level preservation: when task only targets one product
_PRESERVE_EMERALDS = [
    "Keep all EMERALDS quoting parameters unchanged",
    "Do not modify EMERALDS spread widths, take widths, or skew logic",
    "EMERALDS position limits and risk controls must remain identical",
]

_PRESERVE_TOMATOES = [
    "Keep all TOMATOES quoting parameters unchanged",
    "Do not modify TOMATOES spread widths, take widths, or skew logic",
    "TOMATOES position limits and risk controls must remain identical",
]

# Structure-level preservation
_PRESERVE_MAKER_STRUCTURE = [
    "Preserve maker-heavy quoting structure and passive fill approach",
    "Do not reduce quote widths below parent values",
    "Do not increase total aggressiveness globally",
]

_PRESERVE_RISK_CONTROLS = [
    "Do not disable or weaken position limits",
    "Do not remove drawdown protection",
    "Do not increase maximum position beyond parent limits",
]

_PRESERVE_EXECUTION_ORDER = [
    "Keep execution order (make-first vs take-first) unchanged",
    "Do not alter fill priority logic",
]


def _infer_preservation(
    card: dict[str, Any],
    parent: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    """Infer preservation, allowed, and forbidden changes from card + parent.

    Returns (preservation, allowed_changes, forbidden_changes).
    """
    products = card.get("products", ["EMERALDS", "TOMATOES"])
    category = card.get("category", "")
    regime = card.get("regime_definition", {})
    role = regime.get("role", "")
    parent_family = parent.get("parent_family", "")

    preservation = list(_PRESERVE_RISK_CONTROLS)
    allowed = []
    forbidden = []

    # Product scope → preserve the other product
    if products == ["EMERALDS"]:
        preservation.extend(_PRESERVE_TOMATOES)
        allowed.append("EMERALDS quoting parameters")
        allowed.append("EMERALDS spread/take widths under targeted regime only")
    elif products == ["TOMATOES"]:
        preservation.extend(_PRESERVE_EMERALDS)
        allowed.append("TOMATOES quoting parameters")
        allowed.append("TOMATOES spread/take widths under targeted regime only")
    else:
        allowed.append("Parameters for both products under targeted regime")
        forbidden.append("Do not change both products simultaneously in untargeted regimes")

    # Category-specific constraints
    if category == "regime_edge":
        preservation.extend(_PRESERVE_EXECUTION_ORDER)
        allowed.append("Regime-gated parameter adjustments for target regime")
        allowed.append("Conditional aggressiveness changes within the target regime only")
        forbidden.append("Do not change behavior outside the identified regime")

    elif category == "role_mismatch":
        allowed.append("Maker/taker balance under the identified regime")
        allowed.append("Conditional role selection (make vs take) in target regime")
        forbidden.append("Do not shift global maker/taker ratio")
        forbidden.append("Do not change role balance in non-targeted regimes")

    elif category == "danger_refinement":
        preservation.extend(_PRESERVE_MAKER_STRUCTURE)
        allowed.append("Danger zone detection thresholds")
        allowed.append("Regime-gated exposure reduction in target regime")
        forbidden.append("Do not broaden danger zone beyond the evidence")
        forbidden.append("Do not remove activity in regimes that are profitable")

    elif category == "winner_trait":
        allowed.append("Parameters that move metrics toward winner profile")
        allowed.append("Fill quality and passive fill rate tuning")
        forbidden.append("Do not sacrifice Sharpe for higher mean PnL")

    elif category == "inventory_exploit":
        allowed.append("Inventory skew parameters")
        allowed.append("Position-dependent quoting adjustments")
        forbidden.append("Do not change position limits themselves")
        forbidden.append("Do not invert inventory management philosophy")

    elif category == "bot_weakness":
        allowed.append("Reaction logic to detected bot patterns")
        allowed.append("Conditional timing adjustments")
        forbidden.append("Do not rely on patterns that may be noise")

    # Parent-family-specific preservation
    if "maker" in parent_family:
        preservation.extend(_PRESERVE_MAKER_STRUCTURE)

    return preservation, allowed, forbidden


# ---------------------------------------------------------------------------
# Category-specific conversion logic
# ---------------------------------------------------------------------------

def _convert_regime_edge(
    card: dict[str, Any],
    parent: dict[str, Any],
    counter: TaskCounter,
) -> StrategyTask:
    """Convert a regime_edge card to an exploit task."""
    products = card.get("products", [])
    regime = card.get("regime_definition", {})
    evidence = card.get("evidence", {})
    product_str = ", ".join(products)

    regime_mean = evidence.get("regime_mean", 0)
    baseline_mean = evidence.get("baseline_mean", 0)
    diff = evidence.get("difference", regime_mean - baseline_mean)

    preservation, allowed, forbidden = _infer_preservation(card, parent)

    return StrategyTask(
        task_id=counter.next_id(),
        title=f"Exploit {product_str} edge in {_regime_str(regime)}",
        task_type="exploit",
        source_card_id=card.get("card_id", "?"),
        source_card_title=card.get("title", ""),
        product_scope=products,
        regime_targeted=regime,
        exploit_objective=(
            f"Increase activity on {product_str} during {_regime_str(regime)} "
            f"to capture the +{diff:.1f} edge above baseline."
        ),
        expected_mechanism=(
            f"When {_regime_str(regime)} holds, the environment provides "
            f"mean edge {regime_mean:.2f} vs baseline {baseline_mean:.2f}. "
            f"Increasing exposure in this regime should capture more of this edge."
        ),
        main_risk=(
            f"The regime may be correlated with another condition that actually "
            f"drives the edge. Increased aggressiveness could attract adverse fills."
        ),
        parent_id=parent.get("parent_id", "?"),
        parent_family=parent.get("parent_family", "?"),
        parent_rationale=parent.get("rationale", ""),
        preservation=preservation,
        allowed_changes=allowed,
        forbidden_changes=forbidden,
        evaluation_criteria=[
            f"fill_vs_fair in {_regime_str(regime)} improves or stays above {regime_mean:.1f}",
            "Overall PnL does not decrease vs parent",
            "Fill count in target regime increases",
        ],
        success_metric="pnl_mean",
        success_threshold=f"PnL >= parent ({parent.get('pnl_mean', 0):.0f})",
        confidence=card.get("confidence", "medium"),
        priority=_priority_from_card(card),
    )


def _convert_role_mismatch(
    card: dict[str, Any],
    parent: dict[str, Any],
    counter: TaskCounter,
) -> StrategyTask:
    products = card.get("products", [])
    regime = card.get("regime_definition", {})
    evidence = card.get("evidence", {})
    product_str = ", ".join(products)

    maker_mean = evidence.get("maker_mean", 0)
    taker_mean = evidence.get("taker_mean", 0)
    winner_role = "maker" if maker_mean > taker_mean else "taker"
    loser_role = "taker" if winner_role == "maker" else "maker"

    preservation, allowed, forbidden = _infer_preservation(card, parent)

    return StrategyTask(
        task_id=counter.next_id(),
        title=f"Shift {product_str} to {winner_role} in {_regime_str(regime)}",
        task_type="exploit",
        source_card_id=card.get("card_id", "?"),
        source_card_title=card.get("title", ""),
        product_scope=products,
        regime_targeted=regime,
        exploit_objective=(
            f"Shift {product_str} strategy toward {winner_role} activity "
            f"during {_regime_str(regime)}, reducing {loser_role} fills that "
            f"have edge {min(maker_mean, taker_mean):.2f}."
        ),
        expected_mechanism=(
            f"{winner_role.capitalize()} fills have edge {max(maker_mean, taker_mean):.2f} "
            f"vs {loser_role} edge {min(maker_mean, taker_mean):.2f} in this regime. "
            f"Shifting role balance should improve average fill quality."
        ),
        main_risk=(
            f"Reducing {loser_role} activity may miss fills that are valuable "
            f"for other reasons (inventory management, position exits)."
        ),
        parent_id=parent.get("parent_id", "?"),
        parent_family=parent.get("parent_family", "?"),
        parent_rationale=parent.get("rationale", ""),
        preservation=preservation,
        allowed_changes=allowed,
        forbidden_changes=forbidden,
        evaluation_criteria=[
            f"{winner_role} fill fraction in {_regime_str(regime)} increases",
            f"Average fill_vs_fair in {_regime_str(regime)} improves",
            "Overall PnL does not decrease vs parent",
        ],
        success_metric="fill_vs_fair",
        success_threshold=f"fill_vs_fair improves in target regime",
        confidence=card.get("confidence", "medium"),
        priority=_priority_from_card(card),
    )


def _convert_danger_refinement(
    card: dict[str, Any],
    parent: dict[str, Any],
    counter: TaskCounter,
) -> StrategyTask:
    products = card.get("products", [])
    regime = card.get("regime_definition", {})
    evidence = card.get("evidence", {})
    product_str = ", ".join(products)

    regime_mean = evidence.get("regime_mean", 0)
    baseline_mean = evidence.get("baseline_mean", 0)

    preservation, allowed, forbidden = _infer_preservation(card, parent)

    return StrategyTask(
        task_id=counter.next_id(),
        title=f"Reduce {product_str} exposure in {_regime_str(regime)}",
        task_type="defend",
        source_card_id=card.get("card_id", "?"),
        source_card_title=card.get("title", ""),
        product_scope=products,
        regime_targeted=regime,
        exploit_objective=(
            f"Reduce or gate activity on {product_str} during {_regime_str(regime)} "
            f"where edge is {regime_mean:.2f} vs baseline {baseline_mean:.2f}."
        ),
        expected_mechanism=(
            f"The {_regime_str(regime)} state causes fill quality to drop "
            f"significantly. Reducing exposure in this narrow regime should "
            f"remove the drag without affecting profitable regimes."
        ),
        main_risk=(
            f"The danger regime detection may be too aggressive, "
            f"causing the strategy to miss fills that are marginally profitable."
        ),
        parent_id=parent.get("parent_id", "?"),
        parent_family=parent.get("parent_family", "?"),
        parent_rationale=parent.get("rationale", ""),
        preservation=preservation,
        allowed_changes=allowed,
        forbidden_changes=forbidden,
        evaluation_criteria=[
            f"Fill count in {_regime_str(regime)} decreases",
            f"Average fill_vs_fair outside target regime stays stable",
            "Overall PnL improves or stays within 5% of parent",
        ],
        success_metric="pnl_mean",
        success_threshold=f"PnL >= {parent.get('pnl_mean', 0) * 0.95:.0f} (95% of parent)",
        confidence=card.get("confidence", "medium"),
        priority=_priority_from_card(card),
    )


def _convert_winner_trait(
    card: dict[str, Any],
    parent: dict[str, Any],
    counter: TaskCounter,
) -> StrategyTask:
    products = card.get("products", [])
    regime = card.get("regime_definition", {})
    evidence = card.get("evidence", {})
    product_str = ", ".join(products)

    metric_name = regime.get("metric", "unknown_metric")
    winner_mean = evidence.get("winner_mean", 0)
    effect_size = evidence.get("effect_size", 0)

    preservation, allowed, forbidden = _infer_preservation(card, parent)

    return StrategyTask(
        task_id=counter.next_id(),
        title=f"Match winner profile: {metric_name}",
        task_type="exploit",
        source_card_id=card.get("card_id", "?"),
        source_card_title=card.get("title", ""),
        product_scope=products,
        regime_targeted=regime,
        exploit_objective=(
            f"Adjust strategy to move {metric_name} toward the promoted "
            f"candidate mean of {winner_mean:.2f} (effect size {effect_size:.1f})."
        ),
        expected_mechanism=(
            f"Promoted candidates have significantly different {metric_name}. "
            f"Matching this profile should move closer to the frontier."
        ),
        main_risk=(
            f"The trait may be a consequence of winning, not a cause. "
            f"Optimizing for it directly could be misguided."
        ),
        parent_id=parent.get("parent_id", "?"),
        parent_family=parent.get("parent_family", "?"),
        parent_rationale=parent.get("rationale", ""),
        preservation=preservation,
        allowed_changes=allowed,
        forbidden_changes=forbidden,
        evaluation_criteria=[
            f"{metric_name} moves toward {winner_mean:.2f}",
            "Overall PnL improves or stays stable",
            "Sharpe does not decrease significantly",
        ],
        success_metric=metric_name,
        success_threshold=f"{metric_name} >= {winner_mean * 0.8:.2f}",
        confidence=card.get("confidence", "medium"),
        priority=_priority_from_card(card),
        warnings=["Winner traits may be effects not causes — validate with controls"],
    )


def _convert_inventory_exploit(
    card: dict[str, Any],
    parent: dict[str, Any],
    counter: TaskCounter,
) -> StrategyTask:
    products = card.get("products", [])
    regime = card.get("regime_definition", {})
    evidence = card.get("evidence", {})
    product_str = ", ".join(products)

    preservation, allowed, forbidden = _infer_preservation(card, parent)

    return StrategyTask(
        task_id=counter.next_id(),
        title=f"Exploit {product_str} position asymmetry",
        task_type="exploit",
        source_card_id=card.get("card_id", "?"),
        source_card_title=card.get("title", ""),
        product_scope=products,
        regime_targeted=regime,
        exploit_objective=(
            f"Adjust {product_str} inventory management to exploit the "
            f"position asymmetry identified in {_regime_str(regime)}."
        ),
        expected_mechanism=(
            f"Fill quality varies by position state. Skewing quotes or "
            f"adjusting take widths based on current position should capture "
            f"more edge from favorable position states."
        ),
        main_risk=(
            f"Position-dependent behavior may reduce fill rate in some states, "
            f"netting out the edge gain."
        ),
        parent_id=parent.get("parent_id", "?"),
        parent_family=parent.get("parent_family", "?"),
        parent_rationale=parent.get("rationale", ""),
        preservation=preservation,
        allowed_changes=allowed,
        forbidden_changes=forbidden,
        evaluation_criteria=[
            "fill_vs_fair improves in favorable position states",
            "Overall fill count does not drop more than 10%",
            "PnL improves or stays stable",
        ],
        success_metric="pnl_mean",
        success_threshold=f"PnL >= parent ({parent.get('pnl_mean', 0):.0f})",
        confidence=card.get("confidence", "medium"),
        priority=_priority_from_card(card),
    )


def _convert_bot_weakness(
    card: dict[str, Any],
    parent: dict[str, Any],
    counter: TaskCounter,
) -> StrategyTask:
    products = card.get("products", [])
    regime = card.get("regime_definition", {})
    product_str = ", ".join(products)

    preservation, allowed, forbidden = _infer_preservation(card, parent)

    return StrategyTask(
        task_id=counter.next_id(),
        title=f"Exploit bot pattern on {product_str}",
        task_type="exploit",
        source_card_id=card.get("card_id", "?"),
        source_card_title=card.get("title", ""),
        product_scope=products,
        regime_targeted=regime,
        exploit_objective=(
            f"Add targeted logic to exploit the identified bot behavior "
            f"pattern on {product_str} under {_regime_str(regime)}."
        ),
        expected_mechanism=(
            f"Bots exhibit a predictable pattern under the identified conditions. "
            f"Adding specific reaction logic should capture additional edge."
        ),
        main_risk=(
            f"Bot behavior may be an artifact of the simulator, not real. "
            f"Over-fitting to bot patterns may not transfer to the official server."
        ),
        parent_id=parent.get("parent_id", "?"),
        parent_family=parent.get("parent_family", "?"),
        parent_rationale=parent.get("rationale", ""),
        preservation=preservation,
        allowed_changes=allowed,
        forbidden_changes=forbidden,
        evaluation_criteria=[
            "PnL improves in target regime",
            "Fill quality improves under identified conditions",
            "No regression in non-targeted regimes",
        ],
        success_metric="pnl_mean",
        success_threshold=f"PnL >= parent ({parent.get('pnl_mean', 0):.0f})",
        confidence=card.get("confidence", "low"),
        priority="low",
        warnings=["Bot weakness patterns may not transfer to official server"],
    )


# Dispatcher
_CONVERTERS = {
    "regime_edge": _convert_regime_edge,
    "role_mismatch": _convert_role_mismatch,
    "danger_refinement": _convert_danger_refinement,
    "winner_trait": _convert_winner_trait,
    "inventory_exploit": _convert_inventory_exploit,
    "bot_weakness": _convert_bot_weakness,
}


def convert_card_to_task(
    card: dict[str, Any],
    parents: list[dict[str, Any]],
    counter: TaskCounter,
) -> StrategyTask:
    """Convert a single alpha card into a strategy task.

    Selects the best parent, infers constraints, and uses the
    category-specific converter.
    """
    category = card.get("category", "regime_edge")
    converter = _CONVERTERS.get(category, _convert_regime_edge)

    parent_selection = select_parent(parents, card)

    return converter(card, parent_selection, counter)


def convert_cards_to_tasks(
    cards: list[dict[str, Any]],
    parents: list[dict[str, Any]],
    max_tasks: int | None = None,
) -> list[StrategyTask]:
    """Convert a list of alpha cards into strategy tasks.

    Cards are processed in order (assumed pre-ranked by strength).
    """
    counter = TaskCounter()
    tasks = []

    for card in cards:
        if max_tasks and len(tasks) >= max_tasks:
            break
        task = convert_card_to_task(card, parents, counter)
        tasks.append(task)

    return tasks


# ---------------------------------------------------------------------------
# Control task generation
# ---------------------------------------------------------------------------

def generate_control_tasks(
    parent: dict[str, Any],
    counter: TaskCounter,
) -> list[StrategyTask]:
    """Generate control tasks for a given parent strategy.

    Creates:
      1. Near-parent control (minimal cosmetic change)
      2. Calibration check (re-run parent unchanged)
    """
    ps = parent.get("packet_short", parent)
    parent_id = parent.get("_case_id", parent.get("case_id", "?"))
    parent_family = parent.get("_family", parent.get("family", "?"))
    pnl_mean = ps.get("pnl", {}).get("mean", 0)

    controls = []

    # 1. Calibration check
    controls.append(StrategyTask(
        task_id=counter.next_id(),
        title=f"Calibration: re-run {parent_id} as baseline",
        task_type="calibration_check",
        source_card_id="baseline",
        source_card_title="Calibration control",
        product_scope=["EMERALDS", "TOMATOES"],
        regime_targeted={},
        exploit_objective=(
            f"Re-run parent {parent_id} unchanged to establish baseline PnL "
            f"and verify reproducibility."
        ),
        expected_mechanism="No change — confirms simulator consistency.",
        main_risk="None — this is a control.",
        parent_id=parent_id,
        parent_family=parent_family,
        parent_rationale="Exact parent for baseline comparison",
        preservation=["Everything — this is the unmodified parent"],
        allowed_changes=["Nothing"],
        forbidden_changes=["All changes forbidden — run parent as-is"],
        evaluation_criteria=[
            f"PnL should be near {pnl_mean:.0f} (within 10%)",
            "Sharpe should match parent within noise",
        ],
        success_metric="pnl_mean",
        success_threshold=f"PnL within 10% of {pnl_mean:.0f}",
        confidence="high",
        priority="high",
    ))

    # 2. Near-parent control (add a comment, change nothing functional)
    controls.append(StrategyTask(
        task_id=counter.next_id(),
        title=f"Near-parent control: {parent_id} with cosmetic change only",
        task_type="near_parent_control",
        source_card_id="control",
        source_card_title="Near-parent control",
        product_scope=["EMERALDS", "TOMATOES"],
        regime_targeted={},
        exploit_objective=(
            f"Apply one trivial non-functional change to {parent_id} "
            f"(e.g., reorder imports, rename internal variable) to verify "
            f"that perceived gains in exploit tasks come from real changes."
        ),
        expected_mechanism="PnL should match parent — if it differs, noise is high.",
        main_risk="None — this is a control.",
        parent_id=parent_id,
        parent_family=parent_family,
        parent_rationale="Same parent, minimal change for noise estimation",
        preservation=["All functional behavior"],
        allowed_changes=["Cosmetic-only: comments, variable names, import order"],
        forbidden_changes=["Any functional parameter change"],
        evaluation_criteria=[
            f"PnL should be within 2% of {pnl_mean:.0f}",
            "Any deviation indicates noise floor",
        ],
        success_metric="pnl_mean",
        success_threshold=f"PnL within 2% of {pnl_mean:.0f}",
        confidence="high",
        priority="medium",
    ))

    return controls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _regime_str(regime: dict[str, Any]) -> str:
    """Format a regime definition as a readable string."""
    parts = []
    for k, v in regime.items():
        if k in ("product", "role", "comparison", "probe_id", "probe_family", "metric"):
            continue
        parts.append(f"{k}={v}")
    return ", ".join(parts) if parts else "general"


def _priority_from_card(card: dict[str, Any]) -> str:
    """Derive task priority from card confidence and strength."""
    confidence = card.get("confidence", "low")
    strength = card.get("strength", 0)

    if confidence == "high" and strength > 5:
        return "critical"
    if confidence == "high" or (confidence == "medium" and strength > 3):
        return "high"
    if confidence == "medium":
        return "medium"
    return "low"
