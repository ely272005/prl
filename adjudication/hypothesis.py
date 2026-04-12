"""Hypothesis adjudication — judges the alpha card hypothesis, not just the candidate.

For each task/alpha card, answers:
  - did the task validate the intended mechanism?
  - did it help mean? Sharpe? both?
  - did it only help one product?
  - did it simply add aggression?
  - did it damage the preserved base?
  - was the result informative even if PnL didn't improve?

Produces a structured hypothesis verdict with explicit outcome labels.
"""
from __future__ import annotations

from typing import Any

from adjudication.comparison import _extract


HYPOTHESIS_OUTCOMES = (
    "validated",            # mechanism worked as predicted
    "partially_validated",  # mechanism worked but with side effects
    "falsified",            # mechanism clearly didn't work or was harmful
    "inconclusive",         # not enough signal to tell
    "informative_failure",  # failed but taught something specific
)


def adjudicate_hypothesis(
    candidate_verdict: dict[str, Any],
    task: dict[str, Any],
    card: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Judge whether the hypothesis behind a task was validated.

    Parameters
    ----------
    candidate_verdict : dict
        Full verdict from adjudicate_candidate().
    task : dict
        The StrategyTask dict.
    card : dict, optional
        The original alpha card for richer context.
    """
    task_type = task.get("task_type", "exploit")

    # Control tasks don't test hypotheses
    if task_type in ("calibration_check", "near_parent_control"):
        return {
            "hypothesis_id": task.get("source_card_id", "?"),
            "hypothesis_title": task.get("source_card_title", "Control task"),
            "outcome": "not_applicable",
            "reason": "Control task — no hypothesis to test.",
            "checks": [],
            "lessons": [],
        }

    # Extract key values
    pnl_delta = candidate_verdict.get("pnl_delta", 0)
    sharpe_delta = candidate_verdict.get("sharpe_delta", 0)
    em_delta = candidate_verdict.get("emerald_delta", 0)
    tom_delta = candidate_verdict.get("tomato_delta", 0)
    verdict = candidate_verdict.get("verdict", "?")
    attribution = candidate_verdict.get("attribution", {})
    preservation = candidate_verdict.get("preservation_audit", {})
    suspicion = candidate_verdict.get("suspicion", {})

    product_scope = task.get("product_scope", [])
    expected_mechanism = task.get("expected_mechanism", "")
    task_category = card.get("category", "") if card else ""

    # Run checks
    checks = []

    # Check 1: Did mean improve?
    mean_improved = pnl_delta > 0
    checks.append({
        "check": "mean_improved",
        "passed": mean_improved,
        "detail": f"PnL delta: {pnl_delta:+.0f}",
    })

    # Check 2: Did Sharpe improve?
    sharpe_improved = sharpe_delta > 0
    checks.append({
        "check": "sharpe_improved",
        "passed": sharpe_improved,
        "detail": f"Sharpe delta: {sharpe_delta:+.2f}",
    })

    # Check 3: Was the intended product helped?
    intended_helped = _check_intended_product(product_scope, em_delta, tom_delta)
    checks.append({
        "check": "intended_product_helped",
        "passed": intended_helped,
        "detail": f"EMERALDS: {em_delta:+.0f}, TOMATOES: {tom_delta:+.0f}",
    })

    # Check 4: Was the preserved base damaged?
    base_damaged = preservation.get("verdict") in ("violated", "suspect")
    checks.append({
        "check": "preserved_base_intact",
        "passed": not base_damaged,
        "detail": preservation.get("reason", "No violations"),
    })

    # Check 5: Was gain just from aggression?
    just_aggression = _is_just_aggression(attribution, suspicion)
    checks.append({
        "check": "not_just_aggression",
        "passed": not just_aggression,
        "detail": _aggression_detail(attribution),
    })

    # Check 6: Did the mechanism match expectations?
    mechanism_matched = _check_mechanism_match(task_category, attribution, pnl_delta, sharpe_delta)
    checks.append({
        "check": "mechanism_matched",
        "passed": mechanism_matched,
        "detail": f"Dominant: {attribution.get('dominant_mechanism', 'unknown')}",
    })

    # Derive outcome
    outcome, reason = _derive_outcome(checks, pnl_delta, sharpe_delta, verdict)

    # Extract lessons
    lessons = _extract_lessons(
        checks, task, card, candidate_verdict, attribution,
    )

    return {
        "hypothesis_id": task.get("source_card_id", "?"),
        "hypothesis_title": task.get("source_card_title", "?"),
        "task_id": task.get("task_id", "?"),
        "task_type": task_type,
        "expected_mechanism": expected_mechanism,
        "outcome": outcome,
        "reason": reason,
        "checks": checks,
        "lessons": lessons,
        "mean_helped": mean_improved,
        "sharpe_helped": sharpe_improved,
        "single_product_only": _is_single_product_gain(em_delta, tom_delta, pnl_delta),
        "just_aggression": just_aggression,
        "base_damaged": base_damaged,
        "informative": outcome != "inconclusive",
    }


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------

def _check_intended_product(
    product_scope: list[str],
    em_delta: float,
    tom_delta: float,
) -> bool:
    if product_scope == ["EMERALDS"]:
        return em_delta > 0
    if product_scope == ["TOMATOES"]:
        return tom_delta > 0
    # Both products in scope — at least one should improve
    return em_delta > 0 or tom_delta > 0


def _is_just_aggression(
    attribution: dict[str, Any],
    suspicion: dict[str, Any],
) -> bool:
    dominant = attribution.get("dominant_mechanism", "")
    if dominant == "aggressiveness_change":
        return True
    for flag in suspicion.get("flags", []):
        if flag.get("flag") == "aggression_driven_gain":
            return True
    return False


def _aggression_detail(attribution: dict[str, Any]) -> str:
    for attr in attribution.get("attributions", []):
        if attr.get("mechanism") == "aggressiveness_change":
            return attr.get("description", "Aggressiveness changed")
    return "No significant aggressiveness change"


def _check_mechanism_match(
    task_category: str,
    attribution: dict[str, Any],
    pnl_delta: float,
    sharpe_delta: float,
) -> bool:
    """Check if the dominant mechanism matches what the task category expected."""
    dominant = attribution.get("dominant_mechanism", "")

    if task_category == "regime_edge":
        # Expected: product-level gain in specific regime
        return dominant in ("product_shift", "fill_quality_change", "sharpe_decomposition")

    if task_category == "role_mismatch":
        # Expected: role shift (maker/taker rebalance)
        return dominant in ("role_shift", "fill_quality_change")

    if task_category == "danger_refinement":
        # Expected: risk reduction, possibly lower mean but better Sharpe
        return dominant in ("risk_change", "sharpe_decomposition", "aggressiveness_change")

    if task_category == "winner_trait":
        # Expected: quality improvement
        return dominant in ("fill_quality_change", "role_shift", "sharpe_decomposition")

    if task_category == "inventory_exploit":
        # Expected: product-level shift from position management
        return dominant in ("product_shift", "fill_quality_change")

    # Default: any mechanism counts
    return pnl_delta > 0 or sharpe_delta > 0


def _is_single_product_gain(em_delta: float, tom_delta: float, total_delta: float) -> bool:
    if total_delta <= 0:
        return False
    if em_delta > 0 and tom_delta <= 0:
        return True
    if tom_delta > 0 and em_delta <= 0:
        return True
    return False


# ---------------------------------------------------------------------------
# Outcome derivation
# ---------------------------------------------------------------------------

def _derive_outcome(
    checks: list[dict],
    pnl_delta: float,
    sharpe_delta: float,
    verdict: str,
) -> tuple[str, str]:
    """Derive hypothesis outcome from check results."""
    passed = {c["check"]: c["passed"] for c in checks}

    # All checks pass → validated
    all_passed = all(v for v in passed.values())
    if all_passed and (pnl_delta > 0 or sharpe_delta > 0):
        return "validated", (
            f"Mechanism worked as predicted. "
            f"PnL {pnl_delta:+.0f}, Sharpe {sharpe_delta:+.2f}. "
            f"All checks passed."
        )

    # Mean or Sharpe improved but with issues → partially validated
    if (passed.get("mean_improved") or passed.get("sharpe_improved")) and \
       not passed.get("preserved_base_intact"):
        return "partially_validated", (
            f"Mechanism showed effect but damaged preserved base. "
            f"Result is contaminated."
        )

    if (passed.get("mean_improved") or passed.get("sharpe_improved")) and \
       passed.get("not_just_aggression", True) is False:
        return "partially_validated", (
            f"Improvement detected but appears driven by aggression, "
            f"not the intended mechanism."
        )

    if passed.get("mean_improved") and not passed.get("sharpe_improved"):
        return "partially_validated", (
            f"Mean improved ({pnl_delta:+.0f}) but Sharpe degraded ({sharpe_delta:+.2f}). "
            f"Mechanism works but adds variance."
        )

    if passed.get("sharpe_improved") and not passed.get("mean_improved"):
        return "partially_validated", (
            f"Sharpe improved ({sharpe_delta:+.2f}) but mean dropped ({pnl_delta:+.0f}). "
            f"Mechanism cleans the signal but reduces total output."
        )

    # Clear failure with informative signal
    if pnl_delta < -100 and not passed.get("mean_improved"):
        return "informative_failure", (
            f"Mechanism failed clearly (PnL {pnl_delta:+.0f}). "
            f"This direction is likely dead — useful to know."
        )

    # Clean rejection
    if verdict == "reject":
        return "falsified", (
            f"Hypothesis falsified. "
            f"PnL {pnl_delta:+.0f}, Sharpe {sharpe_delta:+.2f}. "
            f"Mechanism did not produce the expected effect."
        )

    # Not enough signal
    if abs(pnl_delta) < 50 and abs(sharpe_delta) < 0.5:
        return "inconclusive", (
            f"No clear signal. PnL {pnl_delta:+.0f}, Sharpe {sharpe_delta:+.2f}. "
            f"May need different parent or more extreme change."
        )

    # Default to inconclusive
    return "inconclusive", (
        f"Mixed results. PnL {pnl_delta:+.0f}, Sharpe {sharpe_delta:+.2f}. "
        f"Hypothesis neither clearly validated nor falsified."
    )


# ---------------------------------------------------------------------------
# Lesson extraction
# ---------------------------------------------------------------------------

def _extract_lessons(
    checks: list[dict],
    task: dict[str, Any],
    card: dict[str, Any] | None,
    candidate_verdict: dict[str, Any],
    attribution: dict[str, Any],
) -> list[str]:
    """Extract specific, actionable lessons from the hypothesis test."""
    lessons = []

    pnl_delta = candidate_verdict.get("pnl_delta", 0)
    sharpe_delta = candidate_verdict.get("sharpe_delta", 0)
    em_delta = candidate_verdict.get("emerald_delta", 0)
    tom_delta = candidate_verdict.get("tomato_delta", 0)
    dominant = attribution.get("dominant_mechanism", "")
    parent_family = task.get("parent_family", "")

    # Product-specific lessons
    if em_delta > 0 and tom_delta < 0:
        lessons.append(
            f"Change helped EMERALDS ({em_delta:+.0f}) but hurt TOMATOES ({tom_delta:+.0f}). "
            f"Consider product-gated application."
        )
    elif tom_delta > 0 and em_delta < 0:
        lessons.append(
            f"Change helped TOMATOES ({tom_delta:+.0f}) but hurt EMERALDS ({em_delta:+.0f}). "
            f"Consider product-gated application."
        )

    # Mechanism lessons
    if dominant == "role_shift":
        role_detail = attribution.get("role_attribution", {})
        direction = role_detail.get("detail", {}).get("direction", "")
        if pnl_delta > 0:
            lessons.append(f"Shifting {direction} was profitable on {parent_family} bases.")
        else:
            lessons.append(f"Shifting {direction} did not help on {parent_family} bases.")

    if dominant == "aggressiveness_change":
        if pnl_delta > 0:
            lessons.append(
                f"Gain came from aggression increase, not mechanism quality. "
                f"Likely hits diminishing returns quickly."
            )
        else:
            lessons.append("More aggressive trading was counterproductive.")

    # Sharpe vs mean trade-off
    if sharpe_delta > 1.0 and pnl_delta < 0:
        lessons.append("Lower mean with better Sharpe suggests risk reduction works. Consider as a quality layer.")
    elif sharpe_delta < -1.0 and pnl_delta > 0:
        lessons.append("Higher mean with worse Sharpe suggests the gain is volatile. Not reliable.")

    # Falsification lessons
    if pnl_delta < -200:
        card_title = card.get("title", task.get("source_card_title", "")) if card else task.get("source_card_title", "")
        lessons.append(f"Direction '{card_title}' is clearly harmful on {parent_family} bases. Stop exploring.")

    return lessons
