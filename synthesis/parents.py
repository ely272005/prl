"""Parent selection — chooses the best base strategy for a given task.

Scores each candidate parent on:
  1. Family match:          does the parent's style match the task need?
  2. Quality:               PnL, Sharpe, promote status
  3. Product specialization: per-product strength alignment
  4. Damage risk:           how much existing edge is at risk from the change
  5. Frontier status:       promoted > rejected

The selector returns a ranked list with explanations.
"""
from __future__ import annotations

import math
from typing import Any


# Family affinity: which families are good parents for which task types
# Higher = better match. 0 = acceptable. Negative = bad fit.
FAMILY_AFFINITY = {
    # Task involves maker activity
    "maker": {
        "maker-heavy": 3, "aggressive": 1, "high-turnover": 1,
        "mixed": 0, "emeralds focused": 1, "tomatoes focused": 0,
        "taker-heavy": -2, "conservative": -1, "low-turnover": -1, "stub-aware": -1,
    },
    # Task involves taker activity
    "taker": {
        "taker-heavy": 2, "aggressive": 2, "high-turnover": 1,
        "mixed": 0, "tomatoes focused": 1, "emeralds focused": 0,
        "maker-heavy": -1, "conservative": -2, "low-turnover": -2, "stub-aware": 0,
    },
    # Task targets EMERALDS specifically
    "emeralds": {
        "emeralds focused": 3, "maker-heavy": 2, "aggressive": 1,
        "high-turnover": 1, "mixed": 0,
        "tomatoes focused": -1, "taker-heavy": -1, "conservative": -1,
        "low-turnover": -1, "stub-aware": -1,
    },
    # Task targets TOMATOES specifically
    "tomatoes": {
        "tomatoes focused": 3, "aggressive": 2, "high-turnover": 1,
        "maker-heavy": 1, "mixed": 0,
        "emeralds focused": -1, "taker-heavy": 0, "conservative": -1,
        "low-turnover": -1, "stub-aware": -1,
    },
    # General / balanced task
    "balanced": {
        "aggressive": 2, "maker-heavy": 2, "high-turnover": 1,
        "mixed": 1, "tomatoes focused": 0, "emeralds focused": 0,
        "taker-heavy": -1, "conservative": -2, "low-turnover": -2, "stub-aware": -1,
    },
}


def _infer_affinity_key(card: dict[str, Any]) -> str:
    """Infer which affinity profile to use from an alpha card."""
    category = card.get("category", "")
    products = card.get("products", [])
    style = card.get("candidate_strategy_style", "").lower()
    regime = card.get("regime_definition", {})
    role = regime.get("role", "")

    # Role-based
    if role == "maker" or "maker" in style or "passive" in style:
        return "maker"
    if role == "taker" or "taker" in style or "active" in style:
        return "taker"

    # Product-based
    if products == ["EMERALDS"]:
        return "emeralds"
    if products == ["TOMATOES"]:
        return "tomatoes"

    return "balanced"


def score_parent(
    parent: dict[str, Any],
    card: dict[str, Any],
    affinity_key: str | None = None,
) -> dict[str, Any]:
    """Score a candidate parent for a given alpha card.

    Returns dict with score breakdown and total.
    """
    if affinity_key is None:
        affinity_key = _infer_affinity_key(card)

    ps = parent.get("packet_short", parent)
    family = parent.get("_family", parent.get("family", "unknown"))
    pnl = ps.get("pnl", {})
    promote = ps.get("promote", {})
    per_product = ps.get("per_product", {})
    fill_quality = ps.get("fill_quality", {})
    efficiency = ps.get("efficiency", {})

    # 1. Family affinity (0-3 scale, can be negative)
    affinity_table = FAMILY_AFFINITY.get(affinity_key, FAMILY_AFFINITY["balanced"])
    family_score = affinity_table.get(family, 0)

    # 2. Quality score (0-5 scale)
    pnl_mean = pnl.get("mean", 0)
    sharpe = pnl.get("sharpe_like", 0)
    quality_score = 0.0
    if promote.get("recommended", False):
        quality_score += 2.0
    quality_score += min(sharpe / 10.0, 2.0)       # Up to 2 for sharpe
    quality_score += min(pnl_mean / 10000.0, 1.0)  # Up to 1 for PnL

    # 3. Product specialization (0-2 scale)
    products = card.get("products", [])
    product_score = 0.0
    if "EMERALDS" in products and len(products) == 1:
        em_pnl = per_product.get("emerald", {}).get("mean", 0)
        product_score = min(em_pnl / 5000.0, 2.0) if em_pnl > 0 else 0
    elif "TOMATOES" in products and len(products) == 1:
        tom_pnl = per_product.get("tomato", {}).get("mean", 0)
        product_score = min(tom_pnl / 5000.0, 2.0) if tom_pnl > 0 else 0
    else:
        # Balanced — reward parents strong in both
        em_pnl = per_product.get("emerald", {}).get("mean", 0)
        tom_pnl = per_product.get("tomato", {}).get("mean", 0)
        if em_pnl > 0 and tom_pnl > 0:
            product_score = min((em_pnl + tom_pnl) / 15000.0, 2.0)

    # 4. Damage risk penalty (0 to -3 scale)
    # Higher PnL parents have more to lose — but only penalize if task
    # targets the parent's strongest product
    damage_penalty = 0.0
    if products == ["EMERALDS"]:
        em_share = per_product.get("emerald", {}).get("mean", 0) / max(pnl_mean, 1)
        if em_share > 0.6:
            damage_penalty = -1.0  # Parent is EMERALDS-heavy; risky to modify EMERALDS
    elif products == ["TOMATOES"]:
        tom_share = per_product.get("tomato", {}).get("mean", 0) / max(pnl_mean, 1)
        if tom_share > 0.6:
            damage_penalty = -1.0

    # 5. Frontier status bonus
    frontier_bonus = 1.0 if promote.get("recommended", False) else 0.0

    total = family_score + quality_score + product_score + damage_penalty + frontier_bonus

    return {
        "parent_id": parent.get("_case_id", parent.get("case_id", "?")),
        "parent_family": family,
        "family_score": family_score,
        "quality_score": round(quality_score, 2),
        "product_score": round(product_score, 2),
        "damage_penalty": round(damage_penalty, 2),
        "frontier_bonus": frontier_bonus,
        "total_score": round(total, 2),
        "promoted": promote.get("recommended", False),
        "pnl_mean": pnl_mean,
        "sharpe": sharpe,
    }


def select_parent(
    parents: list[dict[str, Any]],
    card: dict[str, Any],
    affinity_key: str | None = None,
) -> dict[str, Any]:
    """Select the best parent for an alpha card.

    Returns the top-scoring parent with score breakdown and rationale.
    """
    if not parents:
        return {
            "parent_id": "none",
            "parent_family": "none",
            "total_score": 0,
            "rationale": "No parent candidates available.",
        }

    scored = [score_parent(p, card, affinity_key) for p in parents]
    scored.sort(key=lambda s: s["total_score"], reverse=True)

    best = scored[0]

    # Build rationale
    reasons = []
    if best["family_score"] > 0:
        reasons.append(f"family '{best['parent_family']}' matches task style")
    if best["promoted"]:
        reasons.append("promoted candidate on frontier")
    if best["quality_score"] > 3:
        reasons.append(f"high quality (Sharpe={best['sharpe']:.1f})")
    if best["product_score"] > 1:
        reasons.append("strong in target product")
    if best["damage_penalty"] < 0:
        reasons.append("note: some risk to existing edge")

    best["rationale"] = "; ".join(reasons) if reasons else "best available option"
    best["runner_up"] = scored[1] if len(scored) > 1 else None

    return best
