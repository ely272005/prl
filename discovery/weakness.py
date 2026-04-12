"""Weakness discovery — scans regime profiles and comparisons to generate alpha cards.

Five scanner patterns:
  1. Regime Edge Scanner:     finds regimes with unusually good/bad fill quality
  2. Role Mismatch Scanner:   finds regimes where maker vs taker edge diverges
  3. Winner Trait Scanner:    converts cross-candidate comparison into cards
  4. Probe-Driven Scanner:    converts strong probe verdicts into cards
  5. Inventory Exploit Scanner: finds position states with asymmetric edge
"""
from __future__ import annotations

import math
from typing import Any

from discovery.alpha_card import AlphaCard, CardCounter


# ---------------------------------------------------------------------------
# Thresholds for weakness detection
# ---------------------------------------------------------------------------

# Regime edge: minimum absolute and relative difference to flag
MIN_ABSOLUTE_EDGE_DIFF = 0.8       # |regime_edge - baseline| must exceed this
MIN_RELATIVE_EDGE_DIFF = 0.25      # |diff / max(|baseline|, 1)| must exceed this
MIN_FILLS_PER_BUCKET = 30          # Minimum fills in a regime bucket

# Role mismatch: maker vs taker edge ratio
MIN_ROLE_EDGE_RATIO = 2.0          # One role must have >= 2x edge of the other
MIN_ROLE_FILLS = 20                # Minimum fills per role in a bucket

# Winner trait: effect size threshold
MIN_WINNER_EFFECT_SIZE = 0.8       # Cohen's d equivalent

# Confidence thresholds
HIGH_CONFIDENCE_FILLS = 200
MEDIUM_CONFIDENCE_FILLS = 50


def _confidence_from_fills(n: int) -> str:
    if n >= HIGH_CONFIDENCE_FILLS:
        return "high"
    if n >= MEDIUM_CONFIDENCE_FILLS:
        return "medium"
    return "low"


def _strength_score(effect: float, sample: int, confidence: str) -> float:
    """Compute a ranking score from effect size and sample quality."""
    conf_mult = {"high": 1.0, "medium": 0.7, "low": 0.4}.get(confidence, 0.3)
    return abs(effect) * math.log1p(sample) * conf_mult


# ---------------------------------------------------------------------------
# 1. Regime Edge Scanner
# ---------------------------------------------------------------------------

def scan_regime_edges(
    regime_stats: dict[str, dict[str, dict[str, Any]]],
    counter: CardCounter,
) -> list[AlphaCard]:
    """Scan for regimes with unusually strong or weak fill quality.

    For each product × dimension × role, compares per-label edge to baseline.
    Generates cards for labels that deviate significantly.
    """
    cards: list[AlphaCard] = []

    for product, dims in regime_stats.items():
        for dim, roles in dims.items():
            for role_key, stats in roles.items():
                baseline = stats.get("baseline", {})
                base_mean = baseline.get("mean", math.nan)
                base_count = baseline.get("count", 0)
                if math.isnan(base_mean) or base_count < MIN_FILLS_PER_BUCKET:
                    continue

                by_label = stats.get("by_label", {})
                for label, label_stats in by_label.items():
                    l_mean = label_stats.get("mean", math.nan)
                    l_count = label_stats.get("count", 0)
                    if math.isnan(l_mean) or l_count < MIN_FILLS_PER_BUCKET:
                        continue

                    diff = l_mean - base_mean
                    abs_diff = abs(diff)
                    rel_diff = abs_diff / max(abs(base_mean), 1.0)

                    if abs_diff < MIN_ABSOLUTE_EDGE_DIFF or rel_diff < MIN_RELATIVE_EDGE_DIFF:
                        continue

                    # Check median consistency (median should agree with mean direction)
                    l_median = label_stats.get("median", math.nan)
                    if not math.isnan(l_median):
                        if (diff > 0 and l_median < base_mean) or (diff < 0 and l_median > base_mean):
                            continue  # inconsistent — skip

                    role_desc = role_key if role_key != "all" else "all roles"
                    confidence = _confidence_from_fills(l_count)
                    is_opportunity = diff > 0

                    if is_opportunity:
                        category = "regime_edge"
                        title = f"{product} {role_desc} edge peaks in {dim}={label}"
                        fact = (
                            f"{product} {role_desc} fills in {dim}={label} have mean edge "
                            f"{l_mean:.2f} vs baseline {base_mean:.2f} "
                            f"(+{diff:.2f}, {rel_diff:.0%} above baseline). "
                            f"N={l_count} fills."
                        )
                        interp = (
                            f"The {label} regime in {dim} appears to create favorable "
                            f"conditions for {role_desc} fills on {product}."
                        )
                        exploit = (
                            f"Target {role_desc} activity on {product} specifically "
                            f"when {dim} is in the {label} state."
                        )
                        style = f"{'Passive/maker' if role_key == 'maker' else 'Active/taker' if role_key == 'taker' else 'Adaptive'} in {label} regime"
                    else:
                        category = "danger_refinement"
                        title = f"{product} {role_desc} edge drops in {dim}={label}"
                        fact = (
                            f"{product} {role_desc} fills in {dim}={label} have mean edge "
                            f"{l_mean:.2f} vs baseline {base_mean:.2f} "
                            f"({diff:.2f}, {rel_diff:.0%} below baseline). "
                            f"N={l_count} fills."
                        )
                        interp = (
                            f"The {label} regime in {dim} appears to hurt "
                            f"{role_desc} fill quality on {product}."
                        )
                        exploit = (
                            f"Reduce or avoid {role_desc} activity on {product} "
                            f"when {dim} enters the {label} state."
                        )
                        style = f"Defensive / reduce exposure in {label} regime"

                    experiment = (
                        f"Backtest a variant that {'increases' if is_opportunity else 'reduces'} "
                        f"{role_desc} aggressiveness on {product} when {dim}={label}."
                    )

                    cards.append(AlphaCard(
                        card_id=counter.next_id(category),
                        title=title,
                        category=category,
                        products=[product],
                        observed_fact=fact,
                        interpretation=interp,
                        suggested_exploit=exploit,
                        regime_definition={dim: label, "product": product, "role": role_key},
                        evidence={
                            "regime_mean": l_mean,
                            "regime_median": l_median,
                            "regime_std": label_stats.get("std", math.nan),
                            "baseline_mean": base_mean,
                            "difference": diff,
                            "relative_difference": rel_diff,
                        },
                        baseline={"mean": base_mean, "count": base_count},
                        sample_size={"fills": l_count, "baseline_fills": base_count},
                        confidence=confidence,
                        strength=_strength_score(rel_diff, l_count, confidence),
                        candidate_strategy_style=style,
                        recommended_experiment=experiment,
                    ))

    return cards


# ---------------------------------------------------------------------------
# 2. Role Mismatch Scanner
# ---------------------------------------------------------------------------

def scan_role_mismatches(
    regime_stats: dict[str, dict[str, dict[str, Any]]],
    counter: CardCounter,
) -> list[AlphaCard]:
    """Find regimes where maker and taker edge diverge significantly."""
    cards: list[AlphaCard] = []

    for product, dims in regime_stats.items():
        for dim, roles in dims.items():
            maker_stats = roles.get("maker", {}).get("by_label", {})
            taker_stats = roles.get("taker", {}).get("by_label", {})

            common_labels = set(maker_stats.keys()) & set(taker_stats.keys())

            for label in common_labels:
                m = maker_stats[label]
                t = taker_stats[label]

                m_mean = m.get("mean", math.nan)
                t_mean = t.get("mean", math.nan)
                m_count = m.get("count", 0)
                t_count = t.get("count", 0)

                if math.isnan(m_mean) or math.isnan(t_mean):
                    continue
                if m_count < MIN_ROLE_FILLS or t_count < MIN_ROLE_FILLS:
                    continue

                # Check ratio: one role must dominate
                m_abs = max(abs(m_mean), 0.01)
                t_abs = max(abs(t_mean), 0.01)
                ratio = max(m_abs, t_abs) / min(m_abs, t_abs)

                if ratio < MIN_ROLE_EDGE_RATIO:
                    continue

                # Also require sign difference or large magnitude gap
                if m_mean > 0 and t_mean > 0 and ratio < 3.0:
                    continue  # Both positive but not dramatically different

                maker_wins = m_mean > t_mean
                total_fills = m_count + t_count
                confidence = _confidence_from_fills(total_fills)

                winner_role = "maker" if maker_wins else "taker"
                loser_role = "taker" if maker_wins else "maker"
                w_mean = m_mean if maker_wins else t_mean
                l_mean = t_mean if maker_wins else m_mean

                cards.append(AlphaCard(
                    card_id=counter.next_id("role_mismatch"),
                    title=f"{product}: {winner_role} dominates in {dim}={label}",
                    category="role_mismatch",
                    products=[product],
                    observed_fact=(
                        f"In {dim}={label} on {product}, {winner_role} edge is {w_mean:.2f} "
                        f"vs {loser_role} edge {l_mean:.2f} (ratio {ratio:.1f}x). "
                        f"Maker N={m_count}, Taker N={t_count}."
                    ),
                    interpretation=(
                        f"The {label} regime in {dim} strongly favors {winner_role} "
                        f"activity on {product}. {loser_role.capitalize()} fills in this "
                        f"regime may be destroying value."
                    ),
                    suggested_exploit=(
                        f"In {dim}={label}, shift {product} strategy toward {winner_role} "
                        f"activity and reduce {loser_role} fills."
                    ),
                    regime_definition={dim: label, "product": product},
                    evidence={
                        "maker_mean": m_mean,
                        "taker_mean": t_mean,
                        "ratio": ratio,
                    },
                    baseline={
                        "maker_count": m_count,
                        "taker_count": t_count,
                    },
                    sample_size={"maker_fills": m_count, "taker_fills": t_count},
                    confidence=confidence,
                    strength=_strength_score(ratio, total_fills, confidence),
                    candidate_strategy_style=f"{winner_role.capitalize()}-heavy in {label} regime",
                    recommended_experiment=(
                        f"Compare a {winner_role}-only variant vs mixed strategy "
                        f"on {product} during {dim}={label} periods."
                    ),
                ))

    return cards


# ---------------------------------------------------------------------------
# 3. Winner Trait Scanner
# ---------------------------------------------------------------------------

def scan_winner_traits(
    comparison: dict[str, Any],
    counter: CardCounter,
) -> list[AlphaCard]:
    """Convert cross-candidate comparison findings into alpha cards.

    Looks for metrics where winners and losers differ by >= MIN_WINNER_EFFECT_SIZE.
    """
    cards: list[AlphaCard] = []

    metric_cmp = comparison.get("metric_comparison", {})
    winner_count = comparison.get("winner_count", 0)
    loser_count = comparison.get("loser_count", 0)

    if winner_count < 2 or loser_count < 2:
        return cards

    for metric_name, cmp in metric_cmp.items():
        effect = cmp.get("effect_size", math.nan)
        if math.isnan(effect) or abs(effect) < MIN_WINNER_EFFECT_SIZE:
            continue

        w_mean = cmp.get("winner_mean", math.nan)
        l_mean = cmp.get("loser_mean", math.nan)
        diff = cmp.get("difference", math.nan)
        higher_is_better = cmp.get("higher_is_better")

        if math.isnan(w_mean) or math.isnan(l_mean):
            continue

        # Determine product from metric name
        products = ["EMERALDS", "TOMATOES"]
        if "emerald" in metric_name.lower():
            products = ["EMERALDS"]
        elif "tomato" in metric_name.lower():
            products = ["TOMATOES"]

        # Determine direction interpretation
        if higher_is_better is True:
            winners_better = diff > 0
        elif higher_is_better is False:
            winners_better = diff < 0  # For drawdown, less negative is better
        else:
            winners_better = True  # Neutral metric — just report the difference

        trait_desc = f"{'higher' if diff > 0 else 'lower'} {metric_name}"
        confidence = "medium" if abs(effect) > 1.5 else "low"

        cards.append(AlphaCard(
            card_id=counter.next_id("winner_trait"),
            title=f"Winners have {trait_desc}",
            category="winner_trait",
            products=products,
            observed_fact=(
                f"Promoted candidates have {metric_name}={w_mean:.3f} "
                f"vs rejected {l_mean:.3f} (diff={diff:.3f}, "
                f"effect size={effect:.2f}). "
                f"N={winner_count} winners, {loser_count} losers."
            ),
            interpretation=(
                f"{'Strong' if winners_better else 'Surprising'}: "
                f"{metric_name} is {'positively' if winners_better else 'inversely'} "
                f"associated with promotion. "
                f"{'This aligns with the metric direction.' if winners_better else 'This may indicate a confound.'}"
            ),
            suggested_exploit=(
                f"Design strategies that {'maximize' if (diff > 0) == (higher_is_better or higher_is_better is None) else 'minimize'} "
                f"{metric_name}. Use this as a screening criterion for descendants."
            ),
            regime_definition={"metric": metric_name, "comparison": "promoted_vs_rejected"},
            evidence={
                "winner_mean": w_mean,
                "loser_mean": l_mean,
                "effect_size": effect,
                "direction": "higher" if diff > 0 else "lower",
            },
            baseline={"loser_mean": l_mean, "loser_std": cmp.get("loser_std", 0)},
            sample_size={"winners": winner_count, "losers": loser_count},
            confidence=confidence,
            strength=_strength_score(abs(effect), winner_count + loser_count, confidence),
            candidate_strategy_style="Match promoted candidate profile",
            recommended_experiment=(
                f"Generate descendants that target {metric_name} near the "
                f"winner mean ({w_mean:.2f}) and test for PnL correlation."
            ),
        ))

    # Family-level findings
    family_cmp = comparison.get("family_comparison", {})
    if family_cmp:
        best_fam = max(
            family_cmp.items(),
            key=lambda x: x[1].get("pnl_mean", 0) if not math.isnan(x[1].get("pnl_mean", 0)) else 0,
        )
        worst_fam = min(
            family_cmp.items(),
            key=lambda x: x[1].get("pnl_mean", float("inf")) if not math.isnan(x[1].get("pnl_mean", 0)) else float("inf"),
        )
        if (best_fam[0] != worst_fam[0]
            and best_fam[1].get("count", 0) >= 2
            and worst_fam[1].get("count", 0) >= 2):

            best_pnl = best_fam[1].get("pnl_mean", 0)
            worst_pnl = worst_fam[1].get("pnl_mean", 0)
            if best_pnl > 0 and (best_pnl - worst_pnl) > 1000:
                cards.append(AlphaCard(
                    card_id=counter.next_id("winner_trait"),
                    title=f"Family '{best_fam[0]}' dominates over '{worst_fam[0]}'",
                    category="winner_trait",
                    products=["EMERALDS", "TOMATOES"],
                    observed_fact=(
                        f"Family '{best_fam[0]}' has mean PnL {best_pnl:.0f} "
                        f"({best_fam[1].get('count', 0)} candidates, "
                        f"{best_fam[1].get('promoted', 0)} promoted) vs "
                        f"'{worst_fam[0]}' with mean PnL {worst_pnl:.0f} "
                        f"({worst_fam[1].get('count', 0)} candidates, "
                        f"{worst_fam[1].get('promoted', 0)} promoted)."
                    ),
                    interpretation=(
                        f"The '{best_fam[0]}' strategy style systematically outperforms "
                        f"'{worst_fam[0]}'. This may reflect a structural advantage in "
                        f"this environment."
                    ),
                    suggested_exploit=(
                        f"Prioritize '{best_fam[0]}' style strategies. "
                        f"Investigate what makes '{worst_fam[0]}' fail and "
                        f"avoid those traits."
                    ),
                    regime_definition={"comparison": "family_performance"},
                    evidence={
                        "best_family": best_fam[0],
                        "best_pnl": best_pnl,
                        "best_sharpe": best_fam[1].get("sharpe_mean", 0),
                        "worst_family": worst_fam[0],
                        "worst_pnl": worst_pnl,
                        "worst_sharpe": worst_fam[1].get("sharpe_mean", 0),
                    },
                    baseline={"all_families": {k: v.get("pnl_mean", 0) for k, v in family_cmp.items()}},
                    sample_size={
                        "best_count": best_fam[1].get("count", 0),
                        "worst_count": worst_fam[1].get("count", 0),
                    },
                    confidence="medium",
                    strength=_strength_score(
                        (best_pnl - worst_pnl) / max(best_pnl, 1),
                        best_fam[1].get("count", 0) + worst_fam[1].get("count", 0),
                        "medium",
                    ),
                    candidate_strategy_style=f"{best_fam[0]} style",
                    recommended_experiment=(
                        f"Generate more '{best_fam[0]}' style descendants. "
                        f"Analyze what specific traits of '{worst_fam[0]}' cause losses."
                    ),
                ))

    return cards


# ---------------------------------------------------------------------------
# 4. Probe-Driven Scanner
# ---------------------------------------------------------------------------

def scan_probe_results(
    probe_results: list[dict[str, Any]],
    counter: CardCounter,
) -> list[AlphaCard]:
    """Convert strong probe verdicts into alpha cards.

    Only processes probes with verdict 'supported' or 'refuted' and
    confidence >= 'medium'.
    """
    cards: list[AlphaCard] = []

    for pr in probe_results:
        verdict = pr.get("verdict", "")
        confidence = pr.get("confidence", "low")
        if verdict not in ("supported", "refuted"):
            continue
        if confidence == "low":
            continue

        probe_id = pr.get("probe_id", "unknown")
        product = pr.get("product", "UNKNOWN")
        hypothesis = pr.get("hypothesis", "")
        detail = pr.get("detail", "")
        family = pr.get("family", "unknown")
        sample = pr.get("sample_size", {})

        # Determine category from probe family
        if family in ("passive_fill",):
            category = "regime_edge"
            style = "Maker/passive strategy"
        elif family in ("taking",):
            category = "regime_edge"
            style = "Taker/active strategy"
        elif family in ("inventory",):
            category = "inventory_exploit"
            style = "Inventory-aware strategy"
        elif family in ("danger_zone",):
            category = "danger_refinement"
            style = "Defensive / danger-aware strategy"
        else:
            category = "bot_weakness"
            style = "Adaptive strategy"

        is_supported = verdict == "supported"

        cards.append(AlphaCard(
            card_id=counter.next_id(category),
            title=f"Probe {probe_id}: {verdict} on {product}",
            category=category,
            products=[product],
            observed_fact=f"Probe {probe_id} ({verdict}): {detail}",
            interpretation=(
                f"The hypothesis '{hypothesis}' is "
                f"{'confirmed by evidence' if is_supported else 'contradicted by evidence'} "
                f"for {product}."
            ),
            suggested_exploit=(
                f"{'Exploit this confirmed pattern' if is_supported else 'Avoid strategies that assume this pattern'} "
                f"on {product}."
            ),
            regime_definition={"probe_id": probe_id, "probe_family": family},
            evidence=pr.get("metrics", {}),
            baseline={"hypothesis": hypothesis, "verdict": verdict},
            sample_size=sample,
            confidence=confidence,
            strength=_strength_score(
                1.5 if confidence == "high" else 1.0,
                sum(sample.values()) if sample else 0,
                confidence,
            ),
            candidate_strategy_style=style,
            recommended_experiment=(
                f"Build a strategy variant that specifically "
                f"{'leverages' if is_supported else 'avoids relying on'} "
                f"the pattern from {probe_id} on {product}."
            ),
        ))

    return cards


# ---------------------------------------------------------------------------
# 5. Inventory Exploit Scanner
# ---------------------------------------------------------------------------

def scan_inventory_exploits(
    regime_stats: dict[str, dict[str, dict[str, Any]]],
    counter: CardCounter,
) -> list[AlphaCard]:
    """Find position states with asymmetric fill quality.

    Specifically looks for:
    - Position buckets where inventory-reducing fills have better edge
    - Regimes where high inventory doesn't cause the expected drag
    """
    cards: list[AlphaCard] = []

    for product, dims in regime_stats.items():
        # Check position_bucket for asymmetric edge
        pos_stats = dims.get("position_bucket", {})
        for role_key in ("all", "maker", "taker"):
            stats = pos_stats.get(role_key, {})
            by_label = stats.get("by_label", {})
            baseline = stats.get("baseline", {})
            base_mean = baseline.get("mean", math.nan)
            if math.isnan(base_mean):
                continue

            # Compare opposite position buckets
            for pos_a, pos_b in [("long", "short"), ("deep_long", "deep_short")]:
                a_stats = by_label.get(pos_a, {})
                b_stats = by_label.get(pos_b, {})
                a_mean = a_stats.get("mean", math.nan)
                b_mean = b_stats.get("mean", math.nan)
                a_count = a_stats.get("count", 0)
                b_count = b_stats.get("count", 0)

                if math.isnan(a_mean) or math.isnan(b_mean):
                    continue
                if a_count < MIN_FILLS_PER_BUCKET or b_count < MIN_FILLS_PER_BUCKET:
                    continue

                diff = abs(a_mean - b_mean)
                if diff < MIN_ABSOLUTE_EDGE_DIFF:
                    continue

                better_pos = pos_a if a_mean > b_mean else pos_b
                worse_pos = pos_b if a_mean > b_mean else pos_a
                better_mean = max(a_mean, b_mean)
                worse_mean = min(a_mean, b_mean)
                total_fills = a_count + b_count
                confidence = _confidence_from_fills(total_fills)
                role_desc = role_key if role_key != "all" else "all roles"

                cards.append(AlphaCard(
                    card_id=counter.next_id("inventory_exploit"),
                    title=f"{product}: {better_pos} fills beat {worse_pos} ({role_desc})",
                    category="inventory_exploit",
                    products=[product],
                    observed_fact=(
                        f"On {product} ({role_desc}), fills in {better_pos} position have "
                        f"edge {better_mean:.2f} vs {worse_mean:.2f} in {worse_pos} "
                        f"(diff={diff:.2f}). N={a_count}+{b_count} fills."
                    ),
                    interpretation=(
                        f"Position asymmetry exists: {better_pos} position states "
                        f"yield better fill quality, possibly because inventory pressure "
                        f"creates more favorable counterparty behavior."
                    ),
                    suggested_exploit=(
                        f"Increase aggressiveness when in {better_pos} position on {product}. "
                        f"Skew quotes or increase take_width when position favors it."
                    ),
                    regime_definition={
                        "position_bucket": f"{pos_a} vs {pos_b}",
                        "product": product,
                        "role": role_key,
                    },
                    evidence={
                        f"{pos_a}_mean": a_mean,
                        f"{pos_b}_mean": b_mean,
                        "difference": diff,
                    },
                    baseline={"mean": base_mean, "count": baseline.get("count", 0)},
                    sample_size={f"{pos_a}_fills": a_count, f"{pos_b}_fills": b_count},
                    confidence=confidence,
                    strength=_strength_score(diff, total_fills, confidence),
                    candidate_strategy_style="Inventory-aware skewing",
                    recommended_experiment=(
                        f"Test an asymmetric skew: increase {role_desc} aggressiveness "
                        f"in {better_pos}, reduce in {worse_pos}."
                    ),
                ))

    return cards


# ---------------------------------------------------------------------------
# Master scan: run all scanners, rank and filter
# ---------------------------------------------------------------------------

def run_all_scanners(
    regime_stats: dict[str, dict[str, dict[str, Any]]],
    comparison: dict[str, Any] | None = None,
    probe_results: list[dict[str, Any]] | None = None,
    max_cards: int = 20,
) -> list[AlphaCard]:
    """Run all weakness scanners and return ranked, filtered alpha cards.

    Args:
        regime_stats: from regimes.build_regime_profile()
        comparison: from comparison.run_comparison()
        probe_results: flat list of probe result dicts
        max_cards: maximum number of cards to return
    """
    counter = CardCounter()
    all_cards: list[AlphaCard] = []

    # 1. Regime edge
    all_cards.extend(scan_regime_edges(regime_stats, counter))

    # 2. Role mismatch
    all_cards.extend(scan_role_mismatches(regime_stats, counter))

    # 3. Winner traits
    if comparison:
        all_cards.extend(scan_winner_traits(comparison, counter))

    # 4. Probe-driven
    if probe_results:
        all_cards.extend(scan_probe_results(probe_results, counter))

    # 5. Inventory exploits
    all_cards.extend(scan_inventory_exploits(regime_stats, counter))

    # Rank by strength, keep top max_cards with diversity enforcement
    all_cards.sort(key=lambda c: c.strength, reverse=True)

    # Diversity budget: max cards per (category, product_tuple)
    # This prevents one dominant insight from flooding the output
    max_per_slot = max(3, max_cards // 5)

    seen: set[str] = set()
    slot_counts: dict[str, int] = {}
    filtered: list[AlphaCard] = []

    for card in all_cards:
        # Dedup: same category + product + regime definition → skip
        rd = card.regime_definition
        dedup_key = str((
            card.category,
            tuple(sorted(card.products)),
            tuple(sorted(rd.items())),
        ))
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        # Diversity: cap per (category, product_tuple)
        slot_key = f"{card.category}:{','.join(sorted(card.products))}"
        slot_counts[slot_key] = slot_counts.get(slot_key, 0) + 1
        if slot_counts[slot_key] > max_per_slot:
            continue

        filtered.append(card)
        if len(filtered) >= max_cards:
            break

    # Re-number cards sequentially after filtering
    final_counter = CardCounter()
    for card in filtered:
        card.card_id = final_counter.next_id(card.category)

    return filtered
