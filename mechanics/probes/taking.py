"""Taking probes.

Probes that examine when aggressive (taker) fills are profitable,
how edge varies with distance from fair value, and how market state
conditions the value of taking.
"""
from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd

from mechanics.probe_spec import Probe, ProbeSpec, ProbeResult, register_probe
from mechanics.probes._helpers import (
    enrich_fills,
    build_fair_index,
    fair_change_before_fill,
    bucket_value,
    bucket_stats,
    SPREAD_EDGES,
    SPREAD_LABELS,
    DISTANCE_EDGES,
    DISTANCE_LABELS,
)


# ---------------------------------------------------------------------------
# TK01: Taker edge by distance from fair
# ---------------------------------------------------------------------------

@register_probe
class TakeEdgeByDistance(Probe):
    spec = ProbeSpec(
        probe_id="tk01_take_edge_by_distance",
        family="taking",
        title="Taker fill edge by distance from fair value",
        hypothesis="Taking further from fair value captures more edge per fill.",
        required_data=("strategy_fills",),
        metrics_produced=("edge_by_distance_bucket", "edge_increases_with_distance"),
    )

    def run(self, session_ledgers, product, dataset_label=""):
        bucket_edges: dict[str, list[float]] = defaultdict(list)
        total_fills = 0

        for sid, ledger in session_ledgers.items():
            fills = ledger["strategy_fills"]
            if fills.empty:
                continue
            taker = fills[
                (fills["product"] == product) & (fills["strategy_role"] == "taker")
            ]
            for _, row in taker.iterrows():
                if math.isnan(row.get("fill_vs_fair", math.nan)):
                    continue
                distance = abs(row["price"] - row["fair_value"])
                label = bucket_value(distance, DISTANCE_EDGES, DISTANCE_LABELS)
                bucket_edges[label].append(row["fill_vs_fair"])
                total_fills += 1

        if total_fills < 20:
            return self._insufficient(product, dataset_label, f"Only {total_fills} taker fills.")

        edge_by_bucket = {label: bucket_stats(vals) for label, vals in bucket_edges.items()}

        ordered = [l for l in DISTANCE_LABELS if l in edge_by_bucket]
        means = [edge_by_bucket[l]["mean"] for l in ordered if not math.isnan(edge_by_bucket[l]["mean"])]
        increases = all(means[i] <= means[i + 1] for i in range(len(means) - 1)) if len(means) >= 2 else False

        if len(means) < 2:
            verdict = "inconclusive"
            detail = "Not enough distance buckets with data."
        elif increases:
            verdict = "supported"
            detail = f"Taker edge increases with distance across {len(ordered)} buckets. Far fills are better."
        else:
            # Check if close fills are actually negative
            close_mean = edge_by_bucket.get(DISTANCE_LABELS[0], {}).get("mean", math.nan)
            if not math.isnan(close_mean) and close_mean < 0:
                verdict = "refuted"
                detail = f"Close takes (distance < {DISTANCE_EDGES[0]}) have NEGATIVE edge ({close_mean:.2f}). Avoid close taking."
            else:
                verdict = "inconclusive"
                detail = f"Edge does not increase monotonically. Bucket means: {dict(zip(ordered, means))}."

        return ProbeResult(
            probe_id=self.spec.probe_id,
            family=self.spec.family,
            title=self.spec.title,
            hypothesis=self.spec.hypothesis,
            product=product,
            dataset=dataset_label,
            sample_size={"sessions": len(session_ledgers), "taker_fills": total_fills},
            metrics={"edge_by_distance_bucket": edge_by_bucket, "edge_increases_with_distance": increases},
            verdict=verdict,
            confidence="high" if total_fills >= 200 else "medium" if total_fills >= 50 else "low",
            detail=detail,
        )


# ---------------------------------------------------------------------------
# TK02: Taker edge by spread state
# ---------------------------------------------------------------------------

@register_probe
class TakeEdgeBySpread(Probe):
    spec = ProbeSpec(
        probe_id="tk02_take_edge_by_spread",
        family="taking",
        title="Taker fill edge conditioned on spread",
        hypothesis="Taking in wider spreads captures more edge.",
        required_data=("strategy_fills", "prices"),
        metrics_produced=("edge_by_spread_bucket", "best_spread_regime"),
    )

    def run(self, session_ledgers, product, dataset_label=""):
        bucket_edges: dict[str, list[float]] = defaultdict(list)
        total_fills = 0

        for sid, ledger in session_ledgers.items():
            fills = ledger["strategy_fills"]
            if fills.empty:
                continue
            enriched = enrich_fills(fills, ledger["traces"], ledger["prices"])
            taker = enriched[
                (enriched["product"] == product) & (enriched["strategy_role"] == "taker")
            ]
            for _, row in taker.iterrows():
                s = row["spread"]
                if math.isnan(s) or math.isnan(row["fill_vs_fair"]):
                    continue
                label = bucket_value(s, SPREAD_EDGES, SPREAD_LABELS)
                bucket_edges[label].append(row["fill_vs_fair"])
                total_fills += 1

        if total_fills < 20:
            return self._insufficient(product, dataset_label, f"Only {total_fills} taker fills with spread data.")

        edge_by_bucket = {label: bucket_stats(vals) for label, vals in bucket_edges.items()}

        # Find best spread regime
        best_label = None
        best_mean = -math.inf
        for label, stats in edge_by_bucket.items():
            if stats["count"] >= 5 and not math.isnan(stats["mean"]) and stats["mean"] > best_mean:
                best_mean = stats["mean"]
                best_label = label

        # Find worst
        worst_label = None
        worst_mean = math.inf
        for label, stats in edge_by_bucket.items():
            if stats["count"] >= 5 and not math.isnan(stats["mean"]) and stats["mean"] < worst_mean:
                worst_mean = stats["mean"]
                worst_label = label

        if best_label and worst_label and best_label != worst_label:
            verdict = "supported" if best_mean > 0 else "inconclusive"
            detail = (
                f"Best taker regime: spread {best_label} (mean edge {best_mean:.2f}). "
                f"Worst: spread {worst_label} (mean edge {worst_mean:.2f})."
            )
        else:
            verdict = "inconclusive"
            detail = "No clear spread regime preference for taking."

        return ProbeResult(
            probe_id=self.spec.probe_id,
            family=self.spec.family,
            title=self.spec.title,
            hypothesis=self.spec.hypothesis,
            product=product,
            dataset=dataset_label,
            sample_size={"sessions": len(session_ledgers), "taker_fills": total_fills},
            metrics={
                "edge_by_spread_bucket": edge_by_bucket,
                "best_spread_regime": best_label,
                "worst_spread_regime": worst_label,
            },
            verdict=verdict,
            confidence="high" if total_fills >= 200 else "medium" if total_fills >= 50 else "low",
            detail=detail,
        )


# ---------------------------------------------------------------------------
# TK03: Taker direction vs local trend
# ---------------------------------------------------------------------------

@register_probe
class TakeDirectionVsTrend(Probe):
    spec = ProbeSpec(
        probe_id="tk03_take_direction_vs_trend",
        family="taking",
        title="Taker fill edge: with-trend vs against-trend",
        hypothesis="Taking against the local trend (mean-reversion) is better than taking with the trend (momentum).",
        required_data=("strategy_fills", "traces"),
        metrics_produced=("with_trend_edge", "against_trend_edge", "reversion_better"),
    )

    LOOKBACK = 10

    def run(self, session_ledgers, product, dataset_label=""):
        with_trend_edges: list[float] = []
        against_trend_edges: list[float] = []
        total_fills = 0

        for sid, ledger in session_ledgers.items():
            fills = ledger["strategy_fills"]
            traces = ledger["traces"]
            if fills.empty or traces.empty:
                continue

            fair_idx = build_fair_index(traces, product)
            if fair_idx.empty:
                continue

            taker = fills[
                (fills["product"] == product) & (fills["strategy_role"] == "taker")
            ]
            if taker.empty:
                continue

            lookbacks = fair_change_before_fill(taker, fair_idx, self.LOOKBACK)
            for i, (_, fill) in enumerate(taker.iterrows()):
                if i >= len(lookbacks) or math.isnan(lookbacks[i]):
                    continue
                if math.isnan(fill.get("fill_vs_fair", math.nan)):
                    continue

                total_fills += 1
                trend = lookbacks[i]  # positive = price went up recently
                is_buy = fill["strategy_side"] == "buy"

                # With trend: buying after uptrend or selling after downtrend
                with_trend = (is_buy and trend > 0) or (not is_buy and trend < 0)

                if abs(trend) < 0.01:
                    continue  # no trend, skip

                if with_trend:
                    with_trend_edges.append(fill["fill_vs_fair"])
                else:
                    against_trend_edges.append(fill["fill_vs_fair"])

        if total_fills < 20:
            return self._insufficient(product, dataset_label, f"Only {total_fills} taker fills with trend data.")

        with_stats = bucket_stats(with_trend_edges)
        against_stats = bucket_stats(against_trend_edges)

        reversion_better = (
            not math.isnan(against_stats["mean"])
            and not math.isnan(with_stats["mean"])
            and against_stats["mean"] > with_stats["mean"]
        )

        if with_stats["count"] < 10 or against_stats["count"] < 10:
            verdict = "inconclusive"
            detail = f"Not enough fills in one category: with_trend={with_stats['count']}, against_trend={against_stats['count']}."
        elif reversion_better:
            gap = against_stats["mean"] - with_stats["mean"]
            verdict = "supported"
            detail = f"Mean-reversion taking is better by {gap:.2f}. Against-trend edge: {against_stats['mean']:.2f}, with-trend: {with_stats['mean']:.2f}."
        else:
            gap = with_stats["mean"] - against_stats["mean"]
            verdict = "refuted"
            detail = f"Momentum taking is better by {gap:.2f}. With-trend edge: {with_stats['mean']:.2f}, against-trend: {against_stats['mean']:.2f}."

        return ProbeResult(
            probe_id=self.spec.probe_id,
            family=self.spec.family,
            title=self.spec.title,
            hypothesis=self.spec.hypothesis,
            product=product,
            dataset=dataset_label,
            sample_size={"sessions": len(session_ledgers), "taker_fills": total_fills},
            metrics={
                "with_trend_edge": with_stats,
                "against_trend_edge": against_stats,
                "reversion_better": reversion_better,
            },
            verdict=verdict,
            confidence="high" if total_fills >= 200 else "medium" if total_fills >= 50 else "low",
            detail=detail,
        )
