"""Passive fill / maker probes.

Probes that examine when and how passive (maker) fills occur,
how fill quality relates to book state, and adverse selection timing.
"""
from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd

from mechanics.probe_spec import Probe, ProbeSpec, ProbeResult, register_probe
from mechanics.probes._helpers import (
    enrich_fills,
    build_fair_index,
    fair_change_at_fill,
    bucket_value,
    bucket_stats,
    SPREAD_EDGES,
    SPREAD_LABELS,
    ABS_POSITION_EDGES,
    ABS_POSITION_LABELS,
)


# ---------------------------------------------------------------------------
# PF01: Spread bucket vs maker fill edge
# ---------------------------------------------------------------------------

@register_probe
class SpreadVsMakerEdge(Probe):
    spec = ProbeSpec(
        probe_id="pf01_spread_vs_maker_edge",
        family="passive_fill",
        title="Spread bucket vs maker fill edge",
        hypothesis="Wider spreads give makers better fill quality (higher fill_vs_fair).",
        required_data=("strategy_fills", "prices"),
        metrics_produced=("edge_by_spread_bucket", "monotonic_increase"),
    )

    def run(self, session_ledgers, product, dataset_label=""):
        bucket_edges: dict[str, list[float]] = defaultdict(list)
        total_fills = 0

        for sid, ledger in session_ledgers.items():
            fills = ledger["strategy_fills"]
            if fills.empty:
                continue
            enriched = enrich_fills(fills, ledger["traces"], ledger["prices"])
            maker = enriched[
                (enriched["product"] == product) & (enriched["strategy_role"] == "maker")
            ]
            for _, row in maker.iterrows():
                s = row["spread"]
                if math.isnan(s) or math.isnan(row["fill_vs_fair"]):
                    continue
                label = bucket_value(s, SPREAD_EDGES, SPREAD_LABELS)
                bucket_edges[label].append(row["fill_vs_fair"])
                total_fills += 1

        if total_fills < 20:
            return self._insufficient(product, dataset_label, f"Only {total_fills} maker fills with spread data.")

        edge_by_bucket = {label: bucket_stats(vals) for label, vals in bucket_edges.items()}

        # Check monotonic increase: mean edge should rise with wider spread
        ordered = [label for label in SPREAD_LABELS if label in edge_by_bucket]
        means = [edge_by_bucket[l]["mean"] for l in ordered if not math.isnan(edge_by_bucket[l]["mean"])]
        monotonic = all(means[i] <= means[i + 1] for i in range(len(means) - 1)) if len(means) >= 2 else False

        if len(means) < 2:
            verdict = "inconclusive"
            detail = "Not enough spread buckets with data to assess trend."
        elif monotonic:
            verdict = "supported"
            detail = f"Maker edge increases with spread across {len(ordered)} buckets. Widest bucket mean edge: {means[-1]:.2f}."
        else:
            verdict = "refuted"
            detail = f"Maker edge does NOT monotonically increase with spread. Bucket means: {dict(zip(ordered, means))}."

        return ProbeResult(
            probe_id=self.spec.probe_id,
            family=self.spec.family,
            title=self.spec.title,
            hypothesis=self.spec.hypothesis,
            product=product,
            dataset=dataset_label,
            sample_size={"sessions": len(session_ledgers), "maker_fills": total_fills},
            metrics={"edge_by_spread_bucket": edge_by_bucket, "monotonic_increase": monotonic},
            verdict=verdict,
            confidence="high" if total_fills >= 200 else "medium" if total_fills >= 50 else "low",
            detail=detail,
            warnings=self._build_warnings(total_fills, len(session_ledgers)),
        )

    @staticmethod
    def _build_warnings(fills: int, sessions: int) -> list[str]:
        w = []
        if sessions < 10:
            w.append(f"Only {sessions} sessions — regime coverage may be incomplete.")
        if fills < 100:
            w.append(f"Only {fills} maker fills — per-bucket estimates are noisy.")
        return w


# ---------------------------------------------------------------------------
# PF02: Maker adverse selection timing
# ---------------------------------------------------------------------------

@register_probe
class MakerAdverseTiming(Probe):
    spec = ProbeSpec(
        probe_id="pf02_maker_adverse_timing",
        family="passive_fill",
        title="Maker fill adverse selection at multiple horizons",
        hypothesis="Maker fills are adversely selected: fair value moves against the strategy after passive fills.",
        required_data=("strategy_fills", "traces"),
        metrics_produced=("adverse_rate_by_horizon", "maker_vs_taker_adverse"),
    )

    HORIZONS = (5, 10, 25, 50)

    def run(self, session_ledgers, product, dataset_label=""):
        # Collect all maker and taker fills per session with fair index
        maker_adverse: dict[int, dict[str, int]] = {h: {"adverse": 0, "total": 0} for h in self.HORIZONS}
        taker_adverse: dict[int, dict[str, int]] = {h: {"adverse": 0, "total": 0} for h in self.HORIZONS}
        total_maker = 0
        total_taker = 0

        for sid, ledger in session_ledgers.items():
            fills = ledger["strategy_fills"]
            traces = ledger["traces"]
            if fills.empty or traces.empty:
                continue

            fair_idx = build_fair_index(traces, product)
            if fair_idx.empty:
                continue

            prod_fills = fills[fills["product"] == product]
            maker = prod_fills[prod_fills["strategy_role"] == "maker"]
            taker = prod_fills[prod_fills["strategy_role"] == "taker"]
            total_maker += len(maker)
            total_taker += len(taker)

            for horizon in self.HORIZONS:
                for role, role_fills, accum in [("maker", maker, maker_adverse), ("taker", taker, taker_adverse)]:
                    if role_fills.empty:
                        continue
                    fwd = fair_change_at_fill(role_fills, fair_idx, horizon)
                    for i, (_, fill) in enumerate(role_fills.iterrows()):
                        if i >= len(fwd) or math.isnan(fwd[i]):
                            continue
                        d = 1.0 if fill["strategy_side"] == "buy" else -1.0
                        accum[horizon]["total"] += 1
                        if d * fwd[i] < 0:
                            accum[horizon]["adverse"] += 1

        if total_maker < 20:
            return self._insufficient(product, dataset_label, f"Only {total_maker} maker fills.")

        maker_rates = {}
        taker_rates = {}
        for h in self.HORIZONS:
            mt = maker_adverse[h]["total"]
            maker_rates[h] = maker_adverse[h]["adverse"] / mt if mt > 0 else math.nan
            tt = taker_adverse[h]["total"]
            taker_rates[h] = taker_adverse[h]["adverse"] / tt if tt > 0 else math.nan

        # Verdict: if maker adverse rate > 50% at short horizons, makers are adversely selected
        short_rate = maker_rates.get(10, maker_rates.get(5, math.nan))
        if math.isnan(short_rate):
            verdict = "inconclusive"
            detail = "Not enough data at short horizons."
        elif short_rate > 0.55:
            verdict = "supported"
            detail = f"Maker fills show {short_rate:.1%} adverse selection at 10-tick horizon. Makers are getting picked off."
        elif short_rate < 0.45:
            verdict = "refuted"
            detail = f"Maker fills show only {short_rate:.1%} adverse selection at 10-tick horizon. Makers have timing advantage."
        else:
            verdict = "inconclusive"
            detail = f"Maker adverse rate {short_rate:.1%} at 10 ticks is near 50% — no clear signal."

        return ProbeResult(
            probe_id=self.spec.probe_id,
            family=self.spec.family,
            title=self.spec.title,
            hypothesis=self.spec.hypothesis,
            product=product,
            dataset=dataset_label,
            sample_size={"sessions": len(session_ledgers), "maker_fills": total_maker, "taker_fills": total_taker},
            metrics={
                "maker_adverse_rate_by_horizon": {str(h): round(r, 4) for h, r in maker_rates.items()},
                "taker_adverse_rate_by_horizon": {str(h): round(r, 4) for h, r in taker_rates.items()},
                "maker_adverse_counts": {str(h): maker_adverse[h] for h in self.HORIZONS},
            },
            verdict=verdict,
            confidence="high" if total_maker >= 200 else "medium" if total_maker >= 50 else "low",
            detail=detail,
        )


# ---------------------------------------------------------------------------
# PF03: Maker fill rate by inventory level
# ---------------------------------------------------------------------------

@register_probe
class MakerFillRateByInventory(Probe):
    spec = ProbeSpec(
        probe_id="pf03_maker_fill_rate_by_inventory",
        family="passive_fill",
        title="Maker fill rate conditioned on inventory level",
        hypothesis="High inventory reduces the rate of maker fills (skew pushes quotes away from BBO).",
        required_data=("strategy_fills", "traces", "prices"),
        metrics_produced=("fills_per_tick_by_inventory", "rate_drops_with_inventory"),
    )

    def run(self, session_ledgers, product, dataset_label=""):
        ticks_by_bucket: dict[str, int] = defaultdict(int)
        maker_fills_by_bucket: dict[str, int] = defaultdict(int)
        total_ticks = 0

        for sid, ledger in session_ledgers.items():
            traces = ledger["traces"]
            fills = ledger["strategy_fills"]
            if traces.empty:
                continue

            prod_traces = traces[traces["product"] == product]
            total_ticks += len(prod_traces)

            # Count ticks per abs-position bucket
            for _, row in prod_traces.iterrows():
                bucket = bucket_value(abs(row["position"]), ABS_POSITION_EDGES, ABS_POSITION_LABELS)
                ticks_by_bucket[bucket] += 1

            # Count maker fills per abs-position bucket
            if not fills.empty:
                enriched = enrich_fills(fills, traces, ledger["prices"])
                maker = enriched[
                    (enriched["product"] == product) & (enriched["strategy_role"] == "maker")
                ]
                for _, row in maker.iterrows():
                    bucket = bucket_value(abs(row["position_at_fill"]), ABS_POSITION_EDGES, ABS_POSITION_LABELS)
                    maker_fills_by_bucket[bucket] += 1

        if total_ticks < 100:
            return self._insufficient(product, dataset_label, f"Only {total_ticks} ticks.")

        fills_per_tick = {}
        for label in ABS_POSITION_LABELS:
            t = ticks_by_bucket.get(label, 0)
            f = maker_fills_by_bucket.get(label, 0)
            fills_per_tick[label] = round(f / t, 6) if t > 0 else 0.0

        # Check if rate decreases with higher inventory
        ordered_rates = [fills_per_tick.get(l, 0) for l in ABS_POSITION_LABELS]
        rate_drops = all(ordered_rates[i] >= ordered_rates[i + 1] for i in range(len(ordered_rates) - 1)) if len(ordered_rates) >= 2 else False

        total_fills = sum(maker_fills_by_bucket.values())
        if total_fills < 20:
            verdict = "inconclusive"
            detail = f"Only {total_fills} maker fills — not enough to assess inventory effect."
        elif rate_drops:
            verdict = "supported"
            detail = f"Maker fill rate drops with inventory: {fills_per_tick}."
        else:
            verdict = "refuted"
            detail = f"Maker fill rate does NOT consistently drop with inventory: {fills_per_tick}."

        return ProbeResult(
            probe_id=self.spec.probe_id,
            family=self.spec.family,
            title=self.spec.title,
            hypothesis=self.spec.hypothesis,
            product=product,
            dataset=dataset_label,
            sample_size={"sessions": len(session_ledgers), "total_ticks": total_ticks, "maker_fills": total_fills},
            metrics={
                "fills_per_tick_by_inventory": fills_per_tick,
                "ticks_by_inventory": dict(ticks_by_bucket),
                "maker_fills_by_inventory": dict(maker_fills_by_bucket),
                "rate_drops_with_inventory": rate_drops,
            },
            verdict=verdict,
            confidence="high" if total_fills >= 200 else "medium" if total_fills >= 50 else "low",
            detail=detail,
        )
