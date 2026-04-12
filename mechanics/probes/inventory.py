"""Inventory pressure probes.

Probes that examine how inventory level affects PnL growth,
whether inventory skew improves outcomes, and where clearing helps.
"""
from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd

from mechanics.probe_spec import Probe, ProbeSpec, ProbeResult, register_probe
from mechanics.probes._helpers import (
    build_fair_index,
    bucket_value,
    bucket_stats,
    POSITION_EDGES,
    POSITION_LABELS,
)


# ---------------------------------------------------------------------------
# INV01: PnL rate by inventory level
# ---------------------------------------------------------------------------

@register_probe
class PnlByInventoryLevel(Probe):
    spec = ProbeSpec(
        probe_id="inv01_pnl_by_inventory_level",
        family="inventory",
        title="PnL growth rate conditioned on inventory level",
        hypothesis="High absolute inventory reduces per-tick PnL growth (inventory drag).",
        required_data=("traces",),
        metrics_produced=("pnl_rate_by_position_bucket", "drag_at_high_inventory"),
    )

    FORWARD_TICKS = 10

    def run(self, session_ledgers, product, dataset_label=""):
        bucket_pnl_changes: dict[str, list[float]] = defaultdict(list)
        total_ticks = 0

        for sid, ledger in session_ledgers.items():
            traces = ledger["traces"]
            if traces.empty:
                continue

            fair_idx = build_fair_index(traces, product)
            if len(fair_idx) < self.FORWARD_TICKS + 1:
                continue

            positions = fair_idx["position"].tolist()
            pnl_values = fair_idx["mtm_pnl"].tolist()

            for i in range(len(fair_idx) - self.FORWARD_TICKS):
                pos = positions[i]
                pnl_now = pnl_values[i]
                pnl_future = pnl_values[i + self.FORWARD_TICKS]
                delta_pnl = pnl_future - pnl_now

                label = bucket_value(float(pos), POSITION_EDGES, POSITION_LABELS)
                bucket_pnl_changes[label].append(delta_pnl)
                total_ticks += 1

        if total_ticks < 100:
            return self._insufficient(product, dataset_label, f"Only {total_ticks} tick observations.")

        pnl_by_bucket = {label: bucket_stats(vals) for label, vals in bucket_pnl_changes.items()}

        # Compute flat baseline
        flat_stats = pnl_by_bucket.get("flat", {})
        flat_mean = flat_stats.get("mean", math.nan)

        # Check if extreme positions have lower PnL rate than flat
        deep_short_mean = pnl_by_bucket.get("deep_short", {}).get("mean", math.nan)
        deep_long_mean = pnl_by_bucket.get("deep_long", {}).get("mean", math.nan)

        extreme_means = [m for m in [deep_short_mean, deep_long_mean] if not math.isnan(m)]
        drag_present = (
            not math.isnan(flat_mean) and
            len(extreme_means) > 0 and
            all(m < flat_mean for m in extreme_means)
        )

        if math.isnan(flat_mean) or len(extreme_means) == 0:
            verdict = "inconclusive"
            detail = "Not enough data in flat or extreme inventory buckets."
        elif drag_present:
            worst_extreme = min(extreme_means)
            drag_magnitude = flat_mean - worst_extreme
            verdict = "supported"
            detail = (
                f"Inventory drag confirmed. Flat PnL rate: {flat_mean:.3f}/tick. "
                f"Worst extreme: {worst_extreme:.3f}/tick. Drag: {drag_magnitude:.3f}/tick."
            )
        else:
            verdict = "refuted"
            detail = f"Extreme inventory does NOT show lower PnL rate than flat. Flat: {flat_mean:.3f}, extremes: {extreme_means}."

        return ProbeResult(
            probe_id=self.spec.probe_id,
            family=self.spec.family,
            title=self.spec.title,
            hypothesis=self.spec.hypothesis,
            product=product,
            dataset=dataset_label,
            sample_size={"sessions": len(session_ledgers), "tick_observations": total_ticks},
            metrics={
                "pnl_rate_by_position_bucket": pnl_by_bucket,
                "forward_ticks": self.FORWARD_TICKS,
                "drag_at_high_inventory": drag_present,
            },
            verdict=verdict,
            confidence="high" if total_ticks >= 5000 else "medium" if total_ticks >= 500 else "low",
            detail=detail,
        )


# ---------------------------------------------------------------------------
# INV02: Fill quality by inventory state
# ---------------------------------------------------------------------------

@register_probe
class FillQualityByInventory(Probe):
    spec = ProbeSpec(
        probe_id="inv02_fill_quality_by_inventory",
        family="inventory",
        title="Fill quality conditioned on inventory sign and trade direction",
        hypothesis="Inventory skew improves fill quality by leaning into inventory-reducing fills.",
        required_data=("strategy_fills", "traces", "prices"),
        metrics_produced=("edge_by_inventory_and_direction", "skew_helps"),
    )

    def run(self, session_ledgers, product, dataset_label=""):
        from mechanics.probes._helpers import enrich_fills

        # Group: (inventory_sign, trade_side) -> list of fill_vs_fair
        groups: dict[str, list[float]] = defaultdict(list)
        total_fills = 0

        for sid, ledger in session_ledgers.items():
            fills = ledger["strategy_fills"]
            if fills.empty:
                continue
            enriched = enrich_fills(fills, ledger["traces"], ledger["prices"])
            prod = enriched[enriched["product"] == product]

            for _, row in prod.iterrows():
                if math.isnan(row.get("fill_vs_fair", math.nan)):
                    continue

                pos = row["position_at_fill"]
                side = row["strategy_side"]
                total_fills += 1

                # Classify inventory sign
                if pos > 2:
                    inv_sign = "long"
                elif pos < -2:
                    inv_sign = "short"
                else:
                    inv_sign = "flat"

                # Is this fill inventory-reducing?
                reducing = (inv_sign == "long" and side == "sell") or (inv_sign == "short" and side == "buy")
                increasing = (inv_sign == "long" and side == "buy") or (inv_sign == "short" and side == "sell")

                if reducing:
                    groups["reducing"].append(row["fill_vs_fair"])
                elif increasing:
                    groups["increasing"].append(row["fill_vs_fair"])
                else:
                    groups["neutral"].append(row["fill_vs_fair"])

        if total_fills < 20:
            return self._insufficient(product, dataset_label, f"Only {total_fills} fills.")

        stats = {k: bucket_stats(v) for k, v in groups.items()}

        reducing_mean = stats.get("reducing", {}).get("mean", math.nan)
        increasing_mean = stats.get("increasing", {}).get("mean", math.nan)

        skew_helps = (
            not math.isnan(reducing_mean) and
            not math.isnan(increasing_mean) and
            reducing_mean > increasing_mean
        )

        reducing_n = stats.get("reducing", {}).get("count", 0)
        increasing_n = stats.get("increasing", {}).get("count", 0)

        if reducing_n < 10 or increasing_n < 10:
            verdict = "inconclusive"
            detail = f"Not enough fills: reducing={reducing_n}, increasing={increasing_n}."
        elif skew_helps:
            gap = reducing_mean - increasing_mean
            verdict = "supported"
            detail = (
                f"Inventory-reducing fills have {gap:.2f} better edge. "
                f"Reducing: {reducing_mean:.2f} (n={reducing_n}), increasing: {increasing_mean:.2f} (n={increasing_n})."
            )
        else:
            verdict = "refuted"
            detail = (
                f"Inventory-reducing fills do NOT have better edge. "
                f"Reducing: {reducing_mean:.2f}, increasing: {increasing_mean:.2f}."
            )

        return ProbeResult(
            probe_id=self.spec.probe_id,
            family=self.spec.family,
            title=self.spec.title,
            hypothesis=self.spec.hypothesis,
            product=product,
            dataset=dataset_label,
            sample_size={"sessions": len(session_ledgers), "total_fills": total_fills},
            metrics={
                "edge_by_inventory_and_direction": stats,
                "skew_helps": skew_helps,
            },
            verdict=verdict,
            confidence="high" if total_fills >= 200 else "medium" if total_fills >= 50 else "low",
            detail=detail,
        )
