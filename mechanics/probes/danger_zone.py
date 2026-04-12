"""Danger zone probes.

Probes that identify book states or regimes associated with losses,
and test whether stub-avoidance logic was correct or wrong.
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
    SPREAD_EDGES,
    SPREAD_LABELS,
)


# ---------------------------------------------------------------------------
# DZ01: Wide spread loss association (stub hypothesis test)
# ---------------------------------------------------------------------------

@register_probe
class WideSpreadLossAssociation(Probe):
    spec = ProbeSpec(
        probe_id="dz01_wide_spread_loss",
        family="danger_zone",
        title="Wide spread / stub-like conditions and loss association",
        hypothesis="Wide spreads (stub-like books) are associated with subsequent per-tick losses.",
        required_data=("traces", "prices"),
        metrics_produced=("pnl_rate_by_spread_bucket", "wide_spread_harmful"),
    )

    FORWARD_TICKS = 10

    def run(self, session_ledgers, product, dataset_label=""):
        bucket_pnl: dict[str, list[float]] = defaultdict(list)
        total_ticks = 0

        for sid, ledger in session_ledgers.items():
            traces = ledger["traces"]
            prices = ledger["prices"]
            if traces.empty or prices.empty:
                continue

            # Build spread lookup for this session
            spread_lookup: dict[tuple[int, int], float] = {}
            prod_prices = prices[prices["product"] == product]
            for _, row in prod_prices.iterrows():
                bid1, ask1 = row["bid1"], row["ask1"]
                if not (math.isnan(bid1) or math.isnan(ask1)):
                    spread_lookup[(int(row["day"]), int(row["timestamp"]))] = ask1 - bid1

            fair_idx = build_fair_index(traces, product)
            if len(fair_idx) < self.FORWARD_TICKS + 1:
                continue

            pnl_values = fair_idx["mtm_pnl"].tolist()

            for i in range(len(fair_idx) - self.FORWARD_TICKS):
                row = fair_idx.iloc[i]
                key = (int(row["day"]), int(row["timestamp"]))
                s = spread_lookup.get(key, math.nan)
                if math.isnan(s):
                    continue

                pnl_change = pnl_values[i + self.FORWARD_TICKS] - pnl_values[i]
                label = bucket_value(s, SPREAD_EDGES, SPREAD_LABELS)
                bucket_pnl[label].append(pnl_change)
                total_ticks += 1

        if total_ticks < 100:
            return self._insufficient(product, dataset_label, f"Only {total_ticks} ticks with spread data.")

        pnl_by_bucket = {label: bucket_stats(vals) for label, vals in bucket_pnl.items()}

        # Check if wide spreads have lower PnL rate
        tight_stats = pnl_by_bucket.get(SPREAD_LABELS[1], pnl_by_bucket.get(SPREAD_LABELS[0], {}))
        wide_labels = [l for l in SPREAD_LABELS[2:] if l in pnl_by_bucket]

        tight_mean = tight_stats.get("mean", math.nan)
        wide_means = {l: pnl_by_bucket[l]["mean"] for l in wide_labels if not math.isnan(pnl_by_bucket[l]["mean"])}

        if math.isnan(tight_mean) or not wide_means:
            verdict = "inconclusive"
            detail = "Not enough data in tight or wide spread buckets."
        else:
            worst_wide_label = min(wide_means, key=wide_means.get) if wide_means else None
            worst_wide_mean = wide_means.get(worst_wide_label, math.nan)
            harmful = worst_wide_mean < tight_mean and worst_wide_mean < 0

            if harmful:
                verdict = "supported"
                detail = (
                    f"Wide spread ({worst_wide_label}) has negative PnL rate ({worst_wide_mean:.3f}/tick) "
                    f"vs tight ({tight_mean:.3f}/tick). Stub-avoidance may be justified in this regime."
                )
            elif worst_wide_mean < tight_mean:
                verdict = "inconclusive"
                detail = (
                    f"Wide spread has lower PnL rate ({worst_wide_mean:.3f}) vs tight ({tight_mean:.3f}), "
                    f"but still positive. Harm is moderate."
                )
            else:
                verdict = "refuted"
                detail = (
                    f"Wide spread does NOT show lower PnL rate than tight. "
                    f"Wide: {worst_wide_mean:.3f}, tight: {tight_mean:.3f}. "
                    f"Stub-avoidance may be unnecessary."
                )

        return ProbeResult(
            probe_id=self.spec.probe_id,
            family=self.spec.family,
            title=self.spec.title,
            hypothesis=self.spec.hypothesis,
            product=product,
            dataset=dataset_label,
            sample_size={"sessions": len(session_ledgers), "tick_observations": total_ticks},
            metrics={
                "pnl_rate_by_spread_bucket": pnl_by_bucket,
                "forward_ticks": self.FORWARD_TICKS,
                "wide_spread_harmful": harmful if "harmful" in dir() else None,
            },
            verdict=verdict,
            confidence="high" if total_ticks >= 5000 else "medium" if total_ticks >= 500 else "low",
            detail=detail,
        )


# ---------------------------------------------------------------------------
# DZ02: Session phase risk profile
# ---------------------------------------------------------------------------

@register_probe
class SessionPhaseRisk(Probe):
    spec = ProbeSpec(
        probe_id="dz02_session_phase_risk",
        family="danger_zone",
        title="Fill quality and PnL by session phase (early/mid/late)",
        hypothesis="Late-session fills have worse edge due to end-of-session dynamics.",
        required_data=("strategy_fills", "traces"),
        metrics_produced=("fill_edge_by_phase", "pnl_rate_by_phase", "late_phase_harmful"),
    )

    # Phase boundaries: early = first 10%, late = last 10%
    EARLY_FRAC = 0.10
    LATE_FRAC = 0.90

    def run(self, session_ledgers, product, dataset_label=""):
        phase_edges: dict[str, list[float]] = defaultdict(list)
        phase_pnl: dict[str, list[float]] = defaultdict(list)
        total_fills = 0
        total_ticks = 0

        for sid, ledger in session_ledgers.items():
            traces = ledger["traces"]
            fills = ledger["strategy_fills"]
            if traces.empty:
                continue

            fair_idx = build_fair_index(traces, product)
            n_ticks = len(fair_idx)
            if n_ticks < 10:
                continue

            # Phase labeling for each tick
            pnl_values = fair_idx["mtm_pnl"].tolist()
            for i in range(n_ticks - 1):
                frac = i / max(n_ticks - 1, 1)
                if frac <= self.EARLY_FRAC:
                    phase = "early"
                elif frac >= self.LATE_FRAC:
                    phase = "late"
                else:
                    phase = "mid"
                phase_pnl[phase].append(pnl_values[i + 1] - pnl_values[i])
                total_ticks += 1

            # Fill quality by phase
            if not fills.empty:
                # Build timestamp -> tick index
                ts_to_idx: dict[tuple[int, int], int] = {}
                for idx_i in range(n_ticks):
                    row = fair_idx.iloc[idx_i]
                    ts_to_idx[(int(row["day"]), int(row["timestamp"]))] = idx_i

                prod_fills = fills[fills["product"] == product]
                for _, fill in prod_fills.iterrows():
                    if math.isnan(fill.get("fill_vs_fair", math.nan)):
                        continue
                    key = (int(fill["day"]), int(fill["timestamp"]))
                    idx_i = ts_to_idx.get(key)
                    if idx_i is None:
                        continue

                    frac = idx_i / max(n_ticks - 1, 1)
                    if frac <= self.EARLY_FRAC:
                        phase = "early"
                    elif frac >= self.LATE_FRAC:
                        phase = "late"
                    else:
                        phase = "mid"

                    phase_edges[phase].append(fill["fill_vs_fair"])
                    total_fills += 1

        if total_fills < 20 and total_ticks < 100:
            return self._insufficient(product, dataset_label, f"Only {total_fills} fills, {total_ticks} ticks.")

        fill_edge_by_phase = {k: bucket_stats(v) for k, v in phase_edges.items()}
        pnl_rate_by_phase = {k: bucket_stats(v) for k, v in phase_pnl.items()}

        # Check if late phase is worse
        late_edge = fill_edge_by_phase.get("late", {}).get("mean", math.nan)
        mid_edge = fill_edge_by_phase.get("mid", {}).get("mean", math.nan)
        late_pnl = pnl_rate_by_phase.get("late", {}).get("mean", math.nan)
        mid_pnl = pnl_rate_by_phase.get("mid", {}).get("mean", math.nan)

        late_harmful = False
        if not math.isnan(late_edge) and not math.isnan(mid_edge):
            late_harmful = late_edge < mid_edge
        elif not math.isnan(late_pnl) and not math.isnan(mid_pnl):
            late_harmful = late_pnl < mid_pnl

        late_n = fill_edge_by_phase.get("late", {}).get("count", 0)
        if late_n < 10 and pnl_rate_by_phase.get("late", {}).get("count", 0) < 50:
            verdict = "inconclusive"
            detail = f"Not enough late-phase data: {late_n} fills."
        elif late_harmful:
            verdict = "supported"
            detail = (
                f"Late-session is worse. Fill edge: late={late_edge:.2f} vs mid={mid_edge:.2f}. "
                f"PnL rate: late={late_pnl:.4f} vs mid={mid_pnl:.4f}."
                if not math.isnan(late_edge) else
                f"Late-session PnL rate ({late_pnl:.4f}) is lower than mid ({mid_pnl:.4f})."
            )
        else:
            verdict = "refuted"
            detail = f"Late-session is NOT significantly worse. Fill edge: late={late_edge}, mid={mid_edge}."

        return ProbeResult(
            probe_id=self.spec.probe_id,
            family=self.spec.family,
            title=self.spec.title,
            hypothesis=self.spec.hypothesis,
            product=product,
            dataset=dataset_label,
            sample_size={"sessions": len(session_ledgers), "total_fills": total_fills, "total_ticks": total_ticks},
            metrics={
                "fill_edge_by_phase": fill_edge_by_phase,
                "pnl_rate_by_phase": pnl_rate_by_phase,
                "late_phase_harmful": late_harmful,
            },
            verdict=verdict,
            confidence="high" if total_fills >= 200 else "medium" if total_fills >= 50 else "low",
            detail=detail,
        )
