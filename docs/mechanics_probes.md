# Mechanics Probe Engine — Developer Guide

## What this is

A truth-finding layer that runs targeted experiments on backtest session data.
Each probe tests a specific hypothesis about market/simulator behavior and
produces a structured verdict with evidence.

This is NOT a strategy optimizer. It produces reusable findings like:
- "maker edge is strongest in spread bucket Y"
- "taking in TOMATOES at width <= 2 seems good only in state X"
- "stub-avoidance logic was wrong; the harmful region is narrower than assumed"

These findings feed into alpha discovery, prompt engineering for Prosperity GPT,
and targeted strategy modifications.

## Architecture

```
mechanics/
  probe_spec.py         # ProbeSpec, ProbeResult, Probe base class, registry
  runner.py             # ProbeRunner — loads data, executes probes
  report.py             # JSON + markdown report generation
  probes/
    __init__.py          # Imports all families to populate registry
    _helpers.py          # Enrichment, bucketing, fair-value lookups
    passive_fill.py      # pf01, pf02, pf03
    taking.py            # tk01, tk02, tk03
    inventory.py         # inv01, inv02
    danger_zone.py       # dz01, dz02
run_mechanics.py         # CLI entrypoint
```

Data flows through the existing event ledger:
```
Backtest output dir (sessions/*.csv)
  → engine.event_ledger.build_event_ledger()
  → session_ledgers (per-session DataFrames)
  → Probe.run(session_ledgers, product)
  → ProbeResult
  → JSON/Markdown report
```

## Available Probes

### Passive Fill (passive_fill)

| ID | Title | Tests |
|----|-------|-------|
| pf01_spread_vs_maker_edge | Spread bucket vs maker fill edge | Whether wider spreads give makers better fill_vs_fair |
| pf02_maker_adverse_timing | Maker adverse selection at multiple horizons | Whether fair value moves against makers after fills |
| pf03_maker_fill_rate_by_inventory | Maker fill rate by inventory level | Whether high inventory reduces passive fill probability |

### Taking (taking)

| ID | Title | Tests |
|----|-------|-------|
| tk01_take_edge_by_distance | Taker edge by distance from fair | Whether further-from-fair takes have more edge |
| tk02_take_edge_by_spread | Taker edge by spread state | Which spread regime is best/worst for taking |
| tk03_take_direction_vs_trend | Taker edge: with vs against trend | Whether mean-reversion or momentum taking is better |

### Inventory (inventory)

| ID | Title | Tests |
|----|-------|-------|
| inv01_pnl_by_inventory_level | PnL rate by inventory level | Whether high abs-inventory causes drag on per-tick PnL |
| inv02_fill_quality_by_inventory | Fill quality by inventory state | Whether inventory-reducing fills have better edge |

### Danger Zone (danger_zone)

| ID | Title | Tests |
|----|-------|-------|
| dz01_wide_spread_loss | Wide spread / stub-like loss association | Whether wide-spread ticks precede losses |
| dz02_session_phase_risk | Session phase risk profile | Whether late-session fills have worse edge |

## How to add a new probe

1. Pick a family module (or create a new one under `mechanics/probes/`).

2. Define a class that inherits from `Probe` with a `ProbeSpec` class attribute:

```python
from mechanics.probe_spec import Probe, ProbeSpec, ProbeResult, register_probe

@register_probe
class MyNewProbe(Probe):
    spec = ProbeSpec(
        probe_id="xx01_my_new_probe",   # family_prefix + number + short_name
        family="my_family",
        title="What this probe measures",
        hypothesis="The specific claim being tested.",
        required_data=("strategy_fills", "traces"),  # which ledger keys needed
        metrics_produced=("metric_a", "metric_b"),
    )

    def run(self, session_ledgers, product, dataset_label=""):
        # Iterate sessions, compute metrics, form verdict
        ...
        return ProbeResult(
            probe_id=self.spec.probe_id,
            family=self.spec.family,
            title=self.spec.title,
            hypothesis=self.spec.hypothesis,
            product=product,
            dataset=dataset_label,
            sample_size={"sessions": N, "fills": M},
            metrics={"metric_a": value_a},
            verdict="supported",      # or refuted / inconclusive / insufficient_data
            confidence="high",        # or medium / low
            detail="Human-readable explanation.",
        )
```

3. If it's a new family, import it in `mechanics/probes/__init__.py`.

4. Add a test in `tests/test_probes.py`.

## Verdicts

- **supported**: Evidence is consistent with the hypothesis.
- **refuted**: Evidence contradicts the hypothesis.
- **inconclusive**: Evidence is mixed or too noisy to decide.
- **insufficient_data**: Not enough data to evaluate (< 20 fills or < 100 ticks typically).

## Confidence levels

- **high**: >= 200 fills or >= 5000 tick observations
- **medium**: >= 50 fills or >= 500 tick observations
- **low**: Below medium thresholds

## Available data per session

Each session ledger contains:

- `traces`: DataFrame — day, timestamp, product, fair_value, position, cash, mtm_pnl
- `prices`: DataFrame — day, timestamp, product, bid1, ask1, mid_price
- `all_trades`: DataFrame — all fills including bot-vs-bot
- `strategy_fills`: DataFrame — enriched with fair_value, mid_price, fill_vs_fair, fill_vs_mid, strategy_role, strategy_side

The `_helpers.py` module provides:
- `enrich_fills()` — adds spread and position_at_fill to strategy fills
- `build_fair_index()` — ordered fair value time series for a product
- `fair_change_at_fill()` — forward-looking fair value change
- `fair_change_before_fill()` — backward-looking fair value change
- `bucket_value()` — assign values to named buckets
- `bucket_stats()` — compute mean/median/std/count

## CLI usage

```bash
# List all probes
python run_mechanics.py --list

# Run all probes on one backtest output
python run_mechanics.py tmp/bank/73474_output

# Run one family
python run_mechanics.py tmp/bank/73474_output --family taking

# Run one probe on one product
python run_mechanics.py tmp/bank/73474_output --probe tk01_take_edge_by_distance --product TOMATOES

# Multiple input directories (merges sessions)
python run_mechanics.py tmp/bank/73474_output tmp/bank/73479_output

# Custom output location
python run_mechanics.py tmp/bank/73474_output --out tmp/my_results
```

## Output files

- `mechanics_report.json` — structured JSON with all results, metrics, verdicts
- `mechanics_report.md` — readable markdown with findings summary

## How this feeds future work

1. **Alpha discovery**: Supported/refuted verdicts become constraints for strategy search.
   Example: if pf01 shows maker edge peaks in spread 4-6, future strategies should
   target that regime for making.

2. **Prosperity GPT prompts**: Include probe findings as hard evidence in handoff prompts.
   "Evidence: taking at width < 2 is harmful on TOMATOES (probe tk01, refuted)."

3. **Strategy modification**: Use inventory probes to tune skew parameters,
   danger zone probes to set spread thresholds, taking probes to set take_width corridors.

4. **Comparative analysis**: Run the same probes across multiple strategies to see
   which strategies align with favorable mechanics and which fight them.
