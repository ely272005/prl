# Discovery Engine — Developer Guide

## What this is

A regime + weakness discovery layer that scans backtest outputs, research
packets, and mechanics probe results to extract **specific, evidence-backed
candidate alpha opportunities** as structured "alpha cards."

This is NOT a strategy generator. It produces findings like:
- "maker edge peaks in spread bucket 4-6 on EMERALDS (2.1 above baseline, N=340)"
- "promoted candidates have 3.2x higher fill_vs_fair than rejected"
- "taker fills in down-trending TOMATOES destroy value"
- "inventory-reducing fills have better edge in deep_long positions"

These findings feed into:
- targeted prompts for Prosperity GPT,
- specific algorithm modifications,
- strategy generation constraints.

## Architecture

```
discovery/
  alpha_card.py      # AlphaCard dataclass, CardCounter, categories
  regimes.py         # Extended regime labeling + regime-conditioned stats
  comparison.py      # Winner vs loser comparison from research packets
  weakness.py        # 5 weakness scanners → alpha card generation
  scanner.py         # DiscoveryScanner orchestrator
  report.py          # JSON + markdown report generation
run_discovery.py     # CLI entrypoint
```

Data flows:

```
Backtest output dirs (sessions/*.csv)
  → engine.event_ledger.build_event_ledger()
  → session_ledgers
  → regimes.label_extended_regimes()  (per-tick, 10 dimensions)
  → regimes.build_regime_profile()    (regime-conditioned fill stats)
  → weakness scanners                 (5 patterns)
  → AlphaCard list (ranked by strength)
  → JSON/Markdown report

Bank directory (*_packet.json)
  → comparison.load_packets_from_bank()
  → comparison.run_comparison()        (promoted vs rejected)
  → winner trait scanner              (→ alpha cards)
  → included in report

Mechanics probe report (mechanics_report.json)
  → probe-driven scanner             (→ alpha cards)
```

## Regime Dimensions (10 total)

| # | Dimension | Source | Labels |
|---|-----------|--------|--------|
| 1 | spread_bucket | bid-ask width | <2, 2-4, 4-6, 6-8, 8+ |
| 2 | position_bucket | signed position | deep_short, short, flat, long, deep_long |
| 3 | abs_position_bucket | |position| | low, medium, high |
| 4 | volatility_regime | 50-tick rolling fair value change | low, medium, high |
| 5 | spread_regime | percentile-based spread | low, medium, high |
| 6 | inventory_regime | percentile-based |position| | low, medium, high |
| 7 | session_phase | tick progress through session | early, mid, late |
| 8 | trend_10 | 10-tick fair value direction | up, flat, down |
| 9 | fair_vs_mid | sign of fair_value - mid_price | above_mid, at_mid, below_mid |
| 10 | maker_friendly | composite (spread + volatility) | maker_friendly, taker_friendly, neutral |

Dimensions 4-7 come from the existing `analytics/regime_analysis.py`.
Dimensions 1-3, 8-10 are new in this module.

## Weakness Scanners (5 patterns)

### 1. Regime Edge Scanner
Scans every (product × regime_dimension × role) combination. For each regime
label, compares fill_vs_fair to the overall baseline. Flags labels where:
- absolute difference > 0.8
- relative difference > 25%
- median is consistent with mean
- sample >= 30 fills

Produces: `regime_edge` or `danger_refinement` cards.

### 2. Role Mismatch Scanner
For each (product × dimension × label), compares maker edge vs taker edge.
Flags when one role dominates by >= 2x ratio with sufficient sample.

Produces: `role_mismatch` cards.

### 3. Winner Trait Scanner
Loads research packets from the bank, splits into promoted vs rejected,
and compares 14 key metrics. Flags metrics with effect size >= 0.8 (Cohen's d).
Also compares family-level performance.

Produces: `winner_trait` cards.

### 4. Probe-Driven Scanner
Reads mechanics probe results and converts probes with verdict
"supported" or "refuted" (confidence >= medium) into alpha cards.

Produces: cards matching the probe's family category.

### 5. Inventory Exploit Scanner
Compares fill quality between opposite position buckets (long vs short,
deep_long vs deep_short). Flags significant asymmetries.

Produces: `inventory_exploit` cards.

## Alpha Card Structure

Each card clearly separates three layers:

| Layer | Field | What it contains |
|-------|-------|-----------------|
| Fact | observed_fact | What the data shows (no interpretation) |
| Inference | interpretation | What this might mean |
| Speculation | suggested_exploit | What strategy could exploit this |

Additional fields:

| Field | Purpose |
|-------|---------|
| card_id | Sequential ID (RE01, WT01, etc.) |
| category | regime_edge, role_mismatch, bot_weakness, danger_refinement, winner_trait, inventory_exploit |
| products | Which products are affected |
| regime_definition | The specific conditions that define this opportunity |
| evidence | Supporting metrics |
| baseline | What the comparison is against |
| sample_size | How much data backs this finding |
| confidence | high / medium / low |
| strength | Numeric ranking score (higher = stronger finding) |
| candidate_strategy_style | What kind of strategy fits |
| recommended_experiment | What to try next |

## Ranking and Filtering

Cards are ranked by `strength = |effect| × log(1 + sample_size) × confidence_multiplier`.
After ranking, duplicates (same category + product + regime) are deduplicated,
keeping only the stronger card. Output is capped at `max_cards` (default 20).

## CLI Usage

```bash
# Full discovery with bank comparison
python run_discovery.py tmp/bank/73474_output --bank tmp/bank

# With probe results from previous mechanics run
python run_discovery.py tmp/bank/73474_output --bank tmp/bank \
    --probes tmp/bank/73474_output/mechanics/mechanics_report.json

# One product only
python run_discovery.py tmp/bank/73474_output --bank tmp/bank --product TOMATOES

# Multiple output directories
python run_discovery.py tmp/bank/73474_output tmp/bank/73479_output --bank tmp/bank

# Custom output and card limit
python run_discovery.py tmp/bank/73474_output --bank tmp/bank \
    --out tmp/discovery_results --max-cards 10

# Session-only (no bank comparison)
python run_discovery.py tmp/bank/73474_output
```

## Output Files

- `discovery_report.json` — structured JSON with all alpha cards, comparison data, classification
- `discovery_report.md` — readable markdown with ranked findings, evidence, experiments

## How this feeds future work

1. **Prosperity GPT prompts**: Hand top alpha cards directly.
   "Here are the top 5 alpha cards. Build a strategy for card RE01."

2. **Strategy generation**: Use regime definitions as constraints.
   "Attack this TOMATOES regime without harming the EMERALDS base."

3. **Experiment design**: Each card has a `recommended_experiment`.
   Run those experiments to validate or refute the card.

4. **Descendant screening**: Use winner traits as screening criteria.
   "Only keep descendants that match the promoted candidate profile."

5. **Danger avoidance**: Use danger refinement cards as constraints.
   "Avoid these specific regimes identified as harmful."

## Confidence Levels

- **high**: >= 200 fills in the relevant bucket
- **medium**: >= 50 fills
- **low**: < 50 fills (but above the 30-fill minimum threshold)

Cards classified as "strong" have confidence >= medium AND strength > 2.0.
Everything else is "speculative."

## How to extend

### Adding a new scanner

1. Write a function in `weakness.py` with signature:
   ```python
   def scan_my_pattern(
       regime_stats: dict,
       counter: CardCounter,
   ) -> list[AlphaCard]:
   ```

2. Call it from `run_all_scanners()`.

### Adding a new regime dimension

1. Add the label logic in `regimes.label_extended_regimes()`.
2. Add the dimension name to `REGIME_DIMENSIONS`.
3. The regime edge scanner will automatically pick it up.

### Adding new comparison metrics

1. Add a tuple to `COMPARISON_METRICS` in `comparison.py`.
2. The winner trait scanner will automatically pick it up.
