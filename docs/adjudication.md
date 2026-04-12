# Adjudication Engine — Developer Guide

## What this is

The judge + memory + next-action layer that closes the experiment loop.
Turns raw experiment results into structured verdicts, hypothesis outcomes,
frontier updates, learnings, and specific next-action recommendations.

Without this, the loop stays shallow: generate → run → look at numbers → guess.
With this, the loop becomes: generate → run → adjudicate → learn → focus.

## Architecture

```
adjudication/
  comparison.py     # Parent / frontier / family comparison engine
  preservation.py   # Task constraint audit
  attribution.py    # Mechanism attribution (where did the gain come from?)
  suspicion.py      # Noise and suspicion detection
  verdicts.py       # Candidate-level verdict engine (ties everything together)
  hypothesis.py     # Hypothesis-level adjudication
  frontier.py       # Frontier update logic with role-based slots
  learnings.py      # Structured learning extraction
  next_actions.py   # Next-action recommendation layer
  report.py         # JSON + markdown artifact generation
run_adjudication.py # CLI entrypoint
```

## Core Pipeline

```
Candidate Packet  ──┐
Parent Packet     ──┤──→  Comparison  ──→  Attribution  ──→  Suspicion  ──→  Verdict
Task Constraints  ──┤──→  Preservation Audit  ────────────────────────────↗
Frontier Set      ──┘──→  Frontier Comparison  ───────────────────────────↗

Verdict  ──→  Hypothesis Verdict  ──→  Learnings  ──→  Next Actions
         ──→  Frontier Updates
```

## 1. Candidate Verdict Engine (`verdicts.py`)

For each candidate, `adjudicate_candidate()` produces:

| Field | Description |
|-------|-------------|
| `candidate_id` | Packet identifier |
| `task_id` | Source task |
| `parent_id` | Parent strategy used |
| `family` | Family label |
| `pnl_mean/std/sharpe/p05/p50/p95` | Raw metrics |
| `positive_rate` | Fraction of profitable sessions |
| `confidence` | HIGH/MEDIUM/LOW |
| `promote_recommended` / `kill_recommended` | Packet gates |
| `emerald_mean/delta`, `tomato_mean/delta` | Per-product breakdown |
| `pnl_delta`, `sharpe_delta` | Delta vs parent |
| `vs_parent` | Full per-metric comparison |
| `vs_frontier` | Frontier comparison (beats/below) |
| `vs_family` | Best-in-family comparison |
| `preservation_audit` | Constraint violation check |
| `attribution` | Mechanism attribution |
| `suspicion` | Noise/suspicion flags |
| `verdict` | Label (see below) |
| `reason` | Explicit explanation |
| `mechanism_interpretation` | Attribution summary |
| `transfer_risk` | Risk assessment for official server |
| `recommended_next_action` | What to do with this result |

### Verdict Labels

| Label | Meaning |
|-------|---------|
| `frontier_challenger` | Beats frontier on key metrics, clean packet |
| `escalate` | Meaningful improvement, worth more sessions |
| `keep` | Minor improvement or neutral, informative |
| `control_success` | Control reproduced parent as expected |
| `control_failure` | Control deviated — noise floor is high |
| `reject` | Worse than parent or too suspect |
| `suspect_simulator_gain` | Gain exists but likely not real alpha |

### Verdict Assignment Logic

Priority order:
1. Control tasks → `control_success` or `control_failure`
2. Preservation violated → `reject`
3. High suspicion → `suspect_simulator_gain`
4. Kill recommended → `reject`
5. Both PnL and Sharpe worse → `reject`
6. Beats frontier on PnL + Sharpe, clean → `frontier_challenger`
7. Promote recommended + PnL up → `escalate`
8. Sharpe up > 1.0, PnL non-negative → `escalate`
9. PnL up, Sharpe not tanking → `keep` (or `suspect_simulator_gain` if medium suspicion)
10. Mixed → `keep`

## 2. Hypothesis Adjudication (`hypothesis.py`)

Judges the **hypothesis**, not just the candidate. Six checks:

| Check | Question |
|-------|----------|
| `mean_improved` | Did PnL go up? |
| `sharpe_improved` | Did Sharpe go up? |
| `intended_product_helped` | Did the target product improve? |
| `preserved_base_intact` | Were constraints respected? |
| `not_just_aggression` | Is the gain from quality, not volume? |
| `mechanism_matched` | Does the dominant mechanism match the card category? |

### Outcome Labels

| Label | Meaning |
|-------|---------|
| `validated` | All checks pass, positive signal |
| `partially_validated` | Some improvement but with side effects |
| `falsified` | Clearly didn't work |
| `inconclusive` | Not enough signal to tell |
| `informative_failure` | Failed but taught something specific |

Each hypothesis verdict includes extracted **lessons**: product-specific observations, mechanism outcomes, family-level patterns, and direction recommendations.

## 3. Preservation Audit (`preservation.py`)

Data-level audit comparing candidate vs parent outputs against task constraints.

| Check | What it detects |
|-------|-----------------|
| Product preservation | Non-target product changed >5% |
| Risk controls | Drawdown worsened >50% |
| Maker structure | Passive fill rate dropped >10% |
| Aggressiveness | Total fills increased >30% |
| Calibration | More than 10% PnL drift from parent |
| Near-parent | More than 5% PnL drift (cosmetic change only) |

Verdicts: `clean` (no issues), `suspect` (moderate violations), `violated` (critical violations).

## 4. Mechanism Attribution (`attribution.py`)

Six attribution functions, each producing a `{mechanism, magnitude, detail, description}` dict:

| Attribution | What it measures |
|-------------|-----------------|
| `product_shift` | Which product drove the PnL change, share of each |
| `role_shift` | Maker/taker balance change (passive fill rate) |
| `aggressiveness_change` | Total fill count increase/decrease |
| `fill_quality_change` | Fill-vs-fair edge shift per product |
| `risk_change` | Std and drawdown profile change |
| `sharpe_decomposition` | Mean vs std contribution to Sharpe change |

The dominant mechanism (highest magnitude) is reported. The Sharpe decomposition explicitly says whether a Sharpe improvement came from "higher mean" or "lower volatility."

## 5. Suspicion Detection (`suspicion.py`)

Seven suspicion checks:

| Flag | Condition |
|------|-----------|
| `noise_likely` | Gain / parent_std < 0.30 |
| `sharpe_mean_divergence` | Mean up but Sharpe dropped >2 points |
| `aggression_driven_gain` | >25% more fills + gain |
| `preservation_violated` | Critical constraint violations |
| `packet_quality_weakened` | Positive rate or p05 regressed |
| `single_product_fragile` | All gain from one product, other regressed |
| `calibration_suspect` | LOW confidence or calibration warnings |

Suspicion level: `clean` → `low` → `medium` → `high`. Verdicts with medium+ suspicion trigger `suspect_simulator_gain`.

## 6. Frontier Update Logic (`frontier.py`)

### Roles

| Role | Best candidate by |
|------|-------------------|
| `top_mean` | Highest mean PnL |
| `top_sharpe` | Highest Sharpe |
| `best_calibrated` | Highest positive_rate minus drawdown ratio |
| `best_maker_heavy` | Best Sharpe among maker-heavy family |
| `best_active_tomatoes` | Highest tomato PnL contribution |
| `best_control_anchor` | Most reliable promoted candidate |

### Rules

- Add challengers that pass verdict + suspicion checks
- Retire dominated candidates (worse on PnL + Sharpe + p05)
- Max 2 candidates per family (replace worst if over)
- Max 12 frontier size (replace weakest if over)
- Never remove sole calibration anchor without replacement

### Dominated Definition

A dominates B if A is >= B on all of (pnl_mean, sharpe, p05) and strictly > on at least one.

## 7. Learning Write-back (`learnings.py`)

Extracts from adjudicated batch:

| Category | Contents |
|----------|----------|
| `validated_mechanisms` | Hypotheses that worked |
| `falsified_mechanisms` | Hypotheses that failed |
| `suspicious_directions` | Gains that look fake |
| `promising_zones` | Partially validated directions worth more exploration |
| `dead_zones` | Hypotheses that failed 2+ times — direction exhausted |
| `family_lessons` | Per-family: what worked, what failed |
| `card_lessons` | Per-alpha-card outcome |
| `parent_lessons` | Per-parent: which mechanisms worked on this base |

Dead zone detection: if the same hypothesis_id fails across 2+ different parents, it's marked as exhausted.

## 8. Next-Action Recommendations (`next_actions.py`)

| Action Type | When | Priority |
|-------------|------|----------|
| `confirm_challenger` | Frontier challenger found | high |
| `explore_further` | Validated mechanism | high |
| `investigate_noise` | Control task failed | high |
| `promote_to_official` | Challenger added to frontier | medium |
| `stop_exploring` | Dead zone detected | medium |
| `product_gate` | Partial validation, single-product effect | medium |
| `try_different_parent` | Inconclusive on one parent only | low |
| `refine_preservation` | Suspect preservation issues | low |
| `no_action` | Nothing actionable | low |

Actions are sorted by priority. The `format_gpt_summary()` function renders actions + learnings into a Prosperity-GPT-ready markdown block.

## CLI Usage

```bash
# Full batch adjudication
python run_adjudication.py \
    --synthesis tmp/synthesis/synthesis_report.json \
    --candidates tmp/results/ \
    --bank tmp/bank \
    --out tmp/adjudication

# Single candidate vs parent
python run_adjudication.py \
    --candidate-packet tmp/results/mh07_packet.json \
    --parent-packet tmp/bank/73474_packet.json

# Compare to frontier only
python run_adjudication.py \
    --candidate-packet tmp/results/mh07_packet.json \
    --bank tmp/bank \
    --frontier-only

# Filter to rejected only
python run_adjudication.py \
    --synthesis report.json --candidates results/ --bank tmp/bank \
    --filter-verdict reject

# Filter to validated hypotheses only
python run_adjudication.py \
    --synthesis report.json --candidates results/ --bank tmp/bank \
    --filter-hypothesis validated

# GPT-ready summary
python run_adjudication.py \
    --synthesis report.json --candidates results/ --bank tmp/bank \
    --gpt-summary
```

## Output Artifacts

```
adjudication/
  candidate_verdicts.json    # Full per-candidate verdict reports
  candidate_verdicts.md      # Human-readable verdict summary
  hypothesis_verdicts.json   # Per-hypothesis outcomes
  hypothesis_verdicts.md     # Human-readable hypothesis summary
  frontier_updates.json      # Additions, retirements, role assignments
  frontier_updates.md        # Human-readable frontier changes
  batch_learnings.json       # Structured learnings by category
  batch_learnings.md         # Human-readable learning summary
  next_actions.json          # Prioritized action recommendations
  next_actions.md            # GPT-ready summary of what to do next
```

## Tests

```bash
python -m pytest tests/test_adjudication.py -v
```

70 tests covering: comparison engine, preservation audit, mechanism attribution,
suspicion detection, verdict classification, hypothesis adjudication, frontier
update logic, dominated/duplicate handling, learning extraction, next-action
generation, report serialization, and full pipeline integration.
