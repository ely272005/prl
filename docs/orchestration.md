# Campaign Orchestrator — Developer Guide

## What this is

The decision-and-execution planning layer that sits between research learnings
and actual experiment runs. Turns Phase 5 adjudication outputs into explicit
run plans, official-testing recommendations, champion management, and
Prosperity GPT handoff briefs.

Without this, the loop produces verdicts and learnings but nobody decides
what to do with them. With this, the machine says: "run this next, test this
officially, protect this champion, kill this branch, tell GPT exactly this."

## Architecture

```
orchestration/
  campaigns.py       # Campaign abstraction, creation, types
  routing.py         # Verdict → campaign action routing rules
  champions.py       # Champion table management (7 roles)
  redundancy.py      # Near-duplicate, dead-zone, diversity checks
  allocation.py      # Budget-aware exploit/explore allocation
  run_plan.py        # Run-plan generator (main entry point)
  official_queue.py  # Official-testing queue + memo
  handoff.py         # Prosperity GPT handoff packaging
  history.py         # Campaign history persistence
  report.py          # JSON + markdown artifact generation
run_orchestration.py # CLI entrypoint
```

## Core Pipeline

```
Phase 5 Outputs  ──┐
  (verdicts,       │
   learnings,      ├──→  Route  ──→  Create Campaigns  ──→  Check Redundancy
   next_actions,   │                                            │
   frontier)       │                                            ▼
                   │                                    Allocate Budget
Bank Packets     ──┘                                    (exploit/explore)
                                                            │
                                                            ▼
                                          Build Run Plan + Official Queue
                                                            │
                                              ┌─────────────┼─────────────┐
                                              ▼             ▼             ▼
                                          Champions    Handoff Briefs  Reports
                                          Table        (per campaign)  (13 files)
```

## 1. Campaign Abstraction (`campaigns.py`)

### Campaign Types

| Type | Novelty | Scope | Default Budget | Exploit/Explore |
|------|---------|-------|---------------|-----------------|
| `exploration` | high | broad | 6 cand, 20 sess | explore |
| `confirmation` | low | narrow | 3 cand, 50 sess | exploit |
| `official_gate` | none | very narrow | 2 cand, 50 sess | exploit |
| `calibration` | none | narrow | 4 cand, 30 sess | exploit |
| `champion_defense` | none | very narrow | 2 cand, 30 sess | exploit |
| `control_batch` | none | narrow | 3 cand, 20 sess | exploit |

### Campaign Schema

| Field | Type | Description |
|-------|------|-------------|
| `campaign_id` | str | `CAM-YYYYMMDD-NNN` |
| `title` | str | Human-readable title |
| `campaign_type` | str | One of 6 types above |
| `objective` | str | Why this campaign exists |
| `status` | str | planned / active / completed / abandoned |
| `priority` | str | high / medium / low |
| `family` | str | Strategy family targeted |
| `target_mechanism` | str | Mechanism being attacked |
| `product_scope` | list | EMERALDS / TOMATOES / both |
| `allowed_parents` | list | Parent strategy IDs |
| `forbidden_directions` | list | Dead zone hypothesis IDs |
| `preservation_constraints` | list | What must not change |
| `success_criteria` | str | What counts as success |
| `failure_criteria` | str | What counts as failure |
| `budget` | dict | max_candidates, max_sessions_per_candidate |
| `exploit_explore` | str | exploit / explore |
| `novelty_tolerance` | str | high / low / none |
| `scope` | str | broad / narrow / very_narrow |
| `source_next_actions` | list | Phase 5 actions that created this |
| `allocated_candidates` | int | Budget actually allocated |
| `planned_roles` | list | Roles for each candidate slot |

### Campaign Creation from Phase 5 Actions

| Action Type | → Campaign Type | Priority |
|-------------|----------------|----------|
| `confirm_challenger` | confirmation | high |
| `explore_further` | exploration | high |
| `investigate_noise` | calibration | high |
| `product_gate` | exploration | medium |
| `try_different_parent` | exploration | low |
| frontier additions | champion_defense | medium |
| `stop_exploring` | (no campaign — skipped) | — |
| `no_action` | (no campaign — skipped) | — |

Dead zone hypotheses are automatically excluded from campaign creation.

## 2. Routing Logic (`routing.py`)

Explicit, rule-based routing from verdict labels to actions:

| Verdict | Action | Official Eligible | Campaign Type |
|---------|--------|-------------------|---------------|
| `frontier_challenger` | official_gate_shortlist | YES | official_gate |
| `escalate` | confirmation_campaign | YES if low transfer risk | confirmation |
| `keep` | archive | NO | — |
| `reject` | dead_branch | NO | — |
| `suspect_simulator_gain` | calibration_only | NO | calibration |
| `control_success` | no_action | NO | — |
| `control_failure` | calibration_campaign | NO | calibration |

### Transfer Risk Refinement

`escalate` candidates with low transfer risk get promoted to official-eligible:
- If `transfer_risk` contains "Low transfer risk" → `official_eligible = True`
- Otherwise stays `confirmation_campaign` only

### Dead Zone Warnings

If a candidate's source hypothesis is in a dead zone, the routing decision
includes a `dead_zone_warning` field.

## 3. Champion Management (`champions.py`)

### Champion Roles

| Role | Selection Criterion |
|------|-------------------|
| `overall_champion` | Best composite (Sharpe + mean/1000) |
| `best_sharpe` | Highest Sharpe-like |
| `best_mean` | Highest mean PnL |
| `best_calibrated` | Highest (positive_rate - drawdown_ratio) |
| `best_maker_heavy` | Best Sharpe among packets with "maker" in family name |
| `best_active_tomatoes` | Highest tomato PnL (if > 0) |
| `best_anchor` | Highest positive_rate among promoted packets |

### Champion Status

| Status | Meaning |
|--------|---------|
| `active` | Current holder of the role |
| `superseded` | Replaced by a better candidate |
| `retired` | Manually retired (no longer competitive) |
| `preserved` | Kept for calibration/anchor reasons |

### Operations

- `build_champion_table(packets)` — initial build from frontier
- `update_champion_table(old, new_packets)` — supersede old, install new
- `promote_champion(table, id, role)` — manual promotion
- `retire_champion(table, id, reason)` — manual retirement
- `preserve_champion(table, id, reason)` — keep for calibration

## 4. Redundancy Control (`redundancy.py`)

Four checks, each producing flagged items:

| Check | What it detects |
|-------|-----------------|
| Near-duplicate | Candidate within 5% of frontier member on key metrics |
| Falsified repeat | Campaign targeting a dead-zone hypothesis |
| Low diversity | Exploration campaign with single parent |
| Cross-campaign overlap | Two campaigns attacking same mechanism + family |

### Near-Duplicate Detection

Compares on: `pnl_mean`, `sharpe`, `positive_rate`, `emerald_mean`, `tomato_mean`.
If average relative difference < 5%, flagged as near-duplicate.

### Recommendations

Each issue produces an action:
- `skip` — do not spend a slot on this candidate
- `cancel_campaign` — campaign targets dead zone
- `add_diversity` — campaign needs more parent diversity
- `merge_campaigns` — two campaigns should be combined

## 5. Budget-Aware Allocation (`allocation.py`)

### Default Budgets

| Parameter | Default |
|-----------|---------|
| Total local candidates | 20 |
| Total official tests | 3 |
| Max campaigns | 6 |
| Exploit ratio | 70% |

### Campaign Sizing

| Type | Default Size | High Priority | Low Priority |
|------|-------------|---------------|-------------|
| confirmation | 3 | 4 | 2 |
| exploration | 6 | 7 | 5 |
| official_gate | 2 | 3 | 1 |
| calibration | 4 | 5 | 3 |
| champion_defense | 2 | 3 | 1 |
| control_batch | 3 | 4 | 2 |

### Allocation Logic

1. Split campaigns into exploit vs explore
2. Calculate exploit_budget = total × exploit_ratio
3. Calculate explore_budget = total × (1 - exploit_ratio)
4. Within each category, allocate by priority (high first)
5. If one category has leftover budget, overflow to the other
6. Trim to max_campaigns (drop lowest priority)

## 6. Official-Testing Queue (`official_queue.py`)

### Ranking Score

```
+100  frontier_challenger verdict
 +50  escalate verdict
 +30  HIGH confidence
 +15  MEDIUM confidence
 +20  low transfer risk
 -15  high transfer risk
 +N   Sharpe (capped at 20)
 +N   positive_rate × 30
 +10  p05 > 0
 -10  p05 < -1000
  -5  low suspicion
 -25  medium suspicion
 -60  high suspicion
 +15  promote gate pass
```

### Official Roles

| Role | Assigned When |
|------|--------------|
| `safest_challenger` | Rank 1 with HIGH confidence + low transfer risk |
| `highest_ceiling` | Best local metrics but higher risk |
| `champion_replacement` | Beats current champion on PnL + Sharpe |
| `control_candidate` | For A/B comparison on official server |
| `calibration_anchor` | Known-good reproduction for baseline |

### Official Memo

For each shortlist, a memo explains:
- What the queue contains
- Why each candidate is there
- What risk each carries
- Which is safest, which has highest ceiling
- Which is control/calibration

## 7. Run Plan (`run_plan.py`)

The main entry point. `build_run_plan()` chains all modules:

1. Route candidates → routing decisions
2. Create campaigns from next actions
3. Check redundancy → filter dead-zone campaigns
4. Allocate budget (exploit/explore split)
5. Assign candidate roles per campaign
6. Collect skipped actions

### Candidate Roles

| Role | Description |
|------|-------------|
| `challenger` | Primary experimental candidate |
| `near_twin_control` | Near-parent control for noise measurement |
| `calibration_anchor` | Exact-parent reproduction |
| `exploration_probe` | Broad-search variant |
| `mechanism_variant` | Same mechanism, different parameter |
| `champion_defender` | Defending current champion |

### Role Assignment by Campaign Type

| Campaign Type | Slot 1 | Slot 2 | Remaining |
|--------------|--------|--------|-----------|
| confirmation | challenger | calibration_anchor | mechanism_variant |
| exploration | exploration_probe | ... probes ... | near_twin_control |
| official_gate | challenger | calibration_anchor | — |
| calibration | calibration_anchor | ... anchors ... | — |
| champion_defense | champion_defender | near_twin_control | — |
| control_batch | near_twin_control | ... controls ... | — |

## 8. Prosperity GPT Handoff (`handoff.py`)

Each campaign handoff brief contains:

| Section | Contents |
|---------|----------|
| Objective | Campaign objective |
| Target mechanism | What to attack |
| Allowed parents | Which parent IDs to use |
| Constraints | Preserve / do not touch |
| Dead zones | Already-falsified directions |
| Validated mechanisms | What has been proven to work |
| Family lessons | What worked/failed for this family |
| Champion context | Current champion metrics |
| Success/failure criteria | What counts |
| Requested candidates | Role + notes for each slot |
| Previously falsified | Reference for avoiding dead ends |

## 9. Campaign History (`history.py`)

Stores per-campaign:

| Field | Description |
|-------|-------------|
| campaign_id | Reference |
| title / type / objective | What it was |
| budget_used | How many candidate slots were spent |
| candidate_verdicts | Summary of what happened |
| hypothesis_verdicts | What hypotheses were tested |
| was_worth_budget | Post-hoc assessment |

## CLI Usage

```bash
# Build full run plan from Phase 5 output
python run_orchestration.py plan \
    --adjudication tmp/adjudication \
    --bank tmp/bank \
    --out tmp/orchestration

# With custom budget and exploit ratio
python run_orchestration.py plan \
    --adjudication tmp/adjudication \
    --bank tmp/bank \
    --budget '{"total_local_candidates": 30, "total_official_tests": 5}' \
    --exploit-ratio 0.60

# Show current champion table
python run_orchestration.py champions --bank tmp/bank

# Show official-testing shortlist
python run_orchestration.py official \
    --adjudication tmp/adjudication \
    --bank tmp/bank

# Route candidates by verdict
python run_orchestration.py route \
    --adjudication tmp/adjudication

# Build Prosperity-GPT handoff briefs
python run_orchestration.py handoff \
    --adjudication tmp/adjudication \
    --bank tmp/bank \
    --out tmp/orchestration/prosperity_handoff

# Show campaign history
python run_orchestration.py history \
    --history tmp/orchestration/campaign_history.json

# Explain why a candidate is/isn't in the official queue
python run_orchestration.py explain \
    --adjudication tmp/adjudication \
    --candidate mh07
```

## Output Artifacts

```
orchestration/
  campaigns.json              # Full campaign definitions
  campaigns.md                # Human-readable campaign list
  run_plan.json               # Full run plan with budget allocation
  run_plan.md                 # Human-readable run plan
  official_queue.json         # Ranked official-testing queue
  official_queue.md           # Official recommendation memo
  champions.json              # Champion table
  champions.md                # Human-readable champion table
  campaign_history.json       # Cumulative campaign history
  campaign_history.md         # Recent campaign summary
  routing_decisions.json      # Per-candidate routing decisions
  routing_decisions.md        # Human-readable routing summary
  prosperity_handoff/
    CAM_*_brief.md            # Per-campaign GPT handoff briefs
```

## How This Connects to the Research Loop

```
Phase 4 (Synthesis)  ──→  Phase 5 (Adjudication)  ──→  Phase 6 (Orchestrator)
        │                          │                           │
  Generate tasks           Judge results              Decide what to run next
  + briefs                 + verdicts                 + queue for official
                           + learnings                + champion management
                                                      + GPT handoff briefs
                                                              │
                                                              ▼
                                                    Back to Phase 4 with
                                                    disciplined campaign brief
```

## Tests

```bash
python -m pytest tests/test_orchestration.py -v
```

72 tests across 12 test classes covering: campaign creation, routing rules,
champion management, redundancy detection, budget allocation, official queue
ranking, run plan generation, handoff artifacts, campaign history, report
serialization, and full pipeline integration.
