# Strategy Synthesis Engine — Developer Guide

## What this is

A targeted strategy synthesis layer that converts alpha cards (from the Discovery
Engine) into **concrete, constrained strategy generation tasks** with parent
selection, preservation constraints, prompt-ready briefs, and experiment batches.

This is NOT a strategy generator. It produces structured tasks like:
- "Exploit EMERALDS edge in tight spreads, starting from parent 73474 (maker-heavy)"
- "Reduce TOMATOES exposure during wide spreads, preserve all EMERALDS behavior"
- "Match winner profile: increase passive fill rate toward 0.72"

These tasks feed into:
- Prosperity GPT prompt briefs for constrained code generation,
- experiment batches with controls for rigorous comparison,
- structured JSON/MD artifacts for tracking and review.

## Architecture

```
synthesis/
  task.py          # StrategyTask + ExperimentBatch dataclasses
  parents.py       # Parent selection with 5-factor scoring
  converter.py     # Alpha-card-to-task conversion with preservation constraints
  briefs.py        # Prosperity GPT prompt brief renderer
  batch.py         # Experiment batch construction (5 modes)
  report.py        # JSON + markdown + brief file output
run_synthesis.py   # CLI entrypoint
```

## Core Pipeline

```
Alpha Cards  →  Parent Selection  →  Task Conversion  →  Batch Construction  →  Output
(discovery)     (parents.py)         (converter.py)       (batch.py)             (report.py, briefs.py)
```

### 1. StrategyTask Schema (`task.py`)

Each task carries:

| Field | Purpose |
|-------|---------|
| `task_id` | Sequential ID (T001, T002, ...) |
| `title` | Human-readable task title |
| `task_type` | One of: exploit, defend, near_parent_control, mechanism_isolation, calibration_check |
| `source_card_id` | Alpha card that generated this task |
| `product_scope` | Products this task modifies (EMERALDS, TOMATOES, or both) |
| `regime_targeted` | The specific regime conditions being attacked |
| `exploit_objective` | What this task is trying to achieve |
| `expected_mechanism` | How the exploit should work mechanically |
| `main_risk` | What could go wrong |
| `parent_id` | Case ID of the parent strategy to start from |
| `parent_family` | Family label of the parent |
| `parent_rationale` | Why this parent was chosen |
| `preservation` | What must NOT change |
| `allowed_changes` | What CAN change |
| `forbidden_changes` | What must NOT be touched |
| `evaluation_criteria` | How to judge success |
| `success_metric` | Primary metric to check |
| `success_threshold` | Concrete threshold |
| `confidence` | high / medium / low |
| `priority` | critical / high / medium / low |
| `warnings` | Caveats and risks |

### 2. Parent Selection (`parents.py`)

Selects the best base strategy for each task using 5-factor scoring:

1. **Family affinity** (0-3, can be negative): Does the parent's style match the task need?
   - Affinity tables for: maker, taker, emeralds, tomatoes, balanced
   - Example: maker-heavy parent scores +3 for maker tasks, -2 for taker tasks

2. **Quality score** (0-5): PnL, Sharpe, promote status
   - +2.0 if promoted, +up to 2.0 for Sharpe, +up to 1.0 for PnL

3. **Product specialization** (0-2): Per-product PnL alignment with task target

4. **Damage risk penalty** (0 to -3): How much existing edge is at risk
   - Penalizes if the task targets the parent's strongest product

5. **Frontier status** (+1.0 if promoted)

The selector returns the top-scoring parent with rationale and runner-up.

### 3. Card-to-Task Conversion (`converter.py`)

Each alpha card category has a dedicated converter:

| Category | Task Type | Key Focus |
|----------|-----------|-----------|
| `regime_edge` | exploit | Increase activity in profitable regime |
| `role_mismatch` | exploit | Shift maker/taker balance |
| `danger_refinement` | defend | Reduce exposure in unprofitable regime |
| `winner_trait` | exploit | Move metrics toward winner profile |
| `inventory_exploit` | exploit | Adjust inventory management |
| `bot_weakness` | exploit | Add targeted bot reaction logic |

#### Preservation Constraints

Automatically inferred from card + parent:

- **Product scope**: Task targets EMERALDS → preserve all TOMATOES (and vice versa)
- **Risk controls**: Always preserved (position limits, drawdown protection)
- **Maker structure**: Preserved when parent is maker-heavy
- **Execution order**: Preserved for regime_edge tasks
- **Category-specific**: e.g., danger_refinement preserves maker structure

#### Priority Inference

| Confidence | Strength | Priority |
|------------|----------|----------|
| high | > 5 | critical |
| high | any | high |
| medium | > 3 | high |
| medium | any | medium |
| low | any | low |

### 4. Prompt Briefs (`briefs.py`)

Renders each task into a structured text block for Prosperity GPT:

```
## Strategy Generation Brief: T001

**Title:** Exploit EMERALDS edge in spread_bucket=tight
**Type:** exploit
**Priority:** critical | **Confidence:** high

### Objective
Increase activity on EMERALDS during spread_bucket=tight ...

### Evidence
- regime_mean: 5.2000
- baseline_mean: 2.0000
**Observed Fact:** ...
**Interpretation:** ...

### Parent Strategy
- **Parent ID:** 73474
- **Why this parent:** family 'maker-heavy' matches task style; promoted

### Allowed Changes
- EMERALDS quoting parameters
- EMERALDS spread/take widths under targeted regime only

### Forbidden Changes
- Do not change behavior outside the identified regime

### Preservation Constraints
- Do not disable or weaken position limits
- Keep all TOMATOES quoting parameters unchanged

### Requested Output
Generate a modified `strategy.py` that:
1. Starts from the parent strategy (73474) as the base
2. Makes ONLY the allowed changes described above
...
```

Control tasks get a simpler brief format.

### 5. Batch Construction (`batch.py`)

Groups tasks into coherent experiment sets with 5 modes:

| Mode | Description |
|------|-------------|
| `top_priority` | Top N tasks by priority |
| `single_parent` | All tasks for one parent |
| `product_focus` | All tasks targeting one product |
| `balanced` | Diverse mix across categories, products, parents |
| `control_attack` | Exploit/defend tasks paired with controls |

Each batch includes:
- **Controls**: Calibration check + near-parent control for each unique parent
- **Diversity notes**: Category/product/parent distribution
- **Overlap warnings**: Tasks targeting the same regime on the same product

### 6. Report Output (`report.py`)

Three output types:
- **JSON report**: Machine-readable with batch, summary breakdowns, brief texts
- **Markdown report**: Human-readable with task table, details, controls
- **Brief files**: One `.md` file per task in a `briefs/` directory

## CLI Usage

```bash
# Full synthesis from discovery report + bank
python run_synthesis.py discovery_report.json --bank tmp/bank

# Top 5 tasks only
python run_synthesis.py discovery_report.json --bank tmp/bank --max-tasks 5

# Product-focused batch
python run_synthesis.py discovery_report.json --bank tmp/bank --product EMERALDS

# Balanced batch mode
python run_synthesis.py discovery_report.json --bank tmp/bank --mode balanced

# Control-attack mode
python run_synthesis.py discovery_report.json --bank tmp/bank --mode control_attack

# Single parent focus
python run_synthesis.py discovery_report.json --bank tmp/bank \
    --mode single_parent --parent-id 73474

# Emit briefs only
python run_synthesis.py discovery_report.json --bank tmp/bank --briefs-only

# Custom output location
python run_synthesis.py discovery_report.json --bank tmp/bank --out tmp/synthesis_results
```

## Tests

```bash
python -m pytest tests/test_synthesis.py -v
```

Covers: task schema, parent selection, conversion for all 6 categories, preservation
constraints, priority inference, brief rendering, batch construction (all 5 modes),
overlap detection, report generation, file I/O, and full pipeline integration.
