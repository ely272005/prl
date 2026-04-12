"""Tests for the Strategy Synthesis Engine — conversion, parents, briefs, batches."""
import json
import math
import pytest
from pathlib import Path

from synthesis.task import (
    StrategyTask,
    ExperimentBatch,
    TaskCounter,
    TASK_TYPES,
    PRIORITY_LEVELS,
    _sanitize,
)
from synthesis.parents import (
    score_parent,
    select_parent,
    _infer_affinity_key,
    FAMILY_AFFINITY,
)
from synthesis.converter import (
    convert_card_to_task,
    convert_cards_to_tasks,
    generate_control_tasks,
    _infer_preservation,
    _priority_from_card,
    _regime_str,
)
from synthesis.briefs import (
    render_brief,
    render_control_brief,
    render_batch_briefs,
    briefs_to_dict,
)
from synthesis.batch import (
    build_batch,
    build_top_priority_batch,
    build_single_parent_batch,
    build_product_focus_batch,
    build_balanced_batch,
    build_control_attack_batch,
    BATCH_MODES,
    _detect_overlaps,
    _diversity_notes,
)
from synthesis.report import (
    build_json_report,
    build_markdown_report,
    write_json_report,
    write_markdown_report,
    write_brief_files,
)


# ---- Fixtures ----

@pytest.fixture
def sample_parents():
    """Create synthetic parent strategies with varied families."""
    parents = []
    families = ["aggressive", "maker-heavy", "conservative", "taker-heavy", "mixed"]
    for i, family in enumerate(families):
        promoted = i < 3
        pnl_mean = 5000 + i * 2000 if promoted else 500 + i * 100
        sharpe = 15.0 + i * 2 if promoted else 3.0 + i * 0.5
        parents.append({
            "_case_id": f"parent_{i:03d}",
            "_family": family,
            "packet_short": {
                "pnl": {
                    "mean": pnl_mean,
                    "std": 1000.0,
                    "sharpe_like": sharpe,
                },
                "per_product": {
                    "emerald": {"mean": pnl_mean * 0.6},
                    "tomato": {"mean": pnl_mean * 0.4},
                },
                "fill_quality": {
                    "mean_fill_vs_fair_emerald": 3.0 + i,
                    "passive_fill_rate": 0.6 + i * 0.05,
                },
                "efficiency": {"pnl_per_fill": 10.0 + i},
                "promote": {"recommended": promoted, "strength": sharpe},
            },
        })
    return parents


@pytest.fixture
def sample_cards():
    """Create synthetic alpha cards of different categories."""
    return [
        {
            "card_id": "RE001",
            "title": "EMERALDS edge in tight spreads",
            "category": "regime_edge",
            "products": ["EMERALDS"],
            "observed_fact": "Edge is +3.2 in tight spread regime",
            "interpretation": "Market-making is more profitable in tight spreads",
            "suggested_exploit": "Increase quoting during tight spreads",
            "regime_definition": {"spread_bucket": "tight", "product": "EMERALDS", "role": "maker"},
            "evidence": {
                "regime_mean": 5.2,
                "baseline_mean": 2.0,
                "difference": 3.2,
                "fill_count": 150,
            },
            "baseline": {"mean": 2.0},
            "sample_size": {"fills": 150},
            "confidence": "high",
            "strength": 6.5,
            "warnings": [],
            "candidate_strategy_style": "maker-heavy",
            "recommended_experiment": "Increase quoting in tight spread regime",
        },
        {
            "card_id": "RM001",
            "title": "Maker dominates on EMERALDS in low vol",
            "category": "role_mismatch",
            "products": ["EMERALDS"],
            "observed_fact": "Maker edge 4.5 vs taker edge -1.2 in low vol",
            "interpretation": "Taker fills are destructive in low vol",
            "suggested_exploit": "Shift to maker in low vol",
            "regime_definition": {"volatility_regime": "low", "product": "EMERALDS"},
            "evidence": {
                "maker_mean": 4.5,
                "taker_mean": -1.2,
                "ratio": 3.75,
                "maker_fills": 200,
                "taker_fills": 50,
            },
            "baseline": {},
            "sample_size": {"maker_fills": 200, "taker_fills": 50},
            "confidence": "high",
            "strength": 5.0,
            "warnings": [],
            "candidate_strategy_style": "mixed",
            "recommended_experiment": "Reduce taker activity in low vol",
        },
        {
            "card_id": "DR001",
            "title": "TOMATOES bleeding in wide spreads",
            "category": "danger_refinement",
            "products": ["TOMATOES"],
            "observed_fact": "Edge is -4.1 in wide spread regime",
            "interpretation": "Wide spreads attract adverse selection",
            "suggested_exploit": "Reduce exposure during wide spreads",
            "regime_definition": {"spread_bucket": "wide", "product": "TOMATOES"},
            "evidence": {
                "regime_mean": -4.1,
                "baseline_mean": 1.5,
                "difference": -5.6,
            },
            "baseline": {"mean": 1.5},
            "sample_size": {"fills": 80},
            "confidence": "medium",
            "strength": 4.2,
            "warnings": ["Small sample in wide spread bucket"],
            "candidate_strategy_style": "aggressive",
            "recommended_experiment": "Gate activity on spread width",
        },
        {
            "card_id": "WT001",
            "title": "Winners have higher passive fill rate",
            "category": "winner_trait",
            "products": ["EMERALDS", "TOMATOES"],
            "observed_fact": "Promoted candidates have 0.72 passive fill rate vs 0.35",
            "interpretation": "Passive fills are higher quality",
            "suggested_exploit": "Increase passive fill rate",
            "regime_definition": {"metric": "passive_fill_rate", "comparison": "winner_vs_loser"},
            "evidence": {
                "winner_mean": 0.72,
                "loser_mean": 0.35,
                "effect_size": 1.8,
            },
            "baseline": {},
            "sample_size": {"winners": 16, "losers": 14},
            "confidence": "medium",
            "strength": 3.5,
            "warnings": ["Winner traits may be effects not causes"],
            "candidate_strategy_style": "",
            "recommended_experiment": "Tune quoting for more passive fills",
        },
        {
            "card_id": "IE001",
            "title": "EMERALDS position asymmetry",
            "category": "inventory_exploit",
            "products": ["EMERALDS"],
            "observed_fact": "Long positions outperform short by 2.8",
            "interpretation": "Bullish bias in EMERALDS trading",
            "suggested_exploit": "Skew quotes to favor long positions",
            "regime_definition": {"position_bucket": "long", "product": "EMERALDS"},
            "evidence": {
                "long_mean": 4.2,
                "short_mean": 1.4,
                "difference": 2.8,
            },
            "baseline": {},
            "sample_size": {"long_fills": 120, "short_fills": 90},
            "confidence": "medium",
            "strength": 3.0,
            "warnings": [],
            "candidate_strategy_style": "maker-heavy",
            "recommended_experiment": "Adjust inventory skew for EMERALDS",
        },
    ]


@pytest.fixture
def sample_tasks(sample_cards, sample_parents):
    """Convert sample cards into tasks."""
    return convert_cards_to_tasks(sample_cards, sample_parents)


# ===========================================================================
# Task data structure tests
# ===========================================================================

class TestTaskDataStructure:

    def test_task_types_defined(self):
        assert len(TASK_TYPES) == 5
        assert "exploit" in TASK_TYPES
        assert "defend" in TASK_TYPES
        assert "near_parent_control" in TASK_TYPES
        assert "calibration_check" in TASK_TYPES

    def test_priority_levels(self):
        assert PRIORITY_LEVELS == ("critical", "high", "medium", "low")

    def test_task_counter_sequential(self):
        counter = TaskCounter()
        assert counter.next_id() == "T001"
        assert counter.next_id() == "T002"
        assert counter.next_id() == "T003"

    def test_task_counter_custom_prefix(self):
        counter = TaskCounter(prefix="C")
        assert counter.next_id() == "C001"
        assert counter.next_id() == "C002"

    def test_task_to_dict(self):
        task = StrategyTask(
            task_id="T001",
            title="Test task",
            task_type="exploit",
            source_card_id="RE001",
            source_card_title="Test card",
            product_scope=["EMERALDS"],
            regime_targeted={"spread_bucket": "tight"},
            exploit_objective="Test objective",
            expected_mechanism="Test mechanism",
            main_risk="Test risk",
            parent_id="parent_001",
            parent_family="aggressive",
            parent_rationale="Best match",
            preservation=["Keep risk controls"],
            allowed_changes=["EMERALDS widths"],
            forbidden_changes=["TOMATOES params"],
            evaluation_criteria=["PnL improves"],
            success_metric="pnl_mean",
            success_threshold="PnL >= 5000",
            confidence="high",
            priority="critical",
            warnings=["Watch out"],
        )
        d = task.to_dict()
        assert d["task_id"] == "T001"
        assert d["task_type"] == "exploit"
        assert d["product_scope"] == ["EMERALDS"]
        assert d["warnings"] == ["Watch out"]

    def test_task_to_dict_sanitizes_nan(self):
        task = StrategyTask(
            task_id="T001", title="t", task_type="exploit",
            source_card_id="x", source_card_title="x",
            product_scope=["EMERALDS"],
            regime_targeted={"val": float("nan")},
            exploit_objective="x", expected_mechanism="x", main_risk="x",
            parent_id="p", parent_family="f", parent_rationale="r",
            preservation=[], allowed_changes=[], forbidden_changes=[],
            evaluation_criteria=[], success_metric="x", success_threshold="x",
            confidence="low", priority="low",
        )
        d = task.to_dict()
        assert d["regime_targeted"]["val"] is None

    def test_experiment_batch_to_dict(self):
        task = StrategyTask(
            task_id="T001", title="t", task_type="exploit",
            source_card_id="x", source_card_title="x",
            product_scope=["EMERALDS"], regime_targeted={},
            exploit_objective="x", expected_mechanism="x", main_risk="x",
            parent_id="p", parent_family="f", parent_rationale="r",
            preservation=[], allowed_changes=[], forbidden_changes=[],
            evaluation_criteria=[], success_metric="x", success_threshold="x",
            confidence="high", priority="high",
        )
        batch = ExperimentBatch(
            batch_id="B001", title="Test batch", rationale="Testing",
            mode="top_priority", tasks=[task], controls=[],
            diversity_notes="Single task",
        )
        d = batch.to_dict()
        assert d["batch_id"] == "B001"
        assert d["task_count"] == 1
        assert d["control_count"] == 0
        assert len(d["tasks"]) == 1

    def test_sanitize_inf(self):
        assert _sanitize(float("inf")) is None
        assert _sanitize(float("-inf")) is None

    def test_sanitize_nested(self):
        d = {"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": 3.14159265}}
        result = _sanitize(d)
        assert result["a"] is None
        assert result["b"][1] is None
        assert result["c"]["d"] == round(3.14159265, 6)


# ===========================================================================
# Parent selection tests
# ===========================================================================

class TestParentSelection:

    def test_infer_affinity_maker(self):
        card = {"regime_definition": {"role": "maker"}}
        assert _infer_affinity_key(card) == "maker"

    def test_infer_affinity_taker(self):
        card = {"regime_definition": {"role": "taker"}}
        assert _infer_affinity_key(card) == "taker"

    def test_infer_affinity_by_style(self):
        card = {"candidate_strategy_style": "passive maker approach", "regime_definition": {}}
        assert _infer_affinity_key(card) == "maker"

    def test_infer_affinity_by_product(self):
        card = {"products": ["EMERALDS"], "regime_definition": {}}
        assert _infer_affinity_key(card) == "emeralds"

        card = {"products": ["TOMATOES"], "regime_definition": {}}
        assert _infer_affinity_key(card) == "tomatoes"

    def test_infer_affinity_balanced_fallback(self):
        card = {"products": ["EMERALDS", "TOMATOES"], "regime_definition": {}}
        assert _infer_affinity_key(card) == "balanced"

    def test_score_parent_basic(self, sample_parents):
        card = {
            "category": "regime_edge",
            "products": ["EMERALDS"],
            "regime_definition": {"role": "maker"},
        }
        parent = sample_parents[1]  # maker-heavy
        score = score_parent(parent, card)

        assert "total_score" in score
        assert "family_score" in score
        assert "quality_score" in score
        assert score["parent_family"] == "maker-heavy"

    def test_score_parent_family_affinity(self, sample_parents):
        card = {"products": ["EMERALDS"], "regime_definition": {"role": "maker"}}

        # maker-heavy should score high on family for a maker card
        maker_score = score_parent(sample_parents[1], card)  # maker-heavy
        taker_score = score_parent(sample_parents[3], card)  # taker-heavy

        assert maker_score["family_score"] > taker_score["family_score"]

    def test_score_parent_promoted_bonus(self, sample_parents):
        card = {"products": ["EMERALDS", "TOMATOES"], "regime_definition": {}}

        promoted_parent = sample_parents[0]  # promoted
        rejected_parent = sample_parents[4]  # not promoted

        promoted_score = score_parent(promoted_parent, card)
        rejected_score = score_parent(rejected_parent, card)

        assert promoted_score["frontier_bonus"] == 1.0
        assert rejected_score["frontier_bonus"] == 0.0

    def test_select_parent_returns_best(self, sample_parents):
        card = {
            "products": ["EMERALDS"],
            "regime_definition": {"role": "maker"},
            "category": "regime_edge",
        }
        result = select_parent(sample_parents, card)

        assert "parent_id" in result
        assert "total_score" in result
        assert "rationale" in result

    def test_select_parent_empty_list(self):
        card = {"products": ["EMERALDS"]}
        result = select_parent([], card)
        assert result["parent_id"] == "none"
        assert result["total_score"] == 0

    def test_select_parent_runner_up(self, sample_parents):
        card = {"products": ["EMERALDS", "TOMATOES"], "regime_definition": {}}
        result = select_parent(sample_parents, card)
        assert result.get("runner_up") is not None

    def test_family_affinity_coverage(self):
        """Every affinity key should have entries."""
        for key in ("maker", "taker", "emeralds", "tomatoes", "balanced"):
            assert key in FAMILY_AFFINITY
            assert len(FAMILY_AFFINITY[key]) > 0


# ===========================================================================
# Converter tests
# ===========================================================================

class TestConverter:

    def test_convert_regime_edge(self, sample_cards, sample_parents):
        card = sample_cards[0]  # regime_edge
        task = convert_card_to_task(card, sample_parents, TaskCounter())
        assert task.task_type == "exploit"
        assert "EMERALDS" in task.product_scope
        assert task.source_card_id == "RE001"
        assert len(task.preservation) > 0
        assert len(task.allowed_changes) > 0

    def test_convert_role_mismatch(self, sample_cards, sample_parents):
        card = sample_cards[1]  # role_mismatch
        task = convert_card_to_task(card, sample_parents, TaskCounter())
        assert task.task_type == "exploit"
        assert "maker" in task.title.lower() or "taker" in task.title.lower()

    def test_convert_danger_refinement(self, sample_cards, sample_parents):
        card = sample_cards[2]  # danger_refinement
        task = convert_card_to_task(card, sample_parents, TaskCounter())
        assert task.task_type == "defend"
        assert "TOMATOES" in task.product_scope

    def test_convert_winner_trait(self, sample_cards, sample_parents):
        card = sample_cards[3]  # winner_trait
        task = convert_card_to_task(card, sample_parents, TaskCounter())
        assert task.task_type == "exploit"
        assert "passive_fill_rate" in task.title

    def test_convert_inventory_exploit(self, sample_cards, sample_parents):
        card = sample_cards[4]  # inventory_exploit
        task = convert_card_to_task(card, sample_parents, TaskCounter())
        assert task.task_type == "exploit"
        assert "EMERALDS" in task.product_scope

    def test_convert_cards_to_tasks(self, sample_cards, sample_parents):
        tasks = convert_cards_to_tasks(sample_cards, sample_parents)
        assert len(tasks) == 5
        # IDs should be sequential
        ids = [t.task_id for t in tasks]
        assert ids == ["T001", "T002", "T003", "T004", "T005"]

    def test_convert_cards_with_limit(self, sample_cards, sample_parents):
        tasks = convert_cards_to_tasks(sample_cards, sample_parents, max_tasks=3)
        assert len(tasks) == 3

    def test_preservation_emeralds_only(self, sample_parents):
        card = {"products": ["EMERALDS"], "category": "regime_edge", "regime_definition": {}}
        parent = {"parent_family": "aggressive"}
        preservation, allowed, forbidden = _infer_preservation(card, parent)

        # Should preserve TOMATOES
        assert any("TOMATOES" in p for p in preservation)
        # Should allow EMERALDS changes
        assert any("EMERALDS" in a for a in allowed)

    def test_preservation_tomatoes_only(self, sample_parents):
        card = {"products": ["TOMATOES"], "category": "regime_edge", "regime_definition": {}}
        parent = {"parent_family": "aggressive"}
        preservation, allowed, forbidden = _infer_preservation(card, parent)

        assert any("EMERALDS" in p for p in preservation)
        assert any("TOMATOES" in a for a in allowed)

    def test_preservation_both_products(self, sample_parents):
        card = {"products": ["EMERALDS", "TOMATOES"], "category": "regime_edge", "regime_definition": {}}
        parent = {"parent_family": "aggressive"}
        preservation, allowed, forbidden = _infer_preservation(card, parent)

        assert any("both products" in a.lower() for a in allowed)

    def test_preservation_maker_parent(self, sample_parents):
        card = {"products": ["EMERALDS"], "category": "regime_edge", "regime_definition": {}}
        parent = {"parent_family": "maker-heavy"}
        preservation, _, _ = _infer_preservation(card, parent)

        assert any("maker" in p.lower() for p in preservation)

    def test_preservation_risk_controls_always(self, sample_parents):
        card = {"products": ["EMERALDS"], "category": "regime_edge", "regime_definition": {}}
        parent = {"parent_family": "aggressive"}
        preservation, _, _ = _infer_preservation(card, parent)

        assert any("position limit" in p.lower() for p in preservation)

    def test_generate_control_tasks(self, sample_parents):
        parent = sample_parents[0]
        counter = TaskCounter(prefix="C")
        controls = generate_control_tasks(parent, counter)

        assert len(controls) == 2
        types = {c.task_type for c in controls}
        assert "calibration_check" in types
        assert "near_parent_control" in types

    def test_control_calibration_preserves_everything(self, sample_parents):
        parent = sample_parents[0]
        controls = generate_control_tasks(parent, TaskCounter(prefix="C"))
        cal = [c for c in controls if c.task_type == "calibration_check"][0]

        assert "Everything" in cal.preservation[0]
        assert "Nothing" in cal.allowed_changes[0]

    def test_priority_from_card(self):
        assert _priority_from_card({"confidence": "high", "strength": 6}) == "critical"
        assert _priority_from_card({"confidence": "high", "strength": 3}) == "high"
        assert _priority_from_card({"confidence": "medium", "strength": 4}) == "high"
        assert _priority_from_card({"confidence": "medium", "strength": 2}) == "medium"
        assert _priority_from_card({"confidence": "low", "strength": 1}) == "low"

    def test_regime_str_formatting(self):
        assert _regime_str({"spread_bucket": "tight"}) == "spread_bucket=tight"
        assert _regime_str({}) == "general"
        # product/role keys are excluded
        assert _regime_str({"product": "EMERALDS", "spread_bucket": "wide"}) == "spread_bucket=wide"


# ===========================================================================
# Brief tests
# ===========================================================================

class TestBriefs:

    def test_render_brief_structure(self, sample_tasks):
        task = sample_tasks[0]
        card = {
            "evidence": {"regime_mean": 5.2, "baseline_mean": 2.0},
            "observed_fact": "Edge is high",
            "interpretation": "Market making works",
            "suggested_exploit": "Increase quoting",
        }
        brief = render_brief(task, card)

        assert "## Strategy Generation Brief" in brief
        assert task.task_id in brief
        assert task.title in brief
        assert "Objective" in brief
        assert "Evidence" in brief
        assert "Parent Strategy" in brief
        assert "Allowed Changes" in brief
        assert "Forbidden Changes" in brief
        assert "Preservation Constraints" in brief
        assert "Success Criteria" in brief
        assert "Requested Output" in brief

    def test_render_brief_without_card(self, sample_tasks):
        task = sample_tasks[0]
        brief = render_brief(task, card=None)

        assert "## Strategy Generation Brief" in brief
        assert "Objective" in brief

    def test_render_control_brief_calibration(self, sample_parents):
        controls = generate_control_tasks(sample_parents[0], TaskCounter(prefix="C"))
        cal = [c for c in controls if c.task_type == "calibration_check"][0]

        brief = render_control_brief(cal)
        assert "## Control Brief" in brief
        assert "NO changes" in brief
        assert "calibration" in brief.lower()

    def test_render_control_brief_near_parent(self, sample_parents):
        controls = generate_control_tasks(sample_parents[0], TaskCounter(prefix="C"))
        np = [c for c in controls if c.task_type == "near_parent_control"][0]

        brief = render_control_brief(np)
        assert "## Control Brief" in brief
        assert "cosmetic" in brief.lower()

    def test_render_batch_briefs(self, sample_tasks, sample_cards, sample_parents):
        controls = generate_control_tasks(sample_parents[0], TaskCounter(prefix="C"))
        doc = render_batch_briefs(
            sample_tasks, sample_cards, controls,
            batch_title="Test Batch",
        )
        assert "# Test Batch" in doc
        assert f"Total tasks: {len(sample_tasks)}" in doc
        assert "# Control Tasks" in doc

    def test_briefs_to_dict(self, sample_tasks, sample_cards, sample_parents):
        controls = generate_control_tasks(sample_parents[0], TaskCounter(prefix="C"))
        result = briefs_to_dict(sample_tasks, sample_cards, controls)

        assert len(result) == len(sample_tasks) + len(controls)
        # Check exploit tasks
        exploit_briefs = [b for b in result if not b["is_control"]]
        assert len(exploit_briefs) == len(sample_tasks)
        assert all("brief_text" in b for b in exploit_briefs)

        # Check control tasks
        control_briefs = [b for b in result if b["is_control"]]
        assert len(control_briefs) == len(controls)


# ===========================================================================
# Batch construction tests
# ===========================================================================

class TestBatchConstruction:

    def test_batch_modes_defined(self):
        assert len(BATCH_MODES) == 5
        assert "top_priority" in BATCH_MODES
        assert "balanced" in BATCH_MODES
        assert "control_attack" in BATCH_MODES

    def test_top_priority_batch(self, sample_tasks, sample_parents):
        batch = build_top_priority_batch(sample_tasks, sample_parents, max_tasks=3)
        assert batch.mode == "top_priority"
        assert len(batch.tasks) <= 3
        assert isinstance(batch.diversity_notes, str)

    def test_top_priority_respects_priority(self, sample_tasks, sample_parents):
        batch = build_top_priority_batch(sample_tasks, sample_parents, max_tasks=2)
        priorities = [t.priority for t in batch.tasks]
        # The first tasks should have higher or equal priority
        prio_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        ranks = [prio_rank[p] for p in priorities]
        assert ranks == sorted(ranks)

    def test_single_parent_batch(self, sample_tasks, sample_parents):
        # Find which parent was assigned
        parent_id = sample_tasks[0].parent_id
        batch = build_single_parent_batch(
            sample_tasks, sample_parents, parent_id=parent_id,
        )
        assert batch.mode == "single_parent"
        assert all(t.parent_id == parent_id for t in batch.tasks)

    def test_product_focus_batch(self, sample_tasks, sample_parents):
        batch = build_product_focus_batch(
            sample_tasks, sample_parents, product="EMERALDS",
        )
        assert batch.mode == "product_focus"
        assert all("EMERALDS" in t.product_scope for t in batch.tasks)

    def test_balanced_batch_diversity(self, sample_tasks, sample_parents):
        batch = build_balanced_batch(sample_tasks, sample_parents, max_tasks=5)
        assert batch.mode == "balanced"
        assert len(batch.tasks) <= 5

    def test_control_attack_batch(self, sample_tasks, sample_parents):
        batch = build_control_attack_batch(sample_tasks, sample_parents, max_tasks=3)
        assert batch.mode == "control_attack"
        # Should have controls
        assert len(batch.controls) > 0

    def test_batch_includes_controls(self, sample_tasks, sample_parents):
        batch = build_top_priority_batch(sample_tasks, sample_parents)
        # Every unique parent used should have controls
        parent_ids_used = {t.parent_id for t in batch.tasks}
        control_parent_ids = {c.parent_id for c in batch.controls}
        assert parent_ids_used <= control_parent_ids

    def test_detect_overlaps_same_regime(self):
        t1 = _make_task("T001", "exploit", ["EMERALDS"], {"spread_bucket": "tight"})
        t2 = _make_task("T002", "exploit", ["EMERALDS"], {"spread_bucket": "tight"})
        warnings = _detect_overlaps([t1, t2])
        assert len(warnings) > 0
        assert "T001" in warnings[0] and "T002" in warnings[0]

    def test_detect_overlaps_no_overlap(self):
        t1 = _make_task("T001", "exploit", ["EMERALDS"], {"spread_bucket": "tight"})
        t2 = _make_task("T002", "exploit", ["TOMATOES"], {"spread_bucket": "wide"})
        warnings = _detect_overlaps([t1, t2])
        assert len(warnings) == 0

    def test_detect_overlaps_conflicting_directions(self):
        t1 = _make_task("T001", "exploit", ["EMERALDS"], {"spread_bucket": "tight"})
        t2 = _make_task("T002", "defend", ["EMERALDS"], {"spread_bucket": "tight"})
        warnings = _detect_overlaps([t1, t2])
        assert any("Conflicting" in w for w in warnings)

    def test_diversity_notes_content(self, sample_tasks):
        notes = _diversity_notes(sample_tasks)
        assert "Task types:" in notes
        assert "Products targeted:" in notes
        assert "Parent strategies used:" in notes

    def test_diversity_notes_empty(self):
        assert _diversity_notes([]) == "Empty batch."

    def test_build_batch_dispatcher(self, sample_tasks, sample_parents):
        batch = build_batch("top_priority", sample_tasks, sample_parents)
        assert batch.mode == "top_priority"

    def test_build_batch_invalid_mode(self, sample_tasks, sample_parents):
        with pytest.raises(ValueError, match="Unknown batch mode"):
            build_batch("nonexistent", sample_tasks, sample_parents)

    def test_build_batch_single_parent_requires_id(self, sample_tasks, sample_parents):
        with pytest.raises(ValueError, match="parent_id"):
            build_batch("single_parent", sample_tasks, sample_parents)

    def test_build_batch_product_focus_requires_product(self, sample_tasks, sample_parents):
        with pytest.raises(ValueError, match="product"):
            build_batch("product_focus", sample_tasks, sample_parents)

    def test_batch_to_dict_roundtrip(self, sample_tasks, sample_parents):
        batch = build_top_priority_batch(sample_tasks, sample_parents, max_tasks=3)
        d = batch.to_dict()
        assert d["task_count"] == len(batch.tasks)
        assert d["control_count"] == len(batch.controls)
        assert isinstance(d["tasks"], list)
        assert isinstance(d["controls"], list)


# ===========================================================================
# Report tests
# ===========================================================================

class TestReports:

    def test_json_report_structure(self, sample_tasks, sample_parents, sample_cards):
        batch = build_top_priority_batch(sample_tasks, sample_parents, max_tasks=3)
        report = build_json_report(batch, sample_cards)

        assert "generated_at" in report
        assert "batch" in report
        assert "summary" in report
        assert "briefs" in report
        assert report["summary"]["task_count"] == len(batch.tasks)

    def test_json_report_breakdowns(self, sample_tasks, sample_parents):
        batch = build_top_priority_batch(sample_tasks, sample_parents)
        report = build_json_report(batch)

        assert "by_type" in report["summary"]
        assert "by_priority" in report["summary"]
        assert "by_product" in report["summary"]

    def test_markdown_report_structure(self, sample_tasks, sample_parents, sample_cards):
        batch = build_top_priority_batch(sample_tasks, sample_parents, max_tasks=3)
        md = build_markdown_report(batch, sample_cards)

        assert "# Synthesis Report" in md
        assert "## Summary" in md
        assert "## Tasks" in md
        assert "## Task Details" in md
        assert batch.batch_id in md

    def test_markdown_report_overlap_warnings(self, sample_parents):
        t1 = _make_task("T001", "exploit", ["EMERALDS"], {"spread_bucket": "tight"})
        t2 = _make_task("T002", "exploit", ["EMERALDS"], {"spread_bucket": "tight"})
        batch = build_top_priority_batch([t1, t2], sample_parents)
        md = build_markdown_report(batch)

        if batch.overlap_warnings:
            assert "Overlap Warnings" in md

    def test_write_json_report(self, sample_tasks, sample_parents, tmp_path):
        batch = build_top_priority_batch(sample_tasks, sample_parents, max_tasks=3)
        out = tmp_path / "report.json"
        write_json_report(batch, out)

        assert out.exists()
        data = json.loads(out.read_text())
        assert "batch" in data

    def test_write_markdown_report(self, sample_tasks, sample_parents, tmp_path):
        batch = build_top_priority_batch(sample_tasks, sample_parents, max_tasks=3)
        out = tmp_path / "report.md"
        write_markdown_report(batch, out)

        assert out.exists()
        content = out.read_text()
        assert "# Synthesis Report" in content

    def test_write_brief_files(self, sample_tasks, sample_parents, sample_cards, tmp_path):
        batch = build_top_priority_batch(sample_tasks, sample_parents, max_tasks=3)
        brief_dir = tmp_path / "briefs"
        written = write_brief_files(batch, brief_dir, sample_cards)

        assert len(written) > 0
        for path in written:
            assert path.exists()
            assert path.suffix == ".md"


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration:

    def test_full_pipeline(self, sample_cards, sample_parents):
        """End-to-end: cards → tasks → batch → reports."""
        # 1. Convert cards to tasks
        tasks = convert_cards_to_tasks(sample_cards, sample_parents)
        assert len(tasks) == len(sample_cards)

        # 2. Build batch
        batch = build_batch("top_priority", tasks, sample_parents, max_tasks=5)
        assert len(batch.tasks) <= 5
        assert len(batch.controls) > 0

        # 3. Generate reports
        json_report = build_json_report(batch, sample_cards)
        assert json_report["summary"]["task_count"] == len(batch.tasks)

        md_report = build_markdown_report(batch, sample_cards)
        assert "# Synthesis Report" in md_report

    def test_balanced_pipeline(self, sample_cards, sample_parents):
        tasks = convert_cards_to_tasks(sample_cards, sample_parents)
        batch = build_batch("balanced", tasks, sample_parents, max_tasks=4)
        assert batch.mode == "balanced"
        assert len(batch.tasks) <= 4

    def test_control_attack_pipeline(self, sample_cards, sample_parents):
        tasks = convert_cards_to_tasks(sample_cards, sample_parents)
        batch = build_batch("control_attack", tasks, sample_parents, max_tasks=3)

        # Controls should exist for each parent used
        parent_ids = {t.parent_id for t in batch.tasks}
        control_parent_ids = {c.parent_id for c in batch.controls}
        assert parent_ids <= control_parent_ids

    def test_brief_output_pipeline(self, sample_cards, sample_parents, tmp_path):
        tasks = convert_cards_to_tasks(sample_cards, sample_parents)
        batch = build_batch("top_priority", tasks, sample_parents, max_tasks=3)

        written = write_brief_files(batch, tmp_path / "briefs", sample_cards)
        assert len(written) == len(batch.tasks) + len(batch.controls)

        # Each brief should be non-empty
        for path in written:
            content = path.read_text()
            assert len(content) > 100

    def test_all_card_categories_convert(self, sample_parents):
        """Every supported card category should produce a valid task."""
        categories = [
            "regime_edge", "role_mismatch", "danger_refinement",
            "winner_trait", "inventory_exploit", "bot_weakness",
        ]
        for cat in categories:
            card = {
                "card_id": f"TEST_{cat}",
                "title": f"Test {cat}",
                "category": cat,
                "products": ["EMERALDS"],
                "regime_definition": {"spread_bucket": "tight"},
                "evidence": {
                    "regime_mean": 3.0,
                    "baseline_mean": 1.0,
                    "difference": 2.0,
                    "maker_mean": 3.0,
                    "taker_mean": 1.0,
                    "winner_mean": 0.7,
                    "effect_size": 1.5,
                },
                "confidence": "medium",
                "strength": 3.0,
            }
            task = convert_card_to_task(card, sample_parents, TaskCounter())
            assert task.task_id == "T001"
            assert task.source_card_id == f"TEST_{cat}"
            assert len(task.preservation) > 0


# ===========================================================================
# Helper
# ===========================================================================

def _make_task(
    task_id: str,
    task_type: str,
    products: list[str],
    regime: dict,
) -> StrategyTask:
    return StrategyTask(
        task_id=task_id, title=f"Test {task_id}", task_type=task_type,
        source_card_id="test", source_card_title="test",
        product_scope=products, regime_targeted=regime,
        exploit_objective="test", expected_mechanism="test", main_risk="test",
        parent_id="parent_000", parent_family="aggressive",
        parent_rationale="test",
        preservation=[], allowed_changes=[], forbidden_changes=[],
        evaluation_criteria=[], success_metric="pnl_mean",
        success_threshold="PnL >= 0",
        confidence="medium", priority="medium",
    )
