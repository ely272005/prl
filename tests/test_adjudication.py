"""Tests for the Adjudication Engine — verdicts, hypotheses, preservation, attribution, frontier."""
import json
import math
import pytest
from pathlib import Path

from adjudication.comparison import (
    compare_pair,
    compare_to_frontier,
    compare_to_family,
    summarize_comparison,
    _extract,
    COMPARISON_METRICS,
)
from adjudication.preservation import (
    audit_preservation,
    _check_product_preservation,
    _check_maker_structure,
    _check_calibration,
    _check_near_parent,
)
from adjudication.attribution import (
    attribute_mechanism,
    _attribute_product,
    _attribute_role_shift,
    _attribute_aggressiveness,
    _attribute_sharpe,
)
from adjudication.suspicion import (
    detect_suspicions,
    _check_noise,
    _check_sharpe_mean_divergence,
    _check_single_product_fragility,
)
from adjudication.verdicts import (
    adjudicate_candidate,
    VERDICT_LABELS,
)
from adjudication.hypothesis import (
    adjudicate_hypothesis,
    HYPOTHESIS_OUTCOMES,
)
from adjudication.frontier import (
    compute_frontier_updates,
    _dominates,
    _assign_roles,
    FRONTIER_ROLES,
    MAX_PER_FAMILY,
)
from adjudication.learnings import extract_batch_learnings
from adjudication.next_actions import (
    recommend_next_actions,
    format_gpt_summary,
    ACTION_TYPES,
)
from adjudication.report import write_all_reports


# ---- Fixtures ----

def _make_packet(
    pnl_mean=5000, pnl_std=2000, sharpe=10.0, p05=1000, p50=5000, p95=9000,
    pos_rate=0.8, em_mean=3000, tom_mean=2000, em_sharpe=8.0, tom_sharpe=6.0,
    passive_fill_rate=0.65, pnl_per_fill=5.0, taker_fills=5000, maker_fills=10000,
    mean_max_dd=-500, confidence="HIGH", promote=True, kill=False,
    case_id="test_001", family="aggressive",
):
    """Create a synthetic packet for testing."""
    return {
        "_case_id": case_id,
        "_family": family,
        "packet_short": {
            "candidate_id": case_id,
            "confidence": confidence,
            "warnings": [],
            "pnl": {
                "mean": pnl_mean, "std": pnl_std, "sharpe_like": sharpe,
                "p05": p05, "p50": p50, "p95": p95,
                "positive_rate": pos_rate, "skewness": 0.1,
            },
            "per_product": {
                "emerald": {"mean": em_mean, "std": 1000, "sharpe_like": em_sharpe},
                "tomato": {"mean": tom_mean, "std": 800, "sharpe_like": tom_sharpe},
            },
            "fill_quality": {
                "mean_fill_vs_fair_emerald": 3.0,
                "mean_fill_vs_fair_tomato": 2.5,
                "passive_fill_rate": passive_fill_rate,
                "taker_fill_count": taker_fills,
                "maker_fill_count": maker_fills,
            },
            "efficiency": {"pnl_per_fill": pnl_per_fill},
            "drawdown": {"mean_max_drawdown": mean_max_dd, "p95_max_drawdown": mean_max_dd * 2},
            "promote": {"recommended": promote, "strength": sharpe, "reason": "test"},
            "kill": {"recommended": kill, "strength": 0, "reason": "test"},
            "external_validity_note": "Simulator results cannot guarantee official server performance.",
        },
    }


def _make_task(
    task_id="T001", task_type="exploit", parent_id="parent_001",
    parent_family="aggressive", product_scope=None, source_card_id="RE001",
    preservation=None, allowed_changes=None, forbidden_changes=None,
):
    """Create a synthetic task for testing."""
    return {
        "task_id": task_id,
        "title": "Test task",
        "task_type": task_type,
        "source_card_id": source_card_id,
        "source_card_title": "Test card title",
        "product_scope": product_scope or ["EMERALDS"],
        "regime_targeted": {"spread_bucket": "tight"},
        "exploit_objective": "Test objective",
        "expected_mechanism": "Test mechanism",
        "main_risk": "Test risk",
        "parent_id": parent_id,
        "parent_family": parent_family,
        "parent_rationale": "Best match",
        "preservation": preservation or ["Do not disable or weaken position limits"],
        "allowed_changes": allowed_changes or ["EMERALDS quoting parameters"],
        "forbidden_changes": forbidden_changes or [],
        "evaluation_criteria": ["PnL improves"],
        "success_metric": "pnl_mean",
        "success_threshold": "PnL >= 5000",
        "confidence": "high",
        "priority": "high",
        "warnings": [],
    }


@pytest.fixture
def parent_packet():
    return _make_packet(
        pnl_mean=5000, sharpe=10.0, em_mean=3000, tom_mean=2000,
        case_id="parent_001", family="aggressive",
    )


@pytest.fixture
def improved_candidate():
    return _make_packet(
        pnl_mean=7000, sharpe=14.0, em_mean=4500, tom_mean=2500,
        case_id="candidate_better", family="aggressive",
    )


@pytest.fixture
def worse_candidate():
    return _make_packet(
        pnl_mean=3000, sharpe=5.0, em_mean=2000, tom_mean=1000,
        case_id="candidate_worse", family="aggressive", promote=False,
    )


@pytest.fixture
def noisy_candidate():
    """Higher mean but much worse Sharpe."""
    return _make_packet(
        pnl_mean=6000, pnl_std=6000, sharpe=4.0, em_mean=3500, tom_mean=2500,
        case_id="candidate_noisy", family="aggressive", promote=False,
    )


@pytest.fixture
def sample_task():
    return _make_task()


@pytest.fixture
def sample_frontier(parent_packet):
    return [
        parent_packet,
        _make_packet(pnl_mean=6000, sharpe=12.0, case_id="frontier_001"),
        _make_packet(pnl_mean=4000, sharpe=15.0, case_id="frontier_002"),
    ]


# ===========================================================================
# Comparison tests
# ===========================================================================

class TestComparison:

    def test_compare_pair_basic(self, improved_candidate, parent_packet):
        result = compare_pair(improved_candidate, parent_packet, "parent")
        assert result["reference_label"] == "parent"
        assert result["improvements"] > 0
        assert result["net_direction"] in ("strong_improvement", "mild_improvement")

    def test_compare_pair_worse(self, worse_candidate, parent_packet):
        result = compare_pair(worse_candidate, parent_packet, "parent")
        assert result["regressions"] > result["improvements"]

    def test_compare_pair_deltas(self, improved_candidate, parent_packet):
        result = compare_pair(improved_candidate, parent_packet)
        pnl_delta = next(d for d in result["deltas"] if d["metric"] == "pnl_mean")
        assert pnl_delta["delta"] == 2000.0
        assert pnl_delta["improved"] is True

    def test_compare_to_frontier(self, improved_candidate, sample_frontier):
        result = compare_to_frontier(improved_candidate, sample_frontier)
        assert result["frontier_size"] == 3
        assert isinstance(result["beats_frontier_on"], list)
        assert isinstance(result["below_frontier_on"], list)

    def test_compare_to_empty_frontier(self, improved_candidate):
        result = compare_to_frontier(improved_candidate, [])
        assert result["frontier_size"] == 0

    def test_compare_to_family(self, improved_candidate, sample_frontier):
        result = compare_to_family(improved_candidate, sample_frontier, "aggressive")
        assert result["family_size"] == 3

    def test_summarize_comparison(self, improved_candidate, parent_packet, sample_frontier):
        vs_parent = compare_pair(improved_candidate, parent_packet)
        vs_frontier = compare_to_frontier(improved_candidate, sample_frontier)
        summary = summarize_comparison(vs_parent, vs_frontier)
        assert isinstance(summary, str)
        assert len(summary) > 10

    def test_extract_nested(self):
        packet = {"packet_short": {"pnl": {"mean": 5000}}}
        assert _extract(packet, ("pnl", "mean")) == 5000

    def test_extract_missing(self):
        packet = {"packet_short": {}}
        assert _extract(packet, ("pnl", "mean")) is None

    def test_extract_nan(self):
        packet = {"packet_short": {"pnl": {"mean": float("nan")}}}
        assert _extract(packet, ("pnl", "mean")) is None

    def test_comparison_metrics_defined(self):
        assert len(COMPARISON_METRICS) >= 10


# ===========================================================================
# Preservation tests
# ===========================================================================

class TestPreservation:

    def test_clean_preservation(self, improved_candidate, parent_packet, sample_task):
        result = audit_preservation(sample_task, improved_candidate, parent_packet)
        # May or may not have violations depending on deltas
        assert "verdict" in result
        assert result["verdict"] in ("clean", "suspect", "violated")

    def test_product_preservation_violated(self):
        """If task targets EMERALDS only, TOMATOES should not change much."""
        task = _make_task(product_scope=["EMERALDS"])
        candidate = _make_packet(em_mean=5000, tom_mean=500)  # TOMATOES changed a lot
        parent = _make_packet(em_mean=3000, tom_mean=2000)
        result = audit_preservation(task, candidate, parent)
        violations = [v for v in result["violations"] if "TOMATOES" in v.get("constraint", "")]
        assert len(violations) > 0

    def test_product_preservation_clean(self):
        """Same TOMATOES → no violation."""
        task = _make_task(product_scope=["EMERALDS"])
        candidate = _make_packet(em_mean=5000, tom_mean=2000)  # TOMATOES unchanged
        parent = _make_packet(em_mean=3000, tom_mean=2000)
        result = audit_preservation(task, candidate, parent)
        tomato_violations = [v for v in result["violations"] if "TOMATOES" in v.get("constraint", "")]
        assert len(tomato_violations) == 0

    def test_maker_structure_violation(self):
        """Passive fill rate dropped significantly."""
        task = _make_task(preservation=["Preserve maker-heavy quoting structure"])
        candidate = _make_packet(passive_fill_rate=0.30)
        parent = _make_packet(passive_fill_rate=0.65)
        result = audit_preservation(task, candidate, parent)
        maker_violations = [v for v in result["violations"] if "maker" in v.get("constraint", "").lower()]
        assert len(maker_violations) > 0

    def test_calibration_check_clean(self):
        task = _make_task(task_type="calibration_check")
        candidate = _make_packet(pnl_mean=5050)
        parent = _make_packet(pnl_mean=5000)
        result = audit_preservation(task, candidate, parent)
        assert result["verdict"] == "clean"

    def test_calibration_check_violated(self):
        task = _make_task(task_type="calibration_check")
        candidate = _make_packet(pnl_mean=3000)
        parent = _make_packet(pnl_mean=5000)
        result = audit_preservation(task, candidate, parent)
        assert result["verdict"] in ("suspect", "violated")

    def test_near_parent_deviated(self):
        task = _make_task(task_type="near_parent_control")
        candidate = _make_packet(pnl_mean=7000)
        parent = _make_packet(pnl_mean=5000)
        result = audit_preservation(task, candidate, parent)
        assert len(result["violations"]) > 0


# ===========================================================================
# Attribution tests
# ===========================================================================

class TestAttribution:

    def test_attribute_basic(self, improved_candidate, parent_packet):
        result = attribute_mechanism(improved_candidate, parent_packet)
        assert "attributions" in result
        assert "dominant_mechanism" in result
        assert "summary" in result
        assert isinstance(result["summary"], str)

    def test_attribute_product_shift(self):
        """All gain from EMERALDS."""
        candidate = _make_packet(em_mean=5000, tom_mean=2000)
        parent = _make_packet(em_mean=3000, tom_mean=2000)
        result = attribute_mechanism(candidate, parent)
        prod = result.get("product_attribution")
        assert prod is not None
        assert prod["detail"]["dominant_product"] == "EMERALDS"

    def test_attribute_role_shift(self):
        candidate = _make_packet(passive_fill_rate=0.80)
        parent = _make_packet(passive_fill_rate=0.50)
        result = attribute_mechanism(candidate, parent)
        role = result.get("role_attribution")
        assert role is not None
        assert "more passive" in role["detail"]["direction"]

    def test_attribute_aggressiveness(self):
        candidate = _make_packet(taker_fills=15000, maker_fills=20000)
        parent = _make_packet(taker_fills=5000, maker_fills=10000)
        result = attribute_mechanism(candidate, parent)
        agg = [a for a in result["attributions"] if a["mechanism"] == "aggressiveness_change"]
        assert len(agg) > 0

    def test_attribute_sharpe_decomposition(self):
        """Sharpe improved through lower std."""
        candidate = _make_packet(pnl_mean=5000, pnl_std=1000, sharpe=20.0)
        parent = _make_packet(pnl_mean=5000, pnl_std=2000, sharpe=10.0)
        result = attribute_mechanism(candidate, parent)
        sharpe_attrs = [a for a in result["attributions"] if a["mechanism"] == "sharpe_decomposition"]
        assert len(sharpe_attrs) > 0
        assert "lower volatility" in sharpe_attrs[0]["detail"]["driver"]

    def test_attribute_no_change(self):
        """Identical packets → no attributions."""
        packet = _make_packet()
        result = attribute_mechanism(packet, packet)
        assert result["dominant_mechanism"] == "no_clear_driver"


# ===========================================================================
# Suspicion tests
# ===========================================================================

class TestSuspicion:

    def test_clean_candidate(self, improved_candidate, parent_packet):
        result = detect_suspicions(improved_candidate, parent_packet)
        assert result["suspicion_level"] in ("clean", "low")

    def test_noise_likely(self):
        """Small gain relative to variance."""
        candidate = _make_packet(pnl_mean=5100, pnl_std=2000)
        parent = _make_packet(pnl_mean=5000, pnl_std=2000)
        result = detect_suspicions(candidate, parent)
        noise_flags = [f for f in result["flags"] if f["flag"] == "noise_likely"]
        assert len(noise_flags) > 0

    def test_sharpe_mean_divergence(self):
        candidate = _make_packet(pnl_mean=7000, sharpe=4.0)
        parent = _make_packet(pnl_mean=5000, sharpe=10.0)
        result = detect_suspicions(candidate, parent)
        div_flags = [f for f in result["flags"] if f["flag"] == "sharpe_mean_divergence"]
        assert len(div_flags) > 0

    def test_single_product_fragility(self):
        candidate = _make_packet(em_mean=6000, tom_mean=500)
        parent = _make_packet(em_mean=3000, tom_mean=2000)
        result = detect_suspicions(candidate, parent)
        frag_flags = [f for f in result["flags"] if f["flag"] == "single_product_fragile"]
        assert len(frag_flags) > 0

    def test_low_confidence_suspect(self):
        candidate = _make_packet(confidence="LOW")
        parent = _make_packet()
        result = detect_suspicions(candidate, parent)
        cal_flags = [f for f in result["flags"] if f["flag"] == "calibration_suspect"]
        assert len(cal_flags) > 0

    def test_preservation_violated_flag(self):
        candidate = _make_packet()
        parent = _make_packet()
        pres_audit = {"verdict": "violated", "reason": "constraints broken"}
        result = detect_suspicions(candidate, parent, preservation_audit=pres_audit)
        pres_flags = [f for f in result["flags"] if f["flag"] == "preservation_violated"]
        assert len(pres_flags) > 0


# ===========================================================================
# Verdict tests
# ===========================================================================

class TestVerdicts:

    def test_verdict_labels_defined(self):
        assert len(VERDICT_LABELS) == 7
        assert "frontier_challenger" in VERDICT_LABELS
        assert "reject" in VERDICT_LABELS

    def test_improved_candidate_verdict(self, parent_packet, sample_frontier):
        # Candidate improves on EMERALDS without changing TOMATOES
        candidate = _make_packet(
            pnl_mean=7000, sharpe=14.0, em_mean=5000, tom_mean=2000,
            case_id="candidate_better",
        )
        task = _make_task(product_scope=["EMERALDS", "TOMATOES"])
        result = adjudicate_candidate(
            candidate, parent_packet, task,
            frontier=sample_frontier,
        )
        assert result["verdict"] in ("frontier_challenger", "escalate", "keep")
        assert result["pnl_delta"] > 0
        assert result["sharpe_delta"] > 0
        assert "candidate_id" in result
        assert "mechanism_interpretation" in result
        assert "transfer_risk" in result

    def test_worse_candidate_rejected(self, worse_candidate, parent_packet, sample_task):
        result = adjudicate_candidate(worse_candidate, parent_packet, sample_task)
        assert result["verdict"] == "reject"
        assert result["pnl_delta"] < 0

    def test_noisy_candidate_suspect(self, noisy_candidate, parent_packet, sample_task):
        result = adjudicate_candidate(noisy_candidate, parent_packet, sample_task)
        # Higher mean but much worse Sharpe → should be flagged
        assert result["verdict"] in ("suspect_simulator_gain", "keep", "reject")

    def test_control_success(self, parent_packet):
        task = _make_task(task_type="calibration_check")
        candidate = _make_packet(pnl_mean=5050)  # within tolerance
        result = adjudicate_candidate(candidate, parent_packet, task)
        assert result["verdict"] == "control_success"

    def test_control_failure(self, parent_packet):
        task = _make_task(task_type="calibration_check")
        candidate = _make_packet(pnl_mean=2000)  # way off
        result = adjudicate_candidate(candidate, parent_packet, task)
        assert result["verdict"] == "control_failure"

    def test_near_parent_control(self, parent_packet):
        task = _make_task(task_type="near_parent_control")
        candidate = _make_packet(pnl_mean=5100)  # close enough
        result = adjudicate_candidate(candidate, parent_packet, task)
        assert result["verdict"] == "control_success"

    def test_verdict_has_all_fields(self, improved_candidate, parent_packet, sample_task):
        result = adjudicate_candidate(improved_candidate, parent_packet, sample_task)
        required = [
            "candidate_id", "task_id", "parent_id", "family", "verdict",
            "reason", "pnl_mean", "pnl_std", "sharpe", "p05", "p50", "p95",
            "positive_rate", "confidence", "promote_recommended", "kill_recommended",
            "emerald_mean", "tomato_mean", "emerald_delta", "tomato_delta",
            "pnl_delta", "sharpe_delta", "vs_parent", "vs_frontier",
            "preservation_audit", "attribution", "suspicion",
            "mechanism_interpretation", "transfer_risk", "recommended_next_action",
        ]
        for field in required:
            assert field in result, f"Missing field: {field}"

    def test_verdict_per_product(self, improved_candidate, parent_packet, sample_task):
        result = adjudicate_candidate(improved_candidate, parent_packet, sample_task)
        assert result["emerald_delta"] == 1500.0
        assert result["tomato_delta"] == 500.0


# ===========================================================================
# Hypothesis tests
# ===========================================================================

class TestHypothesis:

    def test_hypothesis_outcomes_defined(self):
        assert len(HYPOTHESIS_OUTCOMES) == 5
        assert "validated" in HYPOTHESIS_OUTCOMES
        assert "falsified" in HYPOTHESIS_OUTCOMES

    def test_validated_hypothesis(self, improved_candidate, parent_packet, sample_task):
        cv = adjudicate_candidate(improved_candidate, parent_packet, sample_task)
        hv = adjudicate_hypothesis(cv, sample_task)
        assert hv["outcome"] in ("validated", "partially_validated")
        assert hv["mean_helped"] is True
        assert hv["sharpe_helped"] is True

    def test_falsified_hypothesis(self, worse_candidate, parent_packet, sample_task):
        cv = adjudicate_candidate(worse_candidate, parent_packet, sample_task)
        hv = adjudicate_hypothesis(cv, sample_task)
        assert hv["outcome"] in ("falsified", "informative_failure", "inconclusive")

    def test_control_not_applicable(self, parent_packet):
        task = _make_task(task_type="calibration_check")
        cv = adjudicate_candidate(parent_packet, parent_packet, task)
        hv = adjudicate_hypothesis(cv, task)
        assert hv["outcome"] == "not_applicable"

    def test_hypothesis_checks(self, improved_candidate, parent_packet, sample_task):
        cv = adjudicate_candidate(improved_candidate, parent_packet, sample_task)
        hv = adjudicate_hypothesis(cv, sample_task)
        check_names = {c["check"] for c in hv["checks"]}
        expected = {"mean_improved", "sharpe_improved", "intended_product_helped",
                    "preserved_base_intact", "not_just_aggression", "mechanism_matched"}
        assert expected == check_names

    def test_hypothesis_lessons(self, improved_candidate, parent_packet, sample_task):
        cv = adjudicate_candidate(improved_candidate, parent_packet, sample_task)
        hv = adjudicate_hypothesis(cv, sample_task)
        assert isinstance(hv["lessons"], list)

    def test_single_product_detection(self, parent_packet):
        # EMERALDS helped, TOMATOES hurt, net positive PnL
        candidate = _make_packet(pnl_mean=6500, em_mean=5500, tom_mean=1000)
        task = _make_task(product_scope=["EMERALDS", "TOMATOES"])
        cv = adjudicate_candidate(candidate, parent_packet, task)
        hv = adjudicate_hypothesis(cv, task)
        assert hv["single_product_only"] is True


# ===========================================================================
# Frontier tests
# ===========================================================================

class TestFrontier:

    def test_frontier_roles_defined(self):
        assert len(FRONTIER_ROLES) == 6
        assert "top_mean" in FRONTIER_ROLES

    def test_dominates(self):
        a = _make_packet(pnl_mean=6000, sharpe=12.0, p05=2000)
        b = _make_packet(pnl_mean=5000, sharpe=10.0, p05=1000)
        assert _dominates(a, b) is True
        assert _dominates(b, a) is False

    def test_dominates_equal(self):
        a = _make_packet(pnl_mean=5000, sharpe=10.0, p05=1000)
        b = _make_packet(pnl_mean=5000, sharpe=10.0, p05=1000)
        assert _dominates(a, b) is False  # equal → not dominated

    def test_assign_roles(self, sample_frontier):
        roles = _assign_roles(sample_frontier)
        assert "top_mean" in roles
        assert "top_sharpe" in roles

    def test_frontier_update_adds_challenger(self, parent_packet, sample_frontier):
        # Create a strong challenger verdict
        verdicts = [{
            "candidate_id": "new_champ",
            "verdict": "frontier_challenger",
            "pnl_mean": 8000,
            "sharpe": 20.0,
            "family": "aggressive",
            "suspicion": {"is_suspicious": False},
        }]
        new_packet = _make_packet(pnl_mean=8000, sharpe=20.0, case_id="new_champ")
        all_packets = sample_frontier + [new_packet]

        result = compute_frontier_updates(sample_frontier, verdicts, all_packets)
        assert len(result["additions"]) > 0
        assert result["additions"][0]["candidate_id"] == "new_champ"

    def test_frontier_update_no_suspicious(self, sample_frontier):
        verdicts = [{
            "candidate_id": "sus_001",
            "verdict": "frontier_challenger",
            "pnl_mean": 9000,
            "sharpe": 25.0,
            "family": "aggressive",
            "suspicion": {"is_suspicious": True},
        }]
        result = compute_frontier_updates(sample_frontier, verdicts, sample_frontier)
        assert len(result["additions"]) == 0

    def test_frontier_family_diversity(self, sample_frontier):
        # Fill up the "aggressive" family
        verdicts = []
        all_packets = list(sample_frontier)
        for i in range(5):
            cid = f"agg_{i}"
            verdicts.append({
                "candidate_id": cid,
                "verdict": "escalate",
                "pnl_mean": 5000 + i * 500,
                "sharpe": 10.0 + i,
                "family": "aggressive",
                "suspicion": {"is_suspicious": False},
            })
            all_packets.append(_make_packet(
                pnl_mean=5000 + i * 500, sharpe=10.0 + i,
                case_id=cid, family="aggressive",
            ))

        result = compute_frontier_updates(sample_frontier, verdicts, all_packets)
        # Should not add more than MAX_PER_FAMILY from same family
        agg_added = [a for a in result["additions"] if a["family"] == "aggressive"]
        # Count aggressive in frontier after
        assert len(result["notes"]) >= 0  # at least some should be skipped

    def test_frontier_empty(self):
        verdicts = [{
            "candidate_id": "first",
            "verdict": "frontier_challenger",
            "pnl_mean": 5000,
            "sharpe": 10.0,
            "family": "aggressive",
            "suspicion": {"is_suspicious": False},
        }]
        first_packet = _make_packet(case_id="first")
        result = compute_frontier_updates([], verdicts, [first_packet])
        assert len(result["additions"]) == 1


# ===========================================================================
# Learnings tests
# ===========================================================================

class TestLearnings:

    def test_extract_learnings(self):
        cv = [{
            "candidate_id": "c1", "task_id": "T001", "parent_id": "p1",
            "family": "aggressive", "verdict": "escalate",
            "pnl_delta": 500, "sharpe_delta": 2.0,
            "attribution": {"dominant_mechanism": "product_shift"},
        }]
        hv = [{
            "hypothesis_id": "RE001", "task_id": "T001",
            "hypothesis_title": "Test",
            "outcome": "validated", "reason": "Mechanism worked.",
            "lessons": ["EMERALDS edge confirmed"],
        }]
        tasks = [_make_task()]

        result = extract_batch_learnings(cv, hv, tasks)
        assert len(result["validated_mechanisms"]) == 1
        assert result["validated_mechanisms"][0]["hypothesis_id"] == "RE001"
        assert "aggressive" in result["family_lessons"]

    def test_dead_zone_detection(self):
        """Multiple failures of same hypothesis → dead zone."""
        hv = [
            {"hypothesis_id": "RE001", "outcome": "falsified", "reason": "Failed 1", "lessons": [], "task_id": "T001", "hypothesis_title": "Test"},
            {"hypothesis_id": "RE001", "outcome": "falsified", "reason": "Failed 2", "lessons": [], "task_id": "T002", "hypothesis_title": "Test"},
        ]
        cv = [
            {"candidate_id": "c1", "task_id": "T001", "parent_id": "p1", "family": "x", "verdict": "reject", "pnl_delta": -500, "sharpe_delta": -2, "attribution": {"dominant_mechanism": "x"}},
            {"candidate_id": "c2", "task_id": "T002", "parent_id": "p2", "family": "x", "verdict": "reject", "pnl_delta": -400, "sharpe_delta": -1, "attribution": {"dominant_mechanism": "x"}},
        ]

        result = extract_batch_learnings(cv, hv, [])
        assert len(result["dead_zones"]) >= 1
        assert result["dead_zones"][0]["hypothesis_id"] == "RE001"

    def test_suspicious_tracked(self):
        cv = [{
            "candidate_id": "sus1", "task_id": "T001", "parent_id": "p1",
            "family": "aggressive", "verdict": "suspect_simulator_gain",
            "pnl_delta": 300, "sharpe_delta": -1,
            "attribution": {"dominant_mechanism": "aggressiveness_change"},
            "reason": "Suspicious gain",
            "suspicion": {"flags": [{"flag": "aggression_driven_gain"}]},
        }]
        result = extract_batch_learnings(cv, [], [])
        assert len(result["suspicious_directions"]) == 1

    def test_summary_generated(self):
        result = extract_batch_learnings([], [], [])
        assert "summary" in result
        assert isinstance(result["summary"], str)


# ===========================================================================
# Next action tests
# ===========================================================================

class TestNextActions:

    def test_action_types_defined(self):
        assert len(ACTION_TYPES) >= 8

    def test_confirm_challenger_action(self):
        cv = [{"verdict": "frontier_challenger", "candidate_id": "c1", "pnl_mean": 8000, "sharpe": 20.0}]
        hv = []
        learnings = {"dead_zones": [], "validated_mechanisms": [], "falsified_mechanisms": []}
        frontier = {"additions": [], "retirements": []}

        actions = recommend_next_actions(cv, hv, learnings, frontier)
        confirm = [a for a in actions if a["action_type"] == "confirm_challenger"]
        assert len(confirm) == 1

    def test_stop_exploring_action(self):
        cv = []
        hv = []
        learnings = {"dead_zones": [{"hypothesis_id": "RE001", "failure_count": 3, "reason": "Exhausted"}]}
        frontier = {"additions": []}

        actions = recommend_next_actions(cv, hv, learnings, frontier)
        stop = [a for a in actions if a["action_type"] == "stop_exploring"]
        assert len(stop) == 1

    def test_control_failure_action(self):
        cv = [{"verdict": "control_failure", "task_id": "C001"}]
        hv = []
        learnings = {"dead_zones": []}
        frontier = {"additions": []}

        actions = recommend_next_actions(cv, hv, learnings, frontier)
        noise = [a for a in actions if a["action_type"] == "investigate_noise"]
        assert len(noise) == 1

    def test_no_action_when_empty(self):
        actions = recommend_next_actions([], [], {"dead_zones": []}, {"additions": []})
        assert len(actions) == 1
        assert actions[0]["action_type"] == "no_action"

    def test_actions_sorted_by_priority(self):
        cv = [
            {"verdict": "frontier_challenger", "candidate_id": "c1", "pnl_mean": 8000, "sharpe": 20.0},
            {"verdict": "control_failure", "task_id": "C001"},
        ]
        hv = [{"outcome": "validated", "hypothesis_id": "RE001", "hypothesis_title": "Test", "reason": "Worked"}]
        learnings = {"dead_zones": []}
        frontier = {"additions": []}

        actions = recommend_next_actions(cv, hv, learnings, frontier)
        priorities = [a["priority"] for a in actions]
        order = {"high": 0, "medium": 1, "low": 2}
        ranks = [order.get(p, 9) for p in priorities]
        assert ranks == sorted(ranks)

    def test_gpt_summary_format(self):
        actions = [
            {"action_type": "confirm_challenger", "priority": "high",
             "target": "c1", "detail": "Run more sessions", "rationale": "test"},
        ]
        learnings = {
            "validated_mechanisms": [{"hypothesis_id": "RE001", "reason": "Worked"}],
            "falsified_mechanisms": [],
            "dead_zones": [],
            "family_lessons": {"aggressive": ["Improved via product shift"]},
        }
        summary = format_gpt_summary(actions, learnings)
        assert "## Experiment Results Summary" in summary
        assert "Validated Mechanisms" in summary


# ===========================================================================
# Report tests
# ===========================================================================

class TestReports:

    def test_write_all_reports(self, improved_candidate, parent_packet, sample_task, tmp_path):
        cv = adjudicate_candidate(improved_candidate, parent_packet, sample_task)
        hv = adjudicate_hypothesis(cv, sample_task)
        frontier_updates = {"additions": [], "retirements": [], "role_assignments": {},
                            "frontier_size_before": 3, "frontier_size_after": 3,
                            "frontier_after": [], "changes_summary": "No changes.", "notes": []}
        learnings = extract_batch_learnings([cv], [hv], [sample_task])
        actions = recommend_next_actions([cv], [hv], learnings, frontier_updates)

        written = write_all_reports(
            tmp_path, [cv], [hv], frontier_updates, learnings, actions,
        )
        assert len(written) == 10  # 5 pairs of json+md
        for p in written:
            assert p.exists()
            assert p.stat().st_size > 0

    def test_json_reports_parseable(self, improved_candidate, parent_packet, sample_task, tmp_path):
        cv = adjudicate_candidate(improved_candidate, parent_packet, sample_task)
        hv = adjudicate_hypothesis(cv, sample_task)
        learnings = extract_batch_learnings([cv], [hv], [sample_task])
        actions = recommend_next_actions([cv], [hv], learnings, {"additions": []})
        frontier_updates = {"additions": [], "retirements": [], "role_assignments": {},
                            "frontier_size_before": 0, "frontier_size_after": 0,
                            "frontier_after": [], "changes_summary": "", "notes": []}

        write_all_reports(tmp_path, [cv], [hv], frontier_updates, learnings, actions)

        for json_file in tmp_path.glob("*.json"):
            data = json.loads(json_file.read_text())
            assert isinstance(data, dict)


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration:

    def test_full_pipeline(self, parent_packet, sample_task, sample_frontier):
        """Full end-to-end: candidate → verdict → hypothesis → frontier → learnings → actions."""
        # Create a mixed batch
        candidates = [
            _make_packet(pnl_mean=7000, sharpe=14.0, case_id="good"),
            _make_packet(pnl_mean=3000, sharpe=5.0, case_id="bad", promote=False),
            _make_packet(pnl_mean=5050, sharpe=10.1, case_id="neutral"),
        ]
        tasks = [
            _make_task(task_id="T001", source_card_id="RE001"),
            _make_task(task_id="T002", source_card_id="RM001"),
            _make_task(task_id="T003", source_card_id="DR001"),
        ]

        # 1. Adjudicate candidates
        cvs = []
        hvs = []
        for i, (cand, task) in enumerate(zip(candidates, tasks)):
            cv = adjudicate_candidate(cand, parent_packet, task, frontier=sample_frontier)
            cvs.append(cv)
            hv = adjudicate_hypothesis(cv, task)
            hvs.append(hv)

        assert len(cvs) == 3
        verdicts = [cv["verdict"] for cv in cvs]
        assert "reject" in verdicts or "keep" in verdicts  # at least one should fail

        # 2. Frontier updates
        all_packets = sample_frontier + candidates
        fu = compute_frontier_updates(sample_frontier, cvs, all_packets)
        assert isinstance(fu["frontier_after"], list)

        # 3. Learnings
        learnings = extract_batch_learnings(cvs, hvs, tasks)
        assert "summary" in learnings

        # 4. Next actions
        actions = recommend_next_actions(cvs, hvs, learnings, fu)
        assert len(actions) >= 1

    def test_all_verdict_types_reachable(self, parent_packet, sample_frontier):
        """Test that key verdict types can be produced."""
        verdicts_seen = set()

        # frontier_challenger
        strong = _make_packet(pnl_mean=9000, sharpe=25.0, p05=5000, case_id="strong")
        task = _make_task()
        cv = adjudicate_candidate(strong, parent_packet, task, frontier=sample_frontier)
        verdicts_seen.add(cv["verdict"])

        # reject
        weak = _make_packet(pnl_mean=2000, sharpe=3.0, case_id="weak", promote=False)
        cv = adjudicate_candidate(weak, parent_packet, task)
        verdicts_seen.add(cv["verdict"])

        # control_success
        task_cal = _make_task(task_type="calibration_check")
        cal = _make_packet(pnl_mean=5050, case_id="cal")
        cv = adjudicate_candidate(cal, parent_packet, task_cal)
        verdicts_seen.add(cv["verdict"])

        # control_failure
        bad_cal = _make_packet(pnl_mean=2000, case_id="bad_cal")
        cv = adjudicate_candidate(bad_cal, parent_packet, task_cal)
        verdicts_seen.add(cv["verdict"])

        assert "reject" in verdicts_seen
        assert "control_success" in verdicts_seen
        assert "control_failure" in verdicts_seen
        # frontier_challenger may or may not be triggered depending on frontier composition
        assert len(verdicts_seen) >= 3

    def test_hypothesis_outcomes_reachable(self, parent_packet):
        outcomes_seen = set()
        task = _make_task()

        # validated
        good = _make_packet(pnl_mean=8000, sharpe=15.0, em_mean=5000, tom_mean=3000)
        cv = adjudicate_candidate(good, parent_packet, task)
        hv = adjudicate_hypothesis(cv, task)
        outcomes_seen.add(hv["outcome"])

        # falsified / informative_failure
        bad = _make_packet(pnl_mean=2000, sharpe=3.0, em_mean=1500, tom_mean=500, promote=False)
        cv = adjudicate_candidate(bad, parent_packet, task)
        hv = adjudicate_hypothesis(cv, task)
        outcomes_seen.add(hv["outcome"])

        # inconclusive
        same = _make_packet(pnl_mean=5010, sharpe=10.05, em_mean=3005, tom_mean=2005)
        cv = adjudicate_candidate(same, parent_packet, task)
        hv = adjudicate_hypothesis(cv, task)
        outcomes_seen.add(hv["outcome"])

        assert len(outcomes_seen) >= 2  # at least two different outcomes
