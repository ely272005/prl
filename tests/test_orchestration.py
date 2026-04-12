"""Tests for the Campaign Orchestrator (Phase 6).

Covers: campaign creation, routing logic, official queue ranking,
champion management, redundancy detection, exploit/explore allocation,
budget-aware campaign sizing, prosperity handoff artifacts,
campaign history serialization, run plan generation, report writing.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_verdict(
    candidate_id: str = "cand01",
    verdict: str = "escalate",
    pnl_mean: float = 5000,
    sharpe: float = 8.0,
    pnl_delta: float = 500,
    sharpe_delta: float = 1.5,
    confidence: str = "HIGH",
    positive_rate: float = 0.70,
    p05: float = 1200,
    p50: float = 5000,
    family: str = "maker_heavy",
    parent_id: str = "73474",
    task_id: str = "T001",
    task_type: str = "exploit",
    source_hypothesis: str = "AC001",
    promote_recommended: bool = True,
    kill_recommended: bool = False,
    transfer_risk: str = "Low transfer risk — clean gain pattern.",
    suspicion_level: str = "clean",
    suspicion_flags: list | None = None,
    preservation_verdict: str = "clean",
    preservation_violations: list | None = None,
    dominant_mechanism: str = "product_shift",
    emerald_mean: float = 3000,
    tomato_mean: float = 2000,
    emerald_delta: float = 300,
    tomato_delta: float = 200,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "task_id": task_id,
        "parent_id": parent_id,
        "family": family,
        "batch_id": "B001",
        "source_hypothesis": source_hypothesis,
        "task_type": task_type,
        "pnl_mean": pnl_mean,
        "pnl_std": 1000,
        "sharpe": sharpe,
        "p05": p05,
        "p50": p50,
        "p95": 9000,
        "positive_rate": positive_rate,
        "confidence": confidence,
        "promote_recommended": promote_recommended,
        "kill_recommended": kill_recommended,
        "emerald_mean": emerald_mean,
        "tomato_mean": tomato_mean,
        "emerald_delta": emerald_delta,
        "tomato_delta": tomato_delta,
        "pnl_delta": pnl_delta,
        "sharpe_delta": sharpe_delta,
        "vs_parent": {"reference_label": "parent", "deltas": [], "improvements": 5, "regressions": 2},
        "vs_frontier": {"beats_frontier_on": ["pnl_mean", "sharpe"] if verdict == "frontier_challenger" else [], "below_frontier_on": []},
        "vs_family": None,
        "comparison_summary": "Candidate improves on parent.",
        "preservation_audit": {"verdict": preservation_verdict, "reason": "OK", "violations": preservation_violations or []},
        "attribution": {
            "attributions": [{"mechanism": dominant_mechanism, "magnitude": 500}],
            "dominant_mechanism": dominant_mechanism,
            "summary": f"Gain driven by {dominant_mechanism}.",
        },
        "suspicion": {
            "suspicion_level": suspicion_level,
            "flags": suspicion_flags or [],
            "is_suspicious": suspicion_level in ("high", "medium"),
            "flag_count": len(suspicion_flags or []),
        },
        "verdict": verdict,
        "reason": f"Verdict: {verdict}",
        "mechanism_interpretation": f"Gain driven by {dominant_mechanism}.",
        "transfer_risk": transfer_risk,
        "recommended_next_action": "Run more sessions.",
    }


def _make_hypothesis_verdict(
    hypothesis_id: str = "AC001",
    task_id: str = "T001",
    outcome: str = "validated",
    reason: str = "Mechanism worked.",
    single_product_only: bool = False,
) -> dict:
    return {
        "hypothesis_id": hypothesis_id,
        "hypothesis_title": f"Hypothesis {hypothesis_id}",
        "task_id": task_id,
        "task_type": "exploit",
        "outcome": outcome,
        "reason": reason,
        "checks": [],
        "lessons": ["Lesson A"],
        "mean_helped": outcome in ("validated", "partially_validated"),
        "sharpe_helped": outcome == "validated",
        "single_product_only": single_product_only,
    }


def _make_next_action(
    action_type: str = "confirm_challenger",
    target: str = "cand01",
    priority: str = "high",
) -> dict:
    return {
        "action_type": action_type,
        "priority": priority,
        "target": target,
        "detail": f"{action_type} for {target}",
        "rationale": "Test rationale.",
    }


def _make_learnings(
    validated: list | None = None,
    falsified: list | None = None,
    dead_zones: list | None = None,
    family_lessons: dict | None = None,
) -> dict:
    return {
        "validated_mechanisms": validated or [],
        "falsified_mechanisms": falsified or [],
        "suspicious_directions": [],
        "promising_zones": [],
        "dead_zones": dead_zones or [],
        "family_lessons": family_lessons or {},
        "card_lessons": {},
        "parent_lessons": {},
        "summary": "Test learnings.",
    }


def _make_frontier_updates(
    additions: list | None = None,
    retirements: list | None = None,
) -> dict:
    return {
        "additions": additions or [],
        "retirements": retirements or [],
        "role_assignments": {},
        "frontier_size_before": 5,
        "frontier_size_after": 5 + len(additions or []),
        "frontier_after": [],
        "changes_summary": "Test frontier update.",
        "notes": [],
    }


def _make_packet(
    case_id: str = "73474",
    pnl_mean: float = 4500,
    sharpe: float = 7.0,
    positive_rate: float = 0.65,
    p05: float = 1000,
    confidence: str = "HIGH",
    promote: bool = True,
    family: str = "maker_heavy",
    em_mean: float = 2500,
    tom_mean: float = 2000,
) -> dict:
    return {
        "_case_id": case_id,
        "_family": family,
        "packet_short": {
            "candidate_id": case_id,
            "confidence": confidence,
            "pnl": {
                "mean": pnl_mean,
                "std": 900,
                "sharpe_like": sharpe,
                "p05": p05,
                "p50": pnl_mean,
                "p95": pnl_mean * 1.8,
                "positive_rate": positive_rate,
            },
            "per_product": {
                "emerald": {"mean": em_mean, "sharpe_like": 5.0},
                "tomato": {"mean": tom_mean, "sharpe_like": 4.0},
            },
            "fill_quality": {
                "passive_fill_rate": 0.60,
                "taker_fill_count": 100,
                "maker_fill_count": 200,
            },
            "drawdown": {"mean_max_drawdown": -800},
            "efficiency": {"pnl_per_fill": 1.5},
            "promote": {"recommended": promote},
            "kill": {"recommended": False},
        },
    }


# ===========================================================================
# Test: Campaigns
# ===========================================================================

class TestCampaigns:
    def setup_method(self):
        from orchestration.campaigns import reset_counter
        reset_counter()

    def test_campaign_types_defined(self):
        from orchestration.campaigns import CAMPAIGN_TYPES
        assert "exploration" in CAMPAIGN_TYPES
        assert "confirmation" in CAMPAIGN_TYPES
        assert "official_gate" in CAMPAIGN_TYPES
        assert "calibration" in CAMPAIGN_TYPES
        assert "champion_defense" in CAMPAIGN_TYPES

    def test_create_campaign_basic(self):
        from orchestration.campaigns import create_campaign
        c = create_campaign(
            title="Test campaign",
            campaign_type="exploration",
            objective="Test objective",
        )
        assert c["title"] == "Test campaign"
        assert c["campaign_type"] == "exploration"
        assert c["status"] == "planned"
        assert c["campaign_id"].startswith("CAM-")
        assert c["exploit_explore"] == "explore"

    def test_create_campaign_confirmation_defaults(self):
        from orchestration.campaigns import create_campaign
        c = create_campaign(
            title="Confirm test",
            campaign_type="confirmation",
            objective="Confirm",
        )
        assert c["exploit_explore"] == "exploit"
        assert c["novelty_tolerance"] == "low"
        assert c["budget"]["max_sessions_per_candidate"] == 50

    def test_create_campaigns_from_actions_confirm(self):
        from orchestration.campaigns import create_campaigns_from_actions
        actions = [_make_next_action("confirm_challenger", "cand01")]
        verdicts = [_make_verdict("cand01", verdict="frontier_challenger")]
        learnings = _make_learnings()
        frontier = _make_frontier_updates()

        campaigns = create_campaigns_from_actions(actions, learnings, frontier, verdicts)
        assert len(campaigns) >= 1
        confirm = [c for c in campaigns if c["campaign_type"] == "confirmation"]
        assert len(confirm) == 1
        assert "cand01" in confirm[0]["title"]

    def test_create_campaigns_from_actions_explore(self):
        from orchestration.campaigns import create_campaigns_from_actions
        actions = [_make_next_action("explore_further", "AC001")]
        learnings = _make_learnings()
        frontier = _make_frontier_updates()

        campaigns = create_campaigns_from_actions(actions, learnings, frontier, [])
        explore = [c for c in campaigns if c["campaign_type"] == "exploration"]
        assert len(explore) == 1

    def test_create_campaigns_skips_dead_zones(self):
        from orchestration.campaigns import create_campaigns_from_actions
        actions = [_make_next_action("explore_further", "DEAD_HYP")]
        learnings = _make_learnings(dead_zones=[{"hypothesis_id": "DEAD_HYP", "reason": "Exhausted"}])
        frontier = _make_frontier_updates()

        campaigns = create_campaigns_from_actions(actions, learnings, frontier, [])
        explore = [c for c in campaigns if c["target_mechanism"] == "DEAD_HYP"]
        assert len(explore) == 0

    def test_calibration_campaign_on_noise(self):
        from orchestration.campaigns import create_campaigns_from_actions
        actions = [_make_next_action("investigate_noise", "batch_controls")]
        learnings = _make_learnings()
        frontier = _make_frontier_updates()

        campaigns = create_campaigns_from_actions(actions, learnings, frontier, [])
        cal = [c for c in campaigns if c["campaign_type"] == "calibration"]
        assert len(cal) == 1

    def test_champion_defense_on_frontier_addition(self):
        from orchestration.campaigns import create_campaigns_from_actions
        actions = []
        learnings = _make_learnings()
        frontier = _make_frontier_updates(
            additions=[{"candidate_id": "new01", "verdict": "frontier_challenger"}],
        )

        campaigns = create_campaigns_from_actions(actions, learnings, frontier, [])
        defense = [c for c in campaigns if c["campaign_type"] == "champion_defense"]
        assert len(defense) == 1

    def test_campaign_ids_unique(self):
        from orchestration.campaigns import create_campaign
        ids = set()
        for i in range(10):
            c = create_campaign(title=f"T{i}", campaign_type="exploration", objective="X")
            ids.add(c["campaign_id"])
        assert len(ids) == 10


# ===========================================================================
# Test: Routing
# ===========================================================================

class TestRouting:
    def test_routing_rules_cover_all_verdicts(self):
        from orchestration.routing import ROUTING_RULES
        from adjudication.verdicts import VERDICT_LABELS
        for label in VERDICT_LABELS:
            assert label in ROUTING_RULES, f"No routing rule for verdict '{label}'"

    def test_frontier_challenger_routes_to_official(self):
        from orchestration.routing import route_candidate
        cv = _make_verdict(verdict="frontier_challenger")
        decision = route_candidate(cv)
        assert decision["action"] == "official_gate_shortlist"
        assert decision["official_eligible"] is True

    def test_escalate_low_risk_becomes_official(self):
        from orchestration.routing import route_candidate
        cv = _make_verdict(verdict="escalate", transfer_risk="Low transfer risk — clean gain pattern.")
        decision = route_candidate(cv)
        assert decision["official_eligible"] is True

    def test_escalate_high_risk_stays_confirmation(self):
        from orchestration.routing import route_candidate
        cv = _make_verdict(verdict="escalate", transfer_risk="Transfer risk: suspicious gain pattern.")
        decision = route_candidate(cv)
        assert decision["official_eligible"] is False
        assert decision["action"] == "confirmation_campaign"

    def test_reject_routes_to_dead_branch(self):
        from orchestration.routing import route_candidate
        cv = _make_verdict(verdict="reject")
        decision = route_candidate(cv)
        assert decision["action"] == "dead_branch"
        assert decision["official_eligible"] is False

    def test_suspect_routes_to_calibration(self):
        from orchestration.routing import route_candidate
        cv = _make_verdict(verdict="suspect_simulator_gain")
        decision = route_candidate(cv)
        assert decision["action"] == "calibration_only"
        assert decision["official_eligible"] is False

    def test_keep_routes_to_archive(self):
        from orchestration.routing import route_candidate
        cv = _make_verdict(verdict="keep")
        decision = route_candidate(cv)
        assert decision["action"] == "archive"

    def test_control_success_no_action(self):
        from orchestration.routing import route_candidate
        cv = _make_verdict(verdict="control_success")
        decision = route_candidate(cv)
        assert decision["action"] == "no_action"

    def test_control_failure_routes_to_calibration(self):
        from orchestration.routing import route_candidate
        cv = _make_verdict(verdict="control_failure")
        decision = route_candidate(cv)
        assert decision["action"] == "calibration_campaign"

    def test_dead_zone_warning_attached(self):
        from orchestration.routing import route_candidate
        cv = _make_verdict(verdict="keep", source_hypothesis="DEAD01")
        learnings = _make_learnings(dead_zones=[{"hypothesis_id": "DEAD01"}])
        decision = route_candidate(cv, learnings=learnings)
        assert "dead_zone_warning" in decision

    def test_route_batch(self):
        from orchestration.routing import route_candidates
        verdicts = [
            _make_verdict("a", verdict="frontier_challenger"),
            _make_verdict("b", verdict="reject"),
            _make_verdict("c", verdict="keep"),
        ]
        decisions = route_candidates(verdicts)
        assert len(decisions) == 3
        actions = {d["candidate_id"]: d["action"] for d in decisions}
        assert actions["a"] == "official_gate_shortlist"
        assert actions["b"] == "dead_branch"
        assert actions["c"] == "archive"


# ===========================================================================
# Test: Champions
# ===========================================================================

class TestChampions:
    def test_champion_roles_defined(self):
        from orchestration.champions import CHAMPION_ROLES
        assert "overall_champion" in CHAMPION_ROLES
        assert "best_sharpe" in CHAMPION_ROLES
        assert "best_mean" in CHAMPION_ROLES

    def test_build_champion_table(self):
        from orchestration.champions import build_champion_table
        packets = [
            _make_packet("p1", pnl_mean=5000, sharpe=8.0),
            _make_packet("p2", pnl_mean=6000, sharpe=6.0),
        ]
        table = build_champion_table(packets)
        champions = table["champions"]
        assert len(champions) > 0
        roles = {c["role"] for c in champions}
        assert "overall_champion" in roles
        assert "best_sharpe" in roles
        assert "best_mean" in roles

    def test_best_mean_is_highest(self):
        from orchestration.champions import build_champion_table
        packets = [
            _make_packet("low", pnl_mean=3000, sharpe=10.0),
            _make_packet("high", pnl_mean=8000, sharpe=5.0),
        ]
        table = build_champion_table(packets)
        best_mean = [c for c in table["champions"] if c["role"] == "best_mean"][0]
        assert best_mean["candidate_id"] == "high"

    def test_best_sharpe_is_highest(self):
        from orchestration.champions import build_champion_table
        packets = [
            _make_packet("low_s", pnl_mean=5000, sharpe=5.0),
            _make_packet("high_s", pnl_mean=5000, sharpe=12.0),
        ]
        table = build_champion_table(packets)
        best_sharpe = [c for c in table["champions"] if c["role"] == "best_sharpe"][0]
        assert best_sharpe["candidate_id"] == "high_s"

    def test_update_supersedes_old(self):
        from orchestration.champions import build_champion_table, update_champion_table
        old_packets = [_make_packet("old", pnl_mean=4000, sharpe=6.0)]
        old_table = build_champion_table(old_packets)

        new_packets = [_make_packet("new", pnl_mean=6000, sharpe=10.0)]
        new_table = update_champion_table(old_table, new_packets)

        superseded = [c for c in new_table["champions"] if c["status"] == "superseded"]
        assert len(superseded) > 0
        assert superseded[0]["superseded_by"] == "new"

    def test_promote_champion(self):
        from orchestration.champions import build_champion_table, promote_champion
        packets = [_make_packet("orig", pnl_mean=5000)]
        table = build_champion_table(packets)
        table = promote_champion(table, "new_champ", "best_mean", {"pnl_mean": 9000})
        active = [c for c in table["champions"] if c["role"] == "best_mean" and c["status"] == "active"]
        assert len(active) == 1
        assert active[0]["candidate_id"] == "new_champ"

    def test_retire_champion(self):
        from orchestration.champions import build_champion_table, retire_champion
        packets = [_make_packet("orig")]
        table = build_champion_table(packets)
        table = retire_champion(table, "orig", "No longer competitive")
        retired = [c for c in table["champions"] if c["candidate_id"] == "orig" and c["status"] == "retired"]
        assert len(retired) > 0

    def test_preserve_champion(self):
        from orchestration.champions import build_champion_table, preserve_champion
        packets = [_make_packet("anchor")]
        table = build_champion_table(packets)
        table = preserve_champion(table, "anchor", "Calibration reference")
        preserved = [c for c in table["champions"] if c["candidate_id"] == "anchor" and c["status"] == "preserved"]
        assert len(preserved) > 0

    def test_empty_frontier(self):
        from orchestration.champions import build_champion_table
        table = build_champion_table([])
        assert table["champions"] == []

    def test_maker_heavy_role(self):
        from orchestration.champions import build_champion_table
        packets = [
            _make_packet("mh01", family="maker_heavy", sharpe=9.0),
            _make_packet("ah01", family="active_heavy", sharpe=7.0),
        ]
        table = build_champion_table(packets)
        mh = [c for c in table["champions"] if c["role"] == "best_maker_heavy"]
        assert len(mh) == 1
        assert mh[0]["candidate_id"] == "mh01"


# ===========================================================================
# Test: Redundancy
# ===========================================================================

class TestRedundancy:
    def test_detect_near_duplicates(self):
        from orchestration.redundancy import detect_near_duplicates
        cand = [_make_verdict("c1", pnl_mean=5000, sharpe=8.0, positive_rate=0.70, emerald_mean=3000, tomato_mean=2000)]
        frontier = [_make_verdict("f1", pnl_mean=5010, sharpe=8.01, positive_rate=0.701, emerald_mean=3005, tomato_mean=2002)]
        dups = detect_near_duplicates(cand, frontier)
        assert len(dups) == 1
        assert dups[0]["candidate_id"] == "c1"

    def test_no_duplicate_if_different(self):
        from orchestration.redundancy import detect_near_duplicates
        cand = [_make_verdict("c1", pnl_mean=5000, sharpe=8.0)]
        frontier = [_make_verdict("f1", pnl_mean=2000, sharpe=3.0)]
        dups = detect_near_duplicates(cand, frontier)
        assert len(dups) == 0

    def test_detect_falsified_repeats(self):
        from orchestration.redundancy import detect_falsified_repeats
        from orchestration.campaigns import create_campaign, reset_counter
        reset_counter()
        camps = [create_campaign(title="Test", campaign_type="exploration", objective="X", target_mechanism="DEAD01")]
        dead = [{"hypothesis_id": "DEAD01", "reason": "Exhausted"}]
        repeats = detect_falsified_repeats(camps, dead)
        assert len(repeats) == 1

    def test_cross_campaign_overlap(self):
        from orchestration.redundancy import detect_cross_campaign_overlap
        from orchestration.campaigns import create_campaign, reset_counter
        reset_counter()
        a = create_campaign(title="A", campaign_type="exploration", objective="X", target_mechanism="M1", family="fam1")
        b = create_campaign(title="B", campaign_type="exploration", objective="Y", target_mechanism="M1", family="fam1")
        overlaps = detect_cross_campaign_overlap([a, b])
        assert len(overlaps) == 1

    def test_no_overlap_different_mechanism(self):
        from orchestration.redundancy import detect_cross_campaign_overlap
        from orchestration.campaigns import create_campaign, reset_counter
        reset_counter()
        a = create_campaign(title="A", campaign_type="exploration", objective="X", target_mechanism="M1", family="fam1")
        b = create_campaign(title="B", campaign_type="exploration", objective="Y", target_mechanism="M2", family="fam1")
        overlaps = detect_cross_campaign_overlap([a, b])
        assert len(overlaps) == 0

    def test_filter_redundant_campaigns(self):
        from orchestration.redundancy import filter_redundant_campaigns
        camps = [{"campaign_id": "C1"}, {"campaign_id": "C2"}]
        redundancy = {"falsified_repeats": [{"campaign_id": "C1"}]}
        filtered = filter_redundant_campaigns(camps, redundancy)
        assert len(filtered) == 1
        assert filtered[0]["campaign_id"] == "C2"

    def test_check_all_redundancy(self):
        from orchestration.redundancy import check_all_redundancy
        from orchestration.campaigns import create_campaign, reset_counter
        reset_counter()
        camps = [create_campaign(title="Test", campaign_type="exploration", objective="X")]
        cands = [_make_verdict("c1")]
        frontier = [_make_verdict("f1", pnl_mean=2000)]
        result = check_all_redundancy(camps, cands, frontier, [])
        assert "total_issues" in result


# ===========================================================================
# Test: Allocation
# ===========================================================================

class TestAllocation:
    def test_split_exploit_explore(self):
        from orchestration.allocation import split_exploit_explore
        camps = [
            {"exploit_explore": "exploit"},
            {"exploit_explore": "explore"},
            {"exploit_explore": "exploit"},
        ]
        exploit, explore = split_exploit_explore(camps)
        assert len(exploit) == 2
        assert len(explore) == 1

    def test_size_campaign_by_type(self):
        from orchestration.allocation import size_campaign
        assert size_campaign("confirmation") == 3
        assert size_campaign("exploration") == 6
        assert size_campaign("official_gate") == 2

    def test_size_campaign_priority_adjusts(self):
        from orchestration.allocation import size_campaign
        high = size_campaign("exploration", "high")
        low = size_campaign("exploration", "low")
        assert high > low

    def test_allocate_budget_basic(self):
        from orchestration.allocation import allocate_budget
        from orchestration.campaigns import create_campaign, reset_counter
        reset_counter()
        camps = [
            create_campaign(title="E1", campaign_type="exploration", objective="X"),
            create_campaign(title="C1", campaign_type="confirmation", objective="Y"),
        ]
        result = allocate_budget(camps, {"total_local_candidates": 20, "total_official_tests": 3, "max_campaigns": 6})
        assert result["total_allocated"] <= 20
        assert len(result["campaigns"]) == 2

    def test_allocate_budget_respects_limit(self):
        from orchestration.allocation import allocate_budget
        from orchestration.campaigns import create_campaign, reset_counter
        reset_counter()
        # 5 campaigns but budget for only ~10
        camps = [
            create_campaign(title=f"C{i}", campaign_type="exploration", objective="X")
            for i in range(5)
        ]
        result = allocate_budget(camps, {"total_local_candidates": 10, "total_official_tests": 3, "max_campaigns": 6})
        assert result["total_allocated"] <= 10

    def test_exploit_explore_ratio(self):
        from orchestration.allocation import allocate_budget
        from orchestration.campaigns import create_campaign, reset_counter
        reset_counter()
        camps = [
            create_campaign(title="Exploit1", campaign_type="confirmation", objective="X"),
            create_campaign(title="Explore1", campaign_type="exploration", objective="Y"),
        ]
        result = allocate_budget(camps, exploit_ratio=0.80)
        assert result["exploit_ratio"] == 0.80
        assert result["exploit_budget"] > result["explore_budget"]

    def test_max_campaigns_trims(self):
        from orchestration.allocation import allocate_budget
        from orchestration.campaigns import create_campaign, reset_counter
        reset_counter()
        camps = [
            create_campaign(title=f"C{i}", campaign_type="exploration", objective="X", priority="low")
            for i in range(10)
        ]
        result = allocate_budget(camps, {"total_local_candidates": 100, "total_official_tests": 3, "max_campaigns": 3})
        assert len(result["campaigns"]) <= 3


# ===========================================================================
# Test: Official Queue
# ===========================================================================

class TestOfficialQueue:
    def test_official_roles_defined(self):
        from orchestration.official_queue import OFFICIAL_ROLES
        assert "safest_challenger" in OFFICIAL_ROLES
        assert "highest_ceiling" in OFFICIAL_ROLES

    def test_build_official_queue_basic(self):
        from orchestration.official_queue import build_official_queue
        verdicts = [
            _make_verdict("c1", verdict="frontier_challenger", sharpe=10.0, confidence="HIGH"),
            _make_verdict("c2", verdict="escalate", sharpe=8.0, confidence="MEDIUM"),
        ]
        routing = [
            {"candidate_id": "c1", "official_eligible": True},
            {"candidate_id": "c2", "official_eligible": True},
        ]
        queue = build_official_queue(verdicts, routing, max_slots=3)
        assert len(queue) == 2
        assert queue[0]["rank"] == 1
        # Frontier challenger should rank higher
        assert queue[0]["candidate_id"] == "c1"

    def test_official_queue_empty_if_none_eligible(self):
        from orchestration.official_queue import build_official_queue
        verdicts = [_make_verdict("c1", verdict="reject")]
        routing = [{"candidate_id": "c1", "official_eligible": False}]
        queue = build_official_queue(verdicts, routing)
        assert len(queue) == 0

    def test_official_queue_respects_max_slots(self):
        from orchestration.official_queue import build_official_queue
        verdicts = [_make_verdict(f"c{i}", verdict="frontier_challenger") for i in range(10)]
        routing = [{"candidate_id": f"c{i}", "official_eligible": True} for i in range(10)]
        queue = build_official_queue(verdicts, routing, max_slots=2)
        assert len(queue) == 2

    def test_official_queue_has_required_fields(self):
        from orchestration.official_queue import build_official_queue
        verdicts = [_make_verdict("c1", verdict="frontier_challenger")]
        routing = [{"candidate_id": "c1", "official_eligible": True}]
        queue = build_official_queue(verdicts, routing)
        entry = queue[0]
        required = ["rank", "candidate_id", "parent_id", "family", "mechanism",
                     "local_metrics", "packet_quality", "transfer_risk",
                     "risk_notes", "reason", "is_main_bet", "is_control", "role", "status"]
        for field in required:
            assert field in entry, f"Missing field: {field}"

    def test_safest_challenger_assigned(self):
        from orchestration.official_queue import build_official_queue
        verdicts = [_make_verdict("c1", verdict="frontier_challenger", confidence="HIGH",
                                   transfer_risk="Low transfer risk — clean gain pattern.")]
        routing = [{"candidate_id": "c1", "official_eligible": True}]
        queue = build_official_queue(verdicts, routing)
        assert queue[0]["role"] == "safest_challenger"

    def test_generate_memo(self):
        from orchestration.official_queue import build_official_queue, generate_official_memo
        verdicts = [_make_verdict("c1", verdict="frontier_challenger")]
        routing = [{"candidate_id": "c1", "official_eligible": True}]
        queue = build_official_queue(verdicts, routing)
        memo = generate_official_memo(queue)
        assert "Official Testing Queue" in memo
        assert "c1" in memo

    def test_ranking_prefers_high_confidence(self):
        from orchestration.official_queue import build_official_queue
        verdicts = [
            _make_verdict("low_conf", verdict="frontier_challenger", confidence="LOW", sharpe=8.0),
            _make_verdict("high_conf", verdict="frontier_challenger", confidence="HIGH", sharpe=8.0),
        ]
        routing = [
            {"candidate_id": "low_conf", "official_eligible": True},
            {"candidate_id": "high_conf", "official_eligible": True},
        ]
        queue = build_official_queue(verdicts, routing)
        assert queue[0]["candidate_id"] == "high_conf"


# ===========================================================================
# Test: Run Plan
# ===========================================================================

class TestRunPlan:
    def setup_method(self):
        from orchestration.campaigns import reset_counter
        reset_counter()

    def test_build_run_plan_basic(self):
        from orchestration.run_plan import build_run_plan
        actions = [_make_next_action("confirm_challenger", "cand01")]
        verdicts = [_make_verdict("cand01", verdict="frontier_challenger")]
        learnings = _make_learnings()
        frontier = _make_frontier_updates()

        plan = build_run_plan(actions, learnings, frontier, verdicts)
        assert "plan_id" in plan
        assert "campaigns" in plan
        assert "routing_decisions" in plan
        assert len(plan["campaigns"]) >= 1

    def test_run_plan_has_summary(self):
        from orchestration.run_plan import build_run_plan
        plan = build_run_plan(
            [_make_next_action("confirm_challenger", "c1")],
            _make_learnings(),
            _make_frontier_updates(),
            [_make_verdict("c1", verdict="frontier_challenger")],
        )
        assert "summary" in plan
        assert len(plan["summary"]) > 0

    def test_run_plan_assigns_roles(self):
        from orchestration.run_plan import build_run_plan
        plan = build_run_plan(
            [_make_next_action("confirm_challenger", "c1")],
            _make_learnings(),
            _make_frontier_updates(),
            [_make_verdict("c1", verdict="frontier_challenger")],
        )
        for camp in plan["campaigns"]:
            if camp.get("allocated_candidates", 0) > 0:
                assert "planned_roles" in camp
                assert len(camp["planned_roles"]) > 0

    def test_run_plan_skips_dead_actions(self):
        from orchestration.run_plan import build_run_plan
        plan = build_run_plan(
            [_make_next_action("stop_exploring", "dead_hyp")],
            _make_learnings(),
            _make_frontier_updates(),
            [],
        )
        assert len(plan["skipped_actions"]) == 1

    def test_candidate_roles_defined(self):
        from orchestration.run_plan import CANDIDATE_ROLES
        assert "challenger" in CANDIDATE_ROLES
        assert "near_twin_control" in CANDIDATE_ROLES
        assert "calibration_anchor" in CANDIDATE_ROLES
        assert "exploration_probe" in CANDIDATE_ROLES


# ===========================================================================
# Test: Handoff
# ===========================================================================

class TestHandoff:
    def setup_method(self):
        from orchestration.campaigns import reset_counter
        reset_counter()

    def test_build_handoff_basic(self):
        from orchestration.handoff import build_campaign_handoff
        from orchestration.campaigns import create_campaign
        camp = create_campaign(
            title="Test",
            campaign_type="exploration",
            objective="Test objective",
            target_mechanism="role_shift",
            allowed_parents=["73474"],
            success_criteria="PnL up",
            failure_criteria="PnL down",
        )
        camp["allocated_candidates"] = 5
        learnings = _make_learnings()
        handoff = build_campaign_handoff(camp, learnings)
        assert "brief_markdown" in handoff
        assert "Objective" in handoff["brief_markdown"]
        assert "role_shift" in handoff["brief_markdown"]

    def test_handoff_includes_dead_zones(self):
        from orchestration.handoff import build_campaign_handoff
        from orchestration.campaigns import create_campaign
        camp = create_campaign(title="T", campaign_type="exploration", objective="X")
        learnings = _make_learnings(dead_zones=[{"hypothesis_id": "DZ01", "reason": "Dead"}])
        handoff = build_campaign_handoff(camp, learnings)
        assert "DZ01" in handoff["brief_markdown"]

    def test_handoff_includes_validated(self):
        from orchestration.handoff import build_campaign_handoff
        from orchestration.campaigns import create_campaign
        camp = create_campaign(title="T", campaign_type="exploration", objective="X")
        learnings = _make_learnings(validated=[{"hypothesis_id": "V01", "reason": "Works"}])
        handoff = build_campaign_handoff(camp, learnings)
        assert "V01" in handoff["brief_markdown"]

    def test_build_all_handoffs(self):
        from orchestration.handoff import build_all_handoffs
        from orchestration.campaigns import create_campaign
        camps = [
            create_campaign(title="A", campaign_type="exploration", objective="X"),
            create_campaign(title="B", campaign_type="confirmation", objective="Y"),
        ]
        camps[0]["allocated_candidates"] = 3
        camps[1]["allocated_candidates"] = 0  # deferred
        learnings = _make_learnings()
        handoffs = build_all_handoffs(camps, learnings)
        # Only active campaigns get handoffs
        assert len(handoffs) == 1

    def test_handoff_has_candidate_roles(self):
        from orchestration.handoff import build_campaign_handoff
        from orchestration.campaigns import create_campaign
        camp = create_campaign(title="T", campaign_type="exploration", objective="X")
        camp["allocated_candidates"] = 3
        camp["planned_roles"] = [
            {"role": "exploration_probe", "notes": "Probe #1."},
            {"role": "exploration_probe", "notes": "Probe #2."},
            {"role": "near_twin_control", "notes": "Control."},
        ]
        learnings = _make_learnings()
        handoff = build_campaign_handoff(camp, learnings)
        assert "exploration_probe" in handoff["brief_markdown"]


# ===========================================================================
# Test: History
# ===========================================================================

class TestHistory:
    def test_load_empty_history(self):
        from orchestration.history import load_history
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "history.json"
            history = load_history(path)
            assert history["campaigns"] == []

    def test_save_and_load(self):
        from orchestration.history import load_history, save_history
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "history.json"
            history = {"campaigns": [{"campaign_id": "C1", "title": "Test"}], "updated_at": None}
            save_history(history, path)
            loaded = load_history(path)
            assert len(loaded["campaigns"]) == 1
            assert loaded["campaigns"][0]["campaign_id"] == "C1"

    def test_record_campaign(self):
        from orchestration.history import record_campaign_result
        history = {"campaigns": [], "updated_at": None}
        campaign = {"campaign_id": "C1", "title": "Test", "campaign_type": "exploration",
                     "objective": "X", "family": "", "target_mechanism": "",
                     "exploit_explore": "explore", "allocated_candidates": 5, "created_at": "2026-01-01"}
        verdicts = [_make_verdict("c1")]
        updated = record_campaign_result(history, campaign, verdicts, was_worth_budget=True)
        assert len(updated["campaigns"]) == 1
        rec = updated["campaigns"][0]
        assert rec["was_worth_budget"] is True
        assert len(rec["candidate_verdicts"]) == 1

    def test_summarize_recent(self):
        from orchestration.history import summarize_recent
        history = {
            "campaigns": [
                {"title": "T1", "campaign_type": "exploration", "was_worth_budget": True,
                 "budget_used": 5, "candidate_verdicts": [{"candidate_id": "c1", "verdict": "keep"}]},
            ],
        }
        summary = summarize_recent(history)
        assert "T1" in summary

    def test_campaign_stats(self):
        from orchestration.history import campaign_stats
        history = {
            "campaigns": [
                {"campaign_type": "exploration", "was_worth_budget": True, "budget_used": 5},
                {"campaign_type": "confirmation", "was_worth_budget": False, "budget_used": 3},
            ],
        }
        stats = campaign_stats(history)
        assert stats["total"] == 2
        assert stats["worth_budget_yes"] == 1
        assert stats["worth_budget_no"] == 1


# ===========================================================================
# Test: Report Writing
# ===========================================================================

class TestReports:
    def setup_method(self):
        from orchestration.campaigns import reset_counter
        reset_counter()

    def test_write_all_reports(self):
        from orchestration.report import write_all_reports
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            plan = {
                "plan_id": "RP-TEST",
                "campaigns": [],
                "routing_decisions": [],
                "redundancy": {"total_issues": 0},
                "budget": {},
                "exploit_ratio": 0.7,
                "total_allocated": 0,
                "summary": "Test",
            }
            written = write_all_reports(
                out, plan, [], {"champions": [], "updated_at": ""}, [],
                {"campaigns": []}, [], None,
            )
            assert len(written) >= 10  # 6 pairs (JSON+MD) + routing pair = 12
            for p in written:
                assert p.exists()

    def test_json_parseable(self):
        from orchestration.report import write_all_reports
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            plan = {
                "plan_id": "RP-TEST",
                "campaigns": [],
                "routing_decisions": [],
                "redundancy": {"total_issues": 0},
                "budget": {},
                "exploit_ratio": 0.7,
                "total_allocated": 0,
                "summary": "Test",
            }
            written = write_all_reports(
                out, plan, [], {"champions": [], "updated_at": ""}, [],
                {"campaigns": []}, [], None,
            )
            json_files = [p for p in written if p.suffix == ".json"]
            for jf in json_files:
                data = json.loads(jf.read_text())
                assert isinstance(data, dict)


# ===========================================================================
# Test: Integration
# ===========================================================================

class TestIntegration:
    def setup_method(self):
        from orchestration.campaigns import reset_counter
        reset_counter()

    def test_full_pipeline(self):
        """End-to-end: Phase 5 outputs → campaigns → routing → official → handoff → reports."""
        from orchestration.run_plan import build_run_plan
        from orchestration.official_queue import build_official_queue
        from orchestration.champions import build_champion_table
        from orchestration.handoff import build_all_handoffs
        from orchestration.report import write_all_reports

        verdicts = [
            _make_verdict("mh07", verdict="frontier_challenger", sharpe=10.0, confidence="HIGH"),
            _make_verdict("ah03", verdict="escalate", sharpe=7.0, confidence="MEDIUM",
                          transfer_risk="Transfer risk: suspicious gain pattern."),
            _make_verdict("ctrl", verdict="control_success", task_type="calibration_check"),
            _make_verdict("bad", verdict="reject", pnl_delta=-500, sharpe_delta=-2.0),
        ]
        hypothesis_verdicts = [
            _make_hypothesis_verdict("AC001", "T001", "validated"),
            _make_hypothesis_verdict("AC002", "T002", "partially_validated"),
        ]
        actions = [
            _make_next_action("confirm_challenger", "mh07"),
            _make_next_action("explore_further", "AC001"),
            _make_next_action("stop_exploring", "DEAD01"),
        ]
        learnings = _make_learnings(
            validated=[{"hypothesis_id": "AC001", "reason": "Works"}],
            dead_zones=[{"hypothesis_id": "DEAD01", "reason": "Exhausted"}],
        )
        frontier = _make_frontier_updates(
            additions=[{"candidate_id": "mh07", "verdict": "frontier_challenger"}],
        )

        # Build run plan
        plan = build_run_plan(actions, learnings, frontier, verdicts, hypothesis_verdicts)
        assert len(plan["campaigns"]) >= 2
        assert plan["total_allocated"] > 0

        # Build official queue
        routing = plan["routing_decisions"]
        packets = [_make_packet("73474")]
        champions = build_champion_table(packets)
        official = build_official_queue(verdicts, routing, champions)
        assert len(official) >= 1
        assert official[0]["candidate_id"] == "mh07"

        # Handoffs
        handoffs = build_all_handoffs(plan["campaigns"], learnings, champions)
        assert len(handoffs) >= 1

        # Write reports
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            written = write_all_reports(
                out, plan, official, champions, routing,
                {"campaigns": []}, handoffs, learnings,
            )
            assert len(written) >= 10
            for p in written:
                assert p.exists()

    def test_exploit_explore_balance(self):
        """Verify exploit/explore campaigns are split correctly."""
        from orchestration.run_plan import build_run_plan

        actions = [
            _make_next_action("confirm_challenger", "c1"),  # → exploit (confirmation)
            _make_next_action("explore_further", "AC001"),  # → explore (exploration)
            _make_next_action("explore_further", "AC002"),  # → explore (exploration)
        ]
        verdicts = [_make_verdict("c1", verdict="frontier_challenger")]
        learnings = _make_learnings()
        frontier = _make_frontier_updates()

        plan = build_run_plan(actions, learnings, frontier, verdicts, exploit_ratio=0.5)
        campaigns = plan["campaigns"]
        exploit = [c for c in campaigns if c["exploit_explore"] == "exploit"]
        explore = [c for c in campaigns if c["exploit_explore"] == "explore"]
        assert len(exploit) >= 1
        assert len(explore) >= 1

    def test_redundancy_filters_dead_zone_campaigns(self):
        """Campaigns targeting dead zones should be filtered out."""
        from orchestration.run_plan import build_run_plan

        actions = [
            _make_next_action("explore_further", "DEAD_HYP"),
        ]
        learnings = _make_learnings(dead_zones=[{"hypothesis_id": "DEAD_HYP", "reason": "Exhausted"}])
        frontier = _make_frontier_updates()

        plan = build_run_plan(actions, learnings, frontier, [])
        # Dead zone campaign should not appear
        for camp in plan["campaigns"]:
            assert camp.get("target_mechanism") != "DEAD_HYP"
