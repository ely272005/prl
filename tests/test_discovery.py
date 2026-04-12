"""Tests for the Discovery Engine — regime extraction, comparison, weakness detection."""
import json
import math
import pytest
from pathlib import Path

from discovery.alpha_card import AlphaCard, CardCounter, _sanitize, CATEGORIES
from discovery.regimes import (
    label_extended_regimes,
    label_fills_with_regimes,
    compute_regime_edge_stats,
    build_regime_profile,
    REGIME_DIMENSIONS,
)
from discovery.comparison import (
    load_packets_from_bank,
    split_winners_losers,
    compare_winners_losers,
    compare_family_performance,
    run_comparison,
)
from discovery.weakness import (
    scan_regime_edges,
    scan_role_mismatches,
    scan_winner_traits,
    scan_probe_results,
    scan_inventory_exploits,
    run_all_scanners,
)
from discovery.scanner import DiscoveryScanner
from discovery.report import (
    build_json_report,
    build_markdown_report,
    write_json_report,
    write_markdown_report,
)
from engine.event_ledger import build_event_ledger


# ---- Fixtures ----

@pytest.fixture
def session_ledgers(output_dir):
    """Build session ledgers from the synthetic output_dir fixture."""
    ledger = build_event_ledger(output_dir)
    return ledger["session_ledgers"]


@pytest.fixture
def sample_packets():
    """Create synthetic research packets for comparison testing."""
    packets = []
    for i in range(10):
        promoted = i >= 5  # top half promoted
        pnl_mean = 1000 + i * 2000 if promoted else 100 + i * 50
        sharpe = 15.0 + i if promoted else 1.0 + i * 0.2
        family = "maker-heavy" if i % 2 == 0 else "aggressive"
        packets.append({
            "_case_id": f"test_{i:04d}",
            "_family": family,
            "pnl": {"mean": pnl_mean, "std": 500.0, "sharpe_like": sharpe, "p05": pnl_mean * 0.8},
            "per_product": {
                "emerald": {"mean": pnl_mean * 0.5, "std": 300.0, "sharpe_like": sharpe * 0.8},
                "tomato": {"mean": pnl_mean * 0.5, "std": 300.0, "sharpe_like": sharpe * 0.7},
            },
            "fill_quality": {
                "mean_fill_vs_fair_emerald": 4.0 + i * 0.3 if promoted else 1.0,
                "mean_fill_vs_fair_tomato": 3.5 + i * 0.2 if promoted else 0.5,
                "passive_fill_rate": 0.7 if promoted else 0.3,
                "taker_fill_count": 5000 + i * 100,
                "maker_fill_count": 12000 + i * 200,
            },
            "efficiency": {"pnl_per_fill": 15.0 if promoted else 2.0},
            "drawdown": {"mean_max_drawdown": -200.0 if promoted else -500.0},
            "inventory": {
                "mean_end_position_emerald": 2.0,
                "mean_end_position_tomato": 5.0,
            },
            "promote": {"recommended": promoted, "strength": sharpe},
            "kill": {"recommended": not promoted, "strength": sharpe},
        })
    return packets


@pytest.fixture
def bank_dir_with_packets(tmp_path, sample_packets):
    """Write sample packets to a temp bank directory."""
    bank = tmp_path / "bank"
    bank.mkdir()
    for i, pkt in enumerate(sample_packets):
        data = {"case_id": pkt["_case_id"], "family": pkt["_family"], "packet_short": pkt}
        path = bank / f"{pkt['_case_id']}_packet.json"
        path.write_text(json.dumps(data))
    return bank


@pytest.fixture
def sample_probe_results():
    """Synthetic probe results for probe-driven scanning."""
    return [
        {
            "probe_id": "pf01_spread_vs_maker_edge",
            "family": "passive_fill",
            "title": "Spread vs maker edge",
            "hypothesis": "Wider spreads give makers better fill quality.",
            "product": "EMERALDS",
            "verdict": "supported",
            "confidence": "high",
            "detail": "Maker edge increases monotonically with spread width.",
            "metrics": {"edge_by_spread_bucket": {"4-6": {"mean": 5.2, "count": 300}}},
            "sample_size": {"sessions": 25, "fills": 1200},
        },
        {
            "probe_id": "tk01_take_edge_by_distance",
            "family": "taking",
            "title": "Taker edge by distance",
            "hypothesis": "Taking further from fair captures more edge.",
            "product": "TOMATOES",
            "verdict": "refuted",
            "confidence": "medium",
            "detail": "No relationship between distance and taker edge.",
            "metrics": {"edge_by_distance_bucket": {}},
            "sample_size": {"sessions": 25, "fills": 800},
        },
        {
            "probe_id": "dz01_wide_spread_loss",
            "family": "danger_zone",
            "title": "Wide spread loss association",
            "hypothesis": "Wide spreads precede losses.",
            "product": "EMERALDS",
            "verdict": "inconclusive",
            "confidence": "low",
            "detail": "Mixed evidence.",
            "metrics": {},
            "sample_size": {"sessions": 25, "fills": 50},
        },
    ]


# ===========================================================================
# AlphaCard tests
# ===========================================================================

class TestAlphaCard:
    def test_card_to_dict(self):
        card = AlphaCard(
            card_id="RE01",
            title="Test card",
            category="regime_edge",
            products=["EMERALDS"],
            observed_fact="Edge is 5.0 in wide spreads.",
            interpretation="Wide spreads favor makers.",
            suggested_exploit="Target wide spread regime for making.",
            regime_definition={"spread_bucket": "6-8"},
            evidence={"mean": 5.0, "count": 200},
            baseline={"mean": 3.0, "count": 1000},
            sample_size={"fills": 200},
            confidence="high",
            strength=4.5,
        )
        d = card.to_dict()
        assert d["card_id"] == "RE01"
        assert d["category"] == "regime_edge"
        assert d["confidence"] == "high"
        assert d["evidence"]["mean"] == 5.0

    def test_card_nan_sanitization(self):
        card = AlphaCard(
            card_id="RE01", title="T", category="regime_edge",
            products=["EMERALDS"], observed_fact="f", interpretation="i",
            suggested_exploit="e",
            regime_definition={"val": math.nan},
            evidence={"x": math.nan, "y": float("inf")},
            baseline={}, sample_size={}, confidence="low",
            strength=math.nan,
        )
        d = card.to_dict()
        assert d["regime_definition"]["val"] is None
        assert d["evidence"]["x"] is None
        assert d["evidence"]["y"] is None
        assert d["strength"] is None

    def test_card_counter(self):
        counter = CardCounter()
        assert counter.next_id("regime_edge") == "RE01"
        assert counter.next_id("regime_edge") == "RE02"
        assert counter.next_id("winner_trait") == "WT01"
        assert counter.next_id("regime_edge") == "RE03"

    def test_sanitize_nested(self):
        obj = {"a": [1.0, math.nan, {"b": float("inf")}], "c": 3.14159}
        result = _sanitize(obj)
        assert result["a"][1] is None
        assert result["a"][2]["b"] is None
        assert result["c"] == round(3.14159, 6)


# ===========================================================================
# Regime extraction tests
# ===========================================================================

class TestRegimeExtraction:
    def test_label_extended_regimes_adds_columns(self, session_ledgers):
        sid = list(session_ledgers.keys())[0]
        traces = session_ledgers[sid]["traces"]
        prices = session_ledgers[sid]["prices"]

        labeled = label_extended_regimes(traces, prices)
        for dim in REGIME_DIMENSIONS:
            assert dim in labeled.columns, f"Missing column: {dim}"
        assert len(labeled) == len(traces)

    def test_label_extended_regimes_empty(self):
        import pandas as pd
        empty_traces = pd.DataFrame(columns=[
            "day", "timestamp", "product", "fair_value", "position", "cash", "mtm_pnl",
        ])
        empty_prices = pd.DataFrame(columns=[
            "day", "timestamp", "product", "bid1", "ask1", "mid_price",
        ])
        labeled = label_extended_regimes(empty_traces, empty_prices)
        for dim in REGIME_DIMENSIONS:
            assert dim in labeled.columns

    def test_spread_bucket_values(self, session_ledgers):
        sid = list(session_ledgers.keys())[0]
        traces = session_ledgers[sid]["traces"]
        prices = session_ledgers[sid]["prices"]
        labeled = label_extended_regimes(traces, prices)

        # EMERALDS has constant spread=4, so should be in "2-4" bucket
        emerald = labeled[labeled["product"] == "EMERALDS"]
        spread_vals = emerald["spread_bucket"].unique()
        assert "2-4" in spread_vals or "4-6" in spread_vals  # spread is exactly 4

    def test_trend_10_values(self, session_ledgers):
        sid = list(session_ledgers.keys())[0]
        traces = session_ledgers[sid]["traces"]
        prices = session_ledgers[sid]["prices"]
        labeled = label_extended_regimes(traces, prices)

        trend_vals = set(labeled["trend_10"].unique())
        # Should contain "unknown" for first 10 ticks and at least one direction
        assert "unknown" in trend_vals

    def test_maker_friendly_values(self, session_ledgers):
        sid = list(session_ledgers.keys())[0]
        traces = session_ledgers[sid]["traces"]
        prices = session_ledgers[sid]["prices"]
        labeled = label_extended_regimes(traces, prices)

        valid = {"maker_friendly", "taker_friendly", "neutral"}
        for val in labeled["maker_friendly"].unique():
            assert val in valid

    def test_build_regime_profile(self, session_ledgers):
        profile = build_regime_profile(session_ledgers)
        assert profile["session_count"] == 2
        assert "EMERALDS" in profile["regime_stats"]
        assert "TOMATOES" in profile["regime_stats"]

        # Check that regime stats contain all dimensions
        for dim in REGIME_DIMENSIONS:
            assert dim in profile["regime_stats"]["EMERALDS"]
            for role_key in ("all", "maker", "taker"):
                stats = profile["regime_stats"]["EMERALDS"][dim].get(role_key, {})
                assert "baseline" in stats
                assert "by_label" in stats

    def test_compute_regime_edge_stats_empty(self):
        import pandas as pd
        empty = pd.DataFrame()
        result = compute_regime_edge_stats(empty, "EMERALDS", "spread_bucket")
        assert result["total_fills"] == 0

    def test_label_fills_with_regimes(self, session_ledgers):
        sid = list(session_ledgers.keys())[0]
        traces = session_ledgers[sid]["traces"]
        prices = session_ledgers[sid]["prices"]
        fills = session_ledgers[sid]["strategy_fills"]

        labeled_traces = label_extended_regimes(traces, prices)
        labeled_fills = label_fills_with_regimes(fills, labeled_traces, prices)

        if not labeled_fills.empty:
            # Fills should have regime columns
            for dim in REGIME_DIMENSIONS:
                assert dim in labeled_fills.columns, f"Missing fill column: {dim}"


# ===========================================================================
# Comparison tests
# ===========================================================================

class TestComparison:
    def test_split_promote(self, sample_packets):
        winners, losers = split_winners_losers(sample_packets, "promote")
        assert len(winners) == 5
        assert len(losers) == 5
        assert all(p["promote"]["recommended"] for p in winners)

    def test_split_median(self, sample_packets):
        winners, losers = split_winners_losers(sample_packets, "median")
        assert len(winners) == 5
        assert len(losers) == 5

    def test_split_quartile(self, sample_packets):
        winners, losers = split_winners_losers(sample_packets, "quartile")
        assert len(winners) >= 2
        assert len(losers) >= 2

    def test_compare_winners_losers(self, sample_packets):
        winners, losers = split_winners_losers(sample_packets, "promote")
        cmp = compare_winners_losers(winners, losers)
        assert "pnl_mean" in cmp
        assert cmp["pnl_mean"]["winner_mean"] > cmp["pnl_mean"]["loser_mean"]
        assert cmp["pnl_mean"]["effect_size"] > 0

    def test_compare_family_performance(self, sample_packets):
        result = compare_family_performance(sample_packets)
        assert "maker-heavy" in result
        assert "aggressive" in result
        assert result["maker-heavy"]["count"] >= 1

    def test_load_packets_from_bank(self, bank_dir_with_packets):
        packets = load_packets_from_bank(bank_dir_with_packets)
        assert len(packets) == 10
        assert all("_case_id" in p for p in packets)
        assert all("pnl" in p for p in packets)

    def test_run_comparison(self, bank_dir_with_packets):
        result = run_comparison(bank_dir_with_packets)
        assert result["packet_count"] == 10
        assert result["winner_count"] == 5
        assert result["loser_count"] == 5
        assert "metric_comparison" in result
        assert "family_comparison" in result

    def test_run_comparison_empty_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = run_comparison(empty)
        assert result["packet_count"] == 0


# ===========================================================================
# Weakness detection tests
# ===========================================================================

class TestWeaknessDetection:
    def test_scan_regime_edges_produces_cards(self, session_ledgers):
        profile = build_regime_profile(session_ledgers)
        counter = CardCounter()
        cards = scan_regime_edges(profile["regime_stats"], counter)
        # May or may not produce cards depending on synthetic data
        for card in cards:
            assert isinstance(card, AlphaCard)
            assert card.category in ("regime_edge", "danger_refinement")
            assert card.confidence in ("high", "medium", "low")
            assert card.sample_size

    def test_scan_role_mismatches(self, session_ledgers):
        profile = build_regime_profile(session_ledgers)
        counter = CardCounter()
        cards = scan_role_mismatches(profile["regime_stats"], counter)
        for card in cards:
            assert card.category == "role_mismatch"

    def test_scan_winner_traits(self, sample_packets):
        winners, losers = split_winners_losers(sample_packets, "promote")
        cmp = compare_winners_losers(winners, losers)
        comparison = {
            "metric_comparison": cmp,
            "family_comparison": compare_family_performance(sample_packets),
            "winner_count": len(winners),
            "loser_count": len(losers),
        }
        counter = CardCounter()
        cards = scan_winner_traits(comparison, counter)
        # Should find at least one winner trait (pnl, fill quality differ)
        assert len(cards) >= 1
        for card in cards:
            assert card.category == "winner_trait"
            assert card.observed_fact
            assert card.interpretation

    def test_scan_probe_results(self, sample_probe_results):
        counter = CardCounter()
        cards = scan_probe_results(sample_probe_results, counter)
        # Should pick up the two with supported/refuted AND confidence >= medium
        assert len(cards) == 2  # pf01 (supported+high) and tk01 (refuted+medium)
        assert cards[0].category in ("regime_edge", "inventory_exploit", "danger_refinement")

    def test_scan_probe_results_skips_low_confidence(self, sample_probe_results):
        counter = CardCounter()
        cards = scan_probe_results(sample_probe_results, counter)
        probe_ids = [c.regime_definition.get("probe_id") for c in cards]
        # dz01 is inconclusive+low → should not appear
        assert "dz01_wide_spread_loss" not in probe_ids

    def test_scan_inventory_exploits(self, session_ledgers):
        profile = build_regime_profile(session_ledgers)
        counter = CardCounter()
        cards = scan_inventory_exploits(profile["regime_stats"], counter)
        for card in cards:
            assert card.category == "inventory_exploit"

    def test_run_all_scanners(self, session_ledgers, sample_packets, sample_probe_results):
        profile = build_regime_profile(session_ledgers)
        winners, losers = split_winners_losers(sample_packets, "promote")
        cmp = compare_winners_losers(winners, losers)
        comparison = {
            "metric_comparison": cmp,
            "family_comparison": compare_family_performance(sample_packets),
            "winner_count": len(winners),
            "loser_count": len(losers),
        }
        cards = run_all_scanners(
            regime_stats=profile["regime_stats"],
            comparison=comparison,
            probe_results=sample_probe_results,
            max_cards=20,
        )
        assert isinstance(cards, list)
        # Should produce at least some cards from winner traits + probes
        assert len(cards) >= 2
        # Cards should be ranked by strength (descending)
        for i in range(len(cards) - 1):
            assert cards[i].strength >= cards[i + 1].strength

    def test_run_all_scanners_empty(self):
        cards = run_all_scanners(regime_stats={}, max_cards=10)
        assert cards == []


# ===========================================================================
# Scanner integration tests
# ===========================================================================

class TestDiscoveryScanner:
    def test_scanner_session_only(self, output_dir):
        scanner = DiscoveryScanner(output_dirs=[output_dir])
        summary = scanner.summary()
        assert summary["session_count"] == 2

    def test_scanner_discover(self, output_dir):
        scanner = DiscoveryScanner(output_dirs=[output_dir])
        cards = scanner.discover(max_cards=10)
        assert isinstance(cards, list)
        for card in cards:
            assert isinstance(card, AlphaCard)

    def test_scanner_with_bank(self, output_dir, bank_dir_with_packets):
        scanner = DiscoveryScanner(
            output_dirs=[output_dir],
            bank_dir=bank_dir_with_packets,
        )
        cards = scanner.discover(max_cards=15)
        # Should have winner trait cards from bank comparison
        categories = {c.category for c in cards}
        assert "winner_trait" in categories or len(cards) >= 1

    def test_scanner_single_product(self, output_dir):
        scanner = DiscoveryScanner(output_dirs=[output_dir], products=["TOMATOES"])
        cards = scanner.discover(max_cards=10)
        for card in cards:
            # All cards should only reference TOMATOES (or both)
            if card.category not in ("winner_trait",):
                assert "TOMATOES" in card.products


# ===========================================================================
# Report tests
# ===========================================================================

class TestReportGeneration:
    def _make_cards(self):
        return [
            AlphaCard(
                card_id="RE01", title="Edge in wide spreads",
                category="regime_edge", products=["EMERALDS"],
                observed_fact="Edge is 5.0.", interpretation="Good.",
                suggested_exploit="Target it.",
                regime_definition={"spread_bucket": "6-8"},
                evidence={"mean": 5.0}, baseline={"mean": 3.0},
                sample_size={"fills": 200}, confidence="high", strength=4.5,
                candidate_strategy_style="Maker", recommended_experiment="Test it.",
            ),
            AlphaCard(
                card_id="WT01", title="Winners have higher PnL",
                category="winner_trait", products=["EMERALDS", "TOMATOES"],
                observed_fact="PnL diff is 10k.", interpretation="Obvious.",
                suggested_exploit="Be better.",
                regime_definition={"metric": "pnl_mean"},
                evidence={"effect_size": 2.0}, baseline={"loser_mean": 200},
                sample_size={"winners": 5, "losers": 5}, confidence="medium",
                strength=3.0,
            ),
        ]

    def test_json_report_structure(self):
        cards = self._make_cards()
        report = build_json_report(cards)
        assert "generated_at" in report
        assert "alpha_cards" in report
        assert report["summary"]["total_cards"] == 2
        assert "strong_patterns" in report["classification"]

    def test_json_serializable(self):
        cards = self._make_cards()
        report = build_json_report(cards)
        serialized = json.dumps(report, default=str)
        assert len(serialized) > 100
        parsed = json.loads(serialized)
        assert parsed["summary"]["total_cards"] == 2

    def test_markdown_report(self):
        cards = self._make_cards()
        md = build_markdown_report(cards)
        assert "# Discovery Report" in md
        assert "RE01" in md
        assert "WT01" in md
        assert "OBSERVED FACT" in md
        assert "INTERPRETATION" in md
        assert "SUGGESTED EXPLOIT" in md

    def test_write_json_report(self, tmp_path):
        cards = self._make_cards()
        path = tmp_path / "test_report.json"
        write_json_report(cards, path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["summary"]["total_cards"] == 2

    def test_write_markdown_report(self, tmp_path):
        cards = self._make_cards()
        path = tmp_path / "test_report.md"
        write_markdown_report(cards, path)
        assert path.exists()
        content = path.read_text()
        assert "Alpha Cards" in content

    def test_report_with_comparison(self):
        cards = self._make_cards()
        comparison = {
            "packet_count": 10,
            "winner_count": 5,
            "loser_count": 5,
            "split_method": "promote",
            "family_comparison": {
                "maker-heavy": {"count": 3, "promoted": 2, "pnl_mean": 13000, "sharpe_mean": 18.0},
                "aggressive": {"count": 2, "promoted": 1, "pnl_mean": 14000, "sharpe_mean": 16.0},
            },
        }
        report = build_json_report(cards, comparison=comparison)
        assert "comparison" in report
        md = build_markdown_report(cards, comparison=comparison)
        assert "Family Performance" in md
