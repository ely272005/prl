"""Tests for analytics/official_compare.py."""
from __future__ import annotations

from analytics.official_compare import compare_official_vs_local, parse_official_result


def _make_official(
    profit: float = 2702.59,
    emerald_pnl: float = 1050.0,
    tomato_pnl: float = 1652.6,
    ticks: int = 2000,
) -> dict:
    """Build a minimal official result JSON for testing."""
    # Build activitiesLog with per-product PnL at final tick
    header = "day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss"
    lines = [header]
    # Write one row per product at each of a few timestamps
    for ts in range(0, ticks * 100, 100):
        lines.append(
            f"-1;{ts};EMERALDS;9998;20;9997;15;;;10002;20;10003;15;;;10000.0;{emerald_pnl}"
        )
        lines.append(
            f"-1;{ts};TOMATOES;4998;15;4997;10;;;5002;15;5003;10;;;5000.0;{tomato_pnl}"
        )

    graph_lines = ["timestamp;value"]
    for i in range(500):
        graph_lines.append(f"{i * 400};{profit * i / 500}")

    return {
        "round": 0,
        "status": "FINISHED",
        "profit": profit,
        "activitiesLog": "\n".join(lines),
        "graphLog": "\n".join(graph_lines),
    }


def _make_packet_short(
    mean_pnl: float = 16552.0,
    std_pnl: float = 1058.0,
    emerald_mean: float = 7543.0,
    tomato_mean: float = 9010.0,
    ticks_per_session: int = 10000,
) -> dict:
    """Build a minimal local Packet Short for testing."""
    return {
        "pnl": {"mean": mean_pnl, "std": std_pnl},
        "per_product": {
            "emerald": {"mean": emerald_mean},
            "tomato": {"mean": tomato_mean},
        },
        "scale": {
            "session_count": 200,
            "sample_session_count": 25,
            "ticks_per_session": ticks_per_session,
        },
    }


class TestParseOfficialResult:
    def test_basic_parse(self):
        off = _make_official()
        parsed = parse_official_result(off)
        assert parsed["total_pnl"] == 2702.59
        assert parsed["status"] == "FINISHED"
        assert "EMERALDS" in parsed["per_product_pnl"]
        assert "TOMATOES" in parsed["per_product_pnl"]
        assert parsed["official_ticks"] > 0

    def test_per_product_pnl(self):
        off = _make_official(emerald_pnl=500.0, tomato_pnl=200.0)
        parsed = parse_official_result(off)
        assert parsed["per_product_pnl"]["EMERALDS"] == 500.0
        assert parsed["per_product_pnl"]["TOMATOES"] == 200.0

    def test_empty_activities(self):
        off = {"profit": 100.0, "status": "FINISHED", "activitiesLog": "", "graphLog": ""}
        parsed = parse_official_result(off)
        assert parsed["total_pnl"] == 100.0
        assert parsed["per_product_pnl"] == {}


class TestCompareOfficialVsLocal:
    def test_basic_comparison(self):
        off = _make_official(profit=2702.59)
        pkt = _make_packet_short(mean_pnl=16552.0, ticks_per_session=10000)
        result = compare_official_vs_local(off, pkt)

        assert result["official_pnl"] == 2702.59
        assert result["local_mean_pnl"] == 16552.0
        assert result["raw_ratio"] is not None
        assert result["raw_ratio"] > 5.0  # local inflated

    def test_time_normalization(self):
        off = _make_official(profit=2702.59, ticks=2000)
        pkt = _make_packet_short(mean_pnl=16552.0, ticks_per_session=10000)
        result = compare_official_vs_local(off, pkt)

        assert result["time_ratio"] == 10000 / 2000  # 5.0
        assert result["normalized_local_pnl"] is not None
        # 16552 / 5 = 3310.4
        assert abs(result["normalized_local_pnl"] - 3310.4) < 1.0
        assert result["normalized_ratio"] is not None

    def test_sign_flip_detected(self):
        """LOW case: TOMATOES profitable locally but losing officially."""
        off = _make_official(profit=347.0, emerald_pnl=495.0, tomato_pnl=-148.0, ticks=2000)
        pkt = _make_packet_short(
            mean_pnl=5152.0, emerald_mean=3545.0, tomato_mean=1608.0,
            ticks_per_session=10000,
        )
        result = compare_official_vs_local(off, pkt)

        # TOMATOES should have sign flip
        assert "TOMATOES" in result["per_product"]
        assert result["per_product"]["TOMATOES"]["sign_flip"] is True

        # Should have a warning about it
        sign_warnings = [w for w in result["warnings"] if "sign flip" in w.lower()]
        assert len(sign_warnings) > 0

        # EMERALDS should NOT have sign flip
        assert result["per_product"]["EMERALDS"]["sign_flip"] is False

    def test_no_sign_flip_when_both_positive(self):
        off = _make_official(profit=2702.0, emerald_pnl=1050.0, tomato_pnl=1652.0, ticks=2000)
        pkt = _make_packet_short(
            mean_pnl=16552.0, emerald_mean=7543.0, tomato_mean=9010.0,
            ticks_per_session=10000,
        )
        result = compare_official_vs_local(off, pkt)
        assert result["per_product"]["EMERALDS"]["sign_flip"] is False
        assert result["per_product"]["TOMATOES"]["sign_flip"] is False

    def test_missing_ticks_warns(self):
        off = _make_official()
        pkt = _make_packet_short()
        pkt["scale"]["ticks_per_session"] = None
        result = compare_official_vs_local(off, pkt)
        assert result["normalized_local_pnl"] is None
        assert any("Cannot time-normalize" in w for w in result["warnings"])

    def test_single_run_warning(self):
        """Should always warn that official is single-run."""
        off = _make_official()
        pkt = _make_packet_short()
        result = compare_official_vs_local(off, pkt)
        assert any("single run" in w.lower() for w in result["warnings"])

    def test_per_product_normalized_ratio(self):
        off = _make_official(profit=1468.0, emerald_pnl=0.0, tomato_pnl=1468.0, ticks=2000)
        pkt = _make_packet_short(
            mean_pnl=7481.0, emerald_mean=0.0, tomato_mean=7481.0,
            ticks_per_session=10000,
        )
        result = compare_official_vs_local(off, pkt)
        tom = result["per_product"]["TOMATOES"]
        # 7481 / 5 = 1496.2, ratio = 1496.2 / 1468 ≈ 1.02
        assert tom["normalized_ratio"] is not None
        assert abs(tom["normalized_ratio"] - 1.02) < 0.05
