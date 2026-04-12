"""Winner vs loser comparison — compares promoted vs rejected candidates.

Loads research packets from the bank and identifies systematic differences
between strong and weak candidates across key metrics.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any


def load_packets_from_bank(bank_dir: Path) -> list[dict[str, Any]]:
    """Load all *_packet.json files from a bank directory.

    Returns list of dicts, each containing the packet_short fields plus
    case_id and family metadata.
    """
    packets = []
    for path in sorted(bank_dir.glob("*_packet.json")):
        try:
            raw = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        ps = raw.get("packet_short", raw)
        ps["_case_id"] = raw.get("case_id", path.stem.replace("_packet", ""))
        ps["_family"] = raw.get("family", "unknown")
        ps["_path"] = str(path)
        packets.append(ps)
    return packets


def split_winners_losers(
    packets: list[dict[str, Any]],
    method: str = "promote",
) -> tuple[list[dict], list[dict]]:
    """Split packets into winners and losers.

    Methods:
        "promote": use promote.recommended flag
        "median": split by median PnL
        "quartile": top 25% vs bottom 25%
    """
    if method == "promote":
        winners = [p for p in packets if p.get("promote", {}).get("recommended", False)]
        losers = [p for p in packets if not p.get("promote", {}).get("recommended", False)]
    elif method == "quartile":
        by_pnl = sorted(packets, key=lambda p: p.get("pnl", {}).get("mean", 0))
        n = len(by_pnl)
        q1 = max(1, n // 4)
        losers = by_pnl[:q1]
        winners = by_pnl[-q1:]
    else:  # median
        by_pnl = sorted(packets, key=lambda p: p.get("pnl", {}).get("mean", 0))
        mid = len(by_pnl) // 2
        losers = by_pnl[:mid]
        winners = by_pnl[mid:]

    return winners, losers


def _safe_mean(values: list[float]) -> float:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    return statistics.fmean(clean) if clean else math.nan


def _safe_std(values: list[float]) -> float:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if len(clean) < 2:
        return 0.0
    return statistics.stdev(clean)


def _extract_metric(packets: list[dict], *keys: str) -> list[float]:
    """Extract a nested metric from a list of packets."""
    values = []
    for p in packets:
        val = p
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                val = None
                break
        if val is not None:
            try:
                values.append(float(val))
            except (TypeError, ValueError):
                pass
    return values


# Metrics to compare between winners and losers
COMPARISON_METRICS = [
    # (display_name, key_path, higher_is_better)
    ("pnl_mean", ("pnl", "mean"), True),
    ("pnl_sharpe", ("pnl", "sharpe_like"), True),
    ("pnl_p05", ("pnl", "p05"), True),
    ("emerald_pnl", ("per_product", "emerald", "mean"), True),
    ("tomato_pnl", ("per_product", "tomato", "mean"), True),
    ("emerald_sharpe", ("per_product", "emerald", "sharpe_like"), True),
    ("tomato_sharpe", ("per_product", "tomato", "sharpe_like"), True),
    ("fill_vs_fair_emerald", ("fill_quality", "mean_fill_vs_fair_emerald"), True),
    ("fill_vs_fair_tomato", ("fill_quality", "mean_fill_vs_fair_tomato"), True),
    ("passive_fill_rate", ("fill_quality", "passive_fill_rate"), None),  # direction ambiguous
    ("taker_fill_count", ("fill_quality", "taker_fill_count"), None),
    ("maker_fill_count", ("fill_quality", "maker_fill_count"), None),
    ("pnl_per_fill", ("efficiency", "pnl_per_fill"), True),
    ("mean_max_drawdown", ("drawdown", "mean_max_drawdown"), False),  # less negative is better
]


def compare_winners_losers(
    winners: list[dict],
    losers: list[dict],
) -> dict[str, Any]:
    """Compare winners vs losers across all comparison metrics.

    Returns a dict with per-metric comparison results including:
    - winner_mean, loser_mean, difference, effect_size
    - whether the difference is meaningful
    """
    results: dict[str, Any] = {}

    for name, key_path, higher_is_better in COMPARISON_METRICS:
        w_vals = _extract_metric(winners, *key_path)
        l_vals = _extract_metric(losers, *key_path)

        w_mean = _safe_mean(w_vals)
        l_mean = _safe_mean(l_vals)
        w_std = _safe_std(w_vals)
        l_std = _safe_std(l_vals)

        diff = w_mean - l_mean if not (math.isnan(w_mean) or math.isnan(l_mean)) else math.nan

        # Effect size: difference normalized by pooled std
        pooled_std = math.sqrt((w_std ** 2 + l_std ** 2) / 2) if (w_std + l_std) > 0 else 1.0
        effect_size = diff / pooled_std if pooled_std > 0 else math.nan

        results[name] = {
            "winner_mean": w_mean,
            "winner_std": w_std,
            "winner_count": len(w_vals),
            "loser_mean": l_mean,
            "loser_std": l_std,
            "loser_count": len(l_vals),
            "difference": diff,
            "effect_size": effect_size,
            "higher_is_better": higher_is_better,
        }

    return results


def compare_family_performance(
    packets: list[dict],
) -> dict[str, dict[str, Any]]:
    """Compare performance across strategy families.

    Returns per-family aggregate stats.
    """
    by_family: dict[str, list[dict]] = {}
    for p in packets:
        fam = p.get("_family", "unknown")
        by_family.setdefault(fam, []).append(p)

    results: dict[str, dict[str, Any]] = {}
    for fam, fam_packets in sorted(by_family.items()):
        pnl_vals = _extract_metric(fam_packets, "pnl", "mean")
        sharpe_vals = _extract_metric(fam_packets, "pnl", "sharpe_like")
        pfr_vals = _extract_metric(fam_packets, "fill_quality", "passive_fill_rate")

        promoted = sum(1 for p in fam_packets if p.get("promote", {}).get("recommended", False))

        results[fam] = {
            "count": len(fam_packets),
            "promoted": promoted,
            "pnl_mean": _safe_mean(pnl_vals),
            "pnl_std": _safe_std(pnl_vals),
            "sharpe_mean": _safe_mean(sharpe_vals),
            "passive_fill_rate_mean": _safe_mean(pfr_vals),
        }

    return results


def run_comparison(
    bank_dir: Path,
    split_method: str = "promote",
) -> dict[str, Any]:
    """Full comparison pipeline: load packets, split, compare.

    Returns dict with all comparison results and metadata.
    """
    packets = load_packets_from_bank(bank_dir)
    if not packets:
        return {
            "packet_count": 0,
            "winners": [],
            "losers": [],
            "metric_comparison": {},
            "family_comparison": {},
            "split_method": split_method,
        }

    winners, losers = split_winners_losers(packets, method=split_method)
    metric_cmp = compare_winners_losers(winners, losers)
    family_cmp = compare_family_performance(packets)

    return {
        "packet_count": len(packets),
        "winner_count": len(winners),
        "loser_count": len(losers),
        "split_method": split_method,
        "winner_ids": [p.get("_case_id", "?") for p in winners],
        "loser_ids": [p.get("_case_id", "?") for p in losers],
        "metric_comparison": metric_cmp,
        "family_comparison": family_cmp,
    }
