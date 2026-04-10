"""Research Packet V1 — assembles Packet Short and Packet Full.

Consumes:
  - dashboard dict (from upstream build_dashboard or dashboard.json)
  - event ledger (from engine/event_ledger.py)
  - fill decomposition (from analytics/fill_decomposition.py)
  - regime analysis (from analytics/regime_analysis.py)

Produces:
  - packet_short: compact dict for the brain's first read
  - packet_full: complete structured dict stored in memory

Provenance rules from Step 2.5 are enforced throughout.
"""
from __future__ import annotations

import hashlib
import math
import statistics
from pathlib import Path
from typing import Any, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Confidence rating (Step 2.5 §3.3)
# ---------------------------------------------------------------------------

def compute_confidence(
    session_count: int,
    sample_count: int,
    pnl_cv: float,
) -> tuple[str, str]:
    """Return (rating, reason) per Step 2.5 trust rules.

    HIGH:   N >= 200, sample >= 20, CV < 2.0
    MEDIUM: N >= 50,  sample >= 5,  CV < 5.0
    LOW:    otherwise
    """
    if session_count >= 200 and sample_count >= 20 and pnl_cv < 2.0:
        return "HIGH", (
            f"{session_count} sessions, {sample_count} sample sessions, "
            f"CV={pnl_cv:.2f}"
        )
    if session_count >= 50 and sample_count >= 5 and pnl_cv < 5.0:
        return "MEDIUM", (
            f"{session_count} sessions, {sample_count} sample sessions, "
            f"CV={pnl_cv:.2f}"
        )
    return "LOW", (
        f"Only {session_count} sessions / {sample_count} sample sessions, "
        f"CV={pnl_cv:.2f}"
    )


# ---------------------------------------------------------------------------
# Drawdown analysis (from sample-session traces)
# ---------------------------------------------------------------------------

def compute_drawdown(mtm_series: list[float]) -> dict[str, float]:
    """Compute max drawdown and recovery ticks from a mtm_pnl time series."""
    if len(mtm_series) < 2:
        return {
            "max_drawdown": 0.0,
            "max_drawdown_tick": 0,
            "recovery_ticks": 0.0,
            "recovered": True,
        }

    running_peak = mtm_series[0]
    peak_tick = 0
    max_dd = 0.0
    max_dd_tick = 0
    max_dd_peak_tick = 0

    for i, val in enumerate(mtm_series):
        if val > running_peak:
            running_peak = val
            peak_tick = i
        dd = val - running_peak
        if dd < max_dd:
            max_dd = dd
            max_dd_tick = i
            max_dd_peak_tick = peak_tick

    # Find recovery point
    peak_value = mtm_series[max_dd_peak_tick]
    recovered = False
    recovery_ticks = float("nan")
    for i in range(max_dd_tick, len(mtm_series)):
        if mtm_series[i] >= peak_value:
            recovery_ticks = float(i - max_dd_tick)
            recovered = True
            break

    return {
        "max_drawdown": max_dd,
        "max_drawdown_tick": max_dd_tick,
        "recovery_ticks": recovery_ticks if recovered else float("nan"),
        "recovered": recovered,
    }


def aggregate_drawdowns(
    session_ledgers: dict[int, dict[str, pd.DataFrame]],
) -> dict[str, Any]:
    """Compute drawdown stats across sample sessions (total P&L)."""
    drawdowns: list[float] = []
    recovery_ticks_list: list[float] = []
    unrecovered_count = 0

    for session_id, ledger in session_ledgers.items():
        traces = ledger["traces"]
        if traces.empty:
            continue

        # Build total mtm_pnl series across products
        products = traces["product"].unique()
        by_tick: dict[tuple[int, int], float] = {}
        for _, row in traces.iterrows():
            key = (int(row["day"]), int(row["timestamp"]))
            by_tick[key] = by_tick.get(key, 0.0) + row["mtm_pnl"]

        sorted_keys = sorted(by_tick.keys())
        total_pnl_series = [by_tick[k] for k in sorted_keys]

        dd = compute_drawdown(total_pnl_series)
        drawdowns.append(dd["max_drawdown"])
        if dd["recovered"]:
            recovery_ticks_list.append(dd["recovery_ticks"])
        else:
            unrecovered_count += 1

    if not drawdowns:
        return {
            "mean_max_drawdown": 0.0,
            "p95_max_drawdown": 0.0,
            "mean_recovery_ticks": 0.0,
            "unrecovered_rate": 0.0,
            "session_count": 0,
        }

    sorted_dd = sorted(drawdowns)
    p95_idx = max(0, int(0.05 * (len(sorted_dd) - 1)))  # 5th percentile of drawdown (most negative)

    return {
        "mean_max_drawdown": statistics.fmean(drawdowns),
        "p95_max_drawdown": sorted_dd[p95_idx],
        "mean_recovery_ticks": (
            statistics.fmean(recovery_ticks_list) if recovery_ticks_list else float("nan")
        ),
        "unrecovered_rate": unrecovered_count / len(drawdowns),
        "session_count": len(drawdowns),
    }


# ---------------------------------------------------------------------------
# Inventory analysis
# ---------------------------------------------------------------------------

def compute_inventory_drag(traces: pd.DataFrame) -> dict[str, float]:
    """Compute inventory drag per product from a single session's traces.

    inventory_drag = sum(position_t * (fair_{t+1} - fair_t))
    """
    result: dict[str, float] = {}
    for product, group in traces.groupby("product"):
        group = group.sort_values(["day", "timestamp"]).reset_index(drop=True)
        positions = group["position"].tolist()
        fairs = group["fair_value"].tolist()
        if len(positions) < 2:
            result[product] = 0.0
            continue
        drag = sum(
            positions[i] * (fairs[i + 1] - fairs[i])
            for i in range(len(positions) - 1)
        )
        result[product] = drag
    return result


def compute_mean_reversion_speed(traces: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Fit AR(1) on |position| per product.

    |pos_{t+1}| = alpha * |pos_t| + epsilon
    mean_reversion_speed = 1 - alpha
    half_life = -ln(2) / ln(alpha) if 0 < alpha < 1
    """
    result: dict[str, dict[str, float]] = {}
    for product, group in traces.groupby("product"):
        group = group.sort_values(["day", "timestamp"]).reset_index(drop=True)
        abs_pos = [abs(int(p)) for p in group["position"].tolist()]
        if len(abs_pos) < 10:
            result[product] = {"alpha": 1.0, "speed": 0.0, "half_life": float("inf"), "r2": 0.0}
            continue

        x = abs_pos[:-1]
        y = abs_pos[1:]
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xx = sum(xi * xi for xi in x)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        denom = n * sum_xx - sum_x * sum_x
        if abs(denom) < 1e-12:
            result[product] = {"alpha": 1.0, "speed": 0.0, "half_life": float("inf"), "r2": 0.0}
            continue

        alpha = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - alpha * sum_x) / n

        # R-squared
        y_mean = sum_y / n
        ss_res = sum((yi - alpha * xi - intercept) ** 2 for xi, yi in zip(x, y))
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        speed = 1.0 - alpha
        half_life = -math.log(2) / math.log(alpha) if 0 < alpha < 1 else float("inf")

        result[product] = {"alpha": alpha, "speed": speed, "half_life": half_life, "r2": r2}

    return result


def aggregate_inventory_analysis(
    session_ledgers: dict[int, dict[str, pd.DataFrame]],
    session_summaries: pd.DataFrame,
) -> dict[str, Any]:
    """Aggregate inventory metrics across sessions."""
    # End position stats from ALL sessions (session_summary.csv)
    end_pos: dict[str, dict[str, float]] = {}
    if not session_summaries.empty:
        for product, col in [("EMERALDS", "emerald_position"), ("TOMATOES", "tomato_position")]:
            vals = session_summaries[col].tolist()
            abs_vals = [abs(v) for v in vals]
            end_pos[product] = {
                "mean": statistics.fmean(vals),
                "mean_abs": statistics.fmean(abs_vals),
                "std": statistics.stdev(vals) if len(vals) >= 2 else 0.0,
            }

    # Inventory drag and mean reversion from SAMPLE sessions
    drags: dict[str, list[float]] = {}
    speeds: dict[str, list[dict[str, float]]] = {}
    for session_id, ledger in session_ledgers.items():
        traces = ledger["traces"]
        if traces.empty:
            continue
        drag = compute_inventory_drag(traces)
        for product, val in drag.items():
            drags.setdefault(product, []).append(val)
        speed = compute_mean_reversion_speed(traces)
        for product, val in speed.items():
            speeds.setdefault(product, []).append(val)

    drag_summary: dict[str, float] = {}
    for product, vals in drags.items():
        drag_summary[product] = statistics.fmean(vals) if vals else 0.0

    speed_summary: dict[str, dict[str, float]] = {}
    for product, vals in speeds.items():
        alphas = [v["alpha"] for v in vals]
        speed_summary[product] = {
            "mean_alpha": statistics.fmean(alphas) if alphas else 1.0,
            "mean_half_life": statistics.fmean(
                [v["half_life"] for v in vals if math.isfinite(v["half_life"])]
            ) if any(math.isfinite(v["half_life"]) for v in vals) else float("inf"),
            "mean_r2": statistics.fmean([v["r2"] for v in vals]) if vals else 0.0,
        }

    return {
        "end_position": end_pos,
        "inventory_drag": drag_summary,
        "mean_reversion": speed_summary,
        "provenance": {
            "end_position_scope": "all",
            "drag_and_reversion_scope": "sample",
            "sample_count": len(session_ledgers),
        },
    }


# ---------------------------------------------------------------------------
# P&L concentration (all sessions)
# ---------------------------------------------------------------------------

def compute_pnl_concentration(pnl_values: list[float]) -> dict[str, float]:
    """Compute concentration metrics from session P&L values."""
    if not pnl_values:
        return {"top_10_pct_share": 0.0, "gini": 0.0}

    abs_pnl = [abs(v) for v in pnl_values]
    total_abs = sum(abs_pnl)
    if total_abs == 0:
        return {"top_10_pct_share": 0.0, "gini": 0.0}

    sorted_abs = sorted(abs_pnl, reverse=True)
    top_k = max(1, len(sorted_abs) // 10)
    top_10_share = sum(sorted_abs[:top_k]) / total_abs

    # Gini coefficient
    sorted_asc = sorted(abs_pnl)
    n = len(sorted_asc)
    numerator = sum((2 * (i + 1) - n - 1) * sorted_asc[i] for i in range(n))
    gini = numerator / (n * sum(sorted_asc)) if sum(sorted_asc) > 0 else 0.0

    return {"top_10_pct_share": top_10_share, "gini": gini}


# ---------------------------------------------------------------------------
# End-of-session PnL at risk
# ---------------------------------------------------------------------------

def compute_pnl_at_risk(
    session_summaries: pd.DataFrame,
    mean_spread: dict[str, float],
) -> dict[str, float]:
    """PnL at risk from final positions.

    pnl_at_risk = mean(|final_position|) * half_spread
    """
    result: dict[str, float] = {}
    for product, col in [("EMERALDS", "emerald_position"), ("TOMATOES", "tomato_position")]:
        if session_summaries.empty:
            result[product] = 0.0
            continue
        mean_abs_pos = statistics.fmean([abs(v) for v in session_summaries[col].tolist()])
        half_spread = mean_spread.get(product, 2.0) / 2.0  # default 1.0 half-spread
        result[product] = mean_abs_pos * half_spread
    return result


# ---------------------------------------------------------------------------
# Kill / Promote recommendations (Step 2.5 §1.12, §1.13)
# ---------------------------------------------------------------------------

def compute_kill(dist: dict[str, Any], confidence: str) -> dict[str, Any]:
    """Kill recommendation: should the strategy be abandoned?"""
    mean_ci_high = dist.get("meanConfidenceHigh95", dist.get("mean_confidence_high_95", 0.0))
    positive_rate = dist.get("positiveRate", dist.get("positive_rate", 0.5))
    mean_val = dist.get("mean", 0.0)
    std_val = dist.get("std", 1.0)

    kill = (
        mean_ci_high < 0
        and positive_rate < 0.35
        and confidence in ("HIGH", "MEDIUM")
    )
    strength = abs(mean_val / std_val) if std_val > 0 else 0.0

    if kill:
        reason = (
            f"95% CI upper bound ({mean_ci_high:.1f}) is negative, "
            f"only {positive_rate:.0%} sessions profitable."
        )
    elif confidence == "LOW":
        reason = "Insufficient evidence to recommend kill. Run more sessions."
    else:
        reason = "Strategy does not meet kill criteria."

    return {"recommended": kill, "strength": round(strength, 4), "reason": reason}


def compute_promote(
    dist: dict[str, Any],
    confidence: str,
    drawdown: dict[str, Any],
) -> dict[str, Any]:
    """Promote recommendation: is the strategy strong enough to advance?"""
    mean_ci_low = dist.get("meanConfidenceLow95", dist.get("mean_confidence_low_95", 0.0))
    positive_rate = dist.get("positiveRate", dist.get("positive_rate", 0.5))
    sharpe = dist.get("sharpeLike", dist.get("sharpe_like", 0.0))
    mean_val = dist.get("mean", 0.0)
    mean_dd = drawdown.get("mean_max_drawdown", 0.0)

    # Tolerable drawdown: |mean_dd| < 3 * mean_pnl (if mean_pnl > 0)
    dd_ok = abs(mean_dd) < 3 * mean_val if mean_val > 0 else False

    promote = (
        mean_ci_low > 0
        and positive_rate > 0.60
        and sharpe > 0.3
        and dd_ok
        and confidence == "HIGH"
    )
    strength = round(sharpe, 4)

    if promote:
        reason = (
            f"95% CI lower bound ({mean_ci_low:.1f}) is positive, "
            f"Sharpe-like={sharpe:.2f}, {positive_rate:.0%} profitable."
        )
    elif confidence != "HIGH":
        reason = f"Confidence is {confidence} — need HIGH for promotion. Run more sessions."
    else:
        parts = []
        if mean_ci_low <= 0:
            parts.append(f"CI lower bound ({mean_ci_low:.1f}) includes zero")
        if positive_rate <= 0.60:
            parts.append(f"positive rate {positive_rate:.0%} <= 60%")
        if sharpe <= 0.3:
            parts.append(f"Sharpe-like {sharpe:.2f} <= 0.3")
        if not dd_ok:
            parts.append("drawdown exceeds 3x mean P&L")
        reason = "Not promoted: " + "; ".join(parts) + "."

    return {"recommended": promote, "strength": strength, "reason": reason}


# ---------------------------------------------------------------------------
# Diagnosis summary (natural language)
# ---------------------------------------------------------------------------

def generate_diagnosis(
    pnl_dist: dict[str, Any],
    fill_decomp: dict[str, Any],
    inventory: dict[str, Any],
    drawdown: dict[str, Any],
    confidence: str,
) -> str:
    """Generate a 2-3 sentence diagnosis for Packet Short."""
    parts = []
    mean = pnl_dist.get("mean", 0.0)
    std = pnl_dist.get("std", 1.0)
    sharpe = pnl_dist.get("sharpeLike", pnl_dist.get("sharpe_like", 0.0))
    pos_rate = pnl_dist.get("positiveRate", pnl_dist.get("positive_rate", 0.5))

    if mean > 0:
        parts.append(
            f"Strategy is profitable (mean={mean:.1f}, Sharpe-like={sharpe:.2f}, "
            f"{pos_rate:.0%} winning sessions)."
        )
    elif mean == 0:
        parts.append("Strategy breaks even on average.")
    else:
        parts.append(
            f"Strategy is unprofitable (mean={mean:.1f}, Sharpe-like={sharpe:.2f}, "
            f"{pos_rate:.0%} winning sessions)."
        )

    # Fill quality insight
    volumes = fill_decomp.get("volumes", {})
    passive_rate = volumes.get("passive_fill_rate", 0.0)
    if passive_rate > 0.5:
        parts.append(f"Predominantly a maker strategy ({passive_rate:.0%} passive fills).")
    elif passive_rate > 0:
        parts.append(f"Mixed maker/taker ({passive_rate:.0%} passive fills).")
    elif volumes.get("taker_fill_count", 0) > 0:
        parts.append("Pure taker strategy — no passive fills observed.")

    # Drawdown concern
    mean_dd = drawdown.get("mean_max_drawdown", 0.0)
    if mean > 0 and abs(mean_dd) > 2 * mean:
        parts.append(
            f"Drawdowns are large relative to mean P&L "
            f"(mean max DD={mean_dd:.1f} vs mean PnL={mean:.1f})."
        )

    if confidence == "LOW":
        parts.append("Confidence is LOW — these conclusions are preliminary.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Mean spread estimation from sample session prices
# ---------------------------------------------------------------------------

def estimate_mean_spread(
    session_ledgers: dict[int, dict[str, pd.DataFrame]],
) -> dict[str, float]:
    """Compute mean bid-ask spread per product from sample session prices."""
    spreads: dict[str, list[float]] = {}
    for session_id, ledger in session_ledgers.items():
        prices = ledger["prices"]
        if prices.empty:
            continue
        for _, row in prices.iterrows():
            bid1 = row["bid1"]
            ask1 = row["ask1"]
            if math.isnan(bid1) or math.isnan(ask1):
                continue
            product = row["product"]
            spreads.setdefault(product, []).append(ask1 - bid1)

    result: dict[str, float] = {}
    for product, vals in spreads.items():
        result[product] = statistics.fmean(vals) if vals else 2.0
    return result


# ---------------------------------------------------------------------------
# Packet assembly
# ---------------------------------------------------------------------------

def _safe_get(d: dict, *keys: str, default: Any = 0.0) -> Any:
    """Nested dict get with default."""
    current = d
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current


def _normalize_dist(dist: dict[str, Any]) -> dict[str, Any]:
    """Normalize dashboard distribution keys to snake_case for internal use."""
    return {
        "mean": dist.get("mean", 0.0),
        "std": dist.get("std", 0.0),
        "sharpe_like": dist.get("sharpeLike", dist.get("sharpe_like", 0.0)),
        "sortino_like": dist.get("sortinoLike", dist.get("sortino_like", 0.0)),
        "p05": dist.get("p05", 0.0),
        "p50": dist.get("p50", 0.0),
        "p95": dist.get("p95", 0.0),
        "positive_rate": dist.get("positiveRate", dist.get("positive_rate", 0.0)),
        "negative_rate": dist.get("negativeRate", dist.get("negative_rate", 0.0)),
        "skewness": dist.get("skewness", 0.0),
        "min": dist.get("min", 0.0),
        "max": dist.get("max", 0.0),
        "p01": dist.get("p01", 0.0),
        "p10": dist.get("p10", 0.0),
        "p25": dist.get("p25", 0.0),
        "p75": dist.get("p75", 0.0),
        "p90": dist.get("p90", 0.0),
        "p99": dist.get("p99", 0.0),
        "var95": dist.get("var95", 0.0),
        "cvar95": dist.get("cvar95", 0.0),
        "var99": dist.get("var99", 0.0),
        "cvar99": dist.get("cvar99", 0.0),
        "mean_confidence_low_95": dist.get("meanConfidenceLow95", dist.get("mean_confidence_low_95", 0.0)),
        "mean_confidence_high_95": dist.get("meanConfidenceHigh95", dist.get("mean_confidence_high_95", 0.0)),
        "count": dist.get("count", 0),
    }


def build_packet(
    dashboard: dict[str, Any],
    event_ledger: dict[str, Any],
    fill_decomp: dict[str, Any],
    regime_summary: dict[str, Any],
    strategy_path: str = "",
) -> dict[str, Any]:
    """Build Packet Short and Packet Full from all inputs.

    Args:
        dashboard: The upstream dashboard dict (MonteCarloDashboard shape).
        event_ledger: Output of build_event_ledger().
        fill_decomp: Output of aggregate_fill_decomposition().
        regime_summary: Output of summarize_regimes().
        strategy_path: Path to the strategy file.

    Returns:
        Dict with keys "short" and "full".
    """
    meta = dashboard.get("meta", {})
    overall = dashboard.get("overall", {})
    session_ledgers = event_ledger.get("session_ledgers", {})
    session_summaries_list = dashboard.get("sessions", [])

    # Build session_summaries DataFrame from dashboard sessions
    session_summaries = pd.DataFrame(session_summaries_list) if session_summaries_list else pd.DataFrame()
    if not session_summaries.empty and "totalPnl" in session_summaries.columns:
        session_summaries = session_summaries.rename(columns={
            "totalPnl": "total_pnl",
            "emeraldPnl": "emerald_pnl",
            "tomatoPnl": "tomato_pnl",
            "emeraldPosition": "emerald_position",
            "tomatoPosition": "tomato_position",
            "emeraldCash": "emerald_cash",
            "tomatoCash": "tomato_cash",
        })

    # --- Extract distributions from dashboard ---
    total_dist = _normalize_dist(overall.get("totalPnl", {}))
    emerald_dist = _normalize_dist(overall.get("emeraldPnl", {}))
    tomato_dist = _normalize_dist(overall.get("tomatoPnl", {}))

    session_count = int(meta.get("sessionCount", total_dist.get("count", 0)))
    sample_count = event_ledger.get("provenance", {}).get("sample_count", 0)

    # Coefficient of variation
    pnl_cv = abs(total_dist["std"] / total_dist["mean"]) if total_dist["mean"] != 0 else 999.0

    # --- Confidence ---
    confidence, confidence_reason = compute_confidence(session_count, sample_count, pnl_cv)

    # --- Warnings ---
    warnings: list[str] = []
    if event_ledger.get("provenance", {}).get("warning"):
        warnings.append(event_ledger["provenance"]["warning"])
    if fill_decomp.get("provenance", {}).get("warning"):
        warnings.append(fill_decomp["provenance"]["warning"])
    if regime_summary.get("provenance", {}).get("warning"):
        warnings.append(regime_summary["provenance"]["warning"])
    if session_count < 100:
        warnings.append(
            f"P&L distribution from only {session_count} sessions — "
            "percentile estimates (especially p01/p99) are unreliable."
        )

    # --- Trend ---
    trend_fits = dashboard.get("trendFits", dashboard.get("trend_fits", {}))
    total_trend = trend_fits.get("total", {})
    total_profitability = total_trend.get("profitability", {})
    mean_slope = total_profitability.get("mean", 0.0)
    mean_r2_values = trend_fits.get("total", {}).get("stability", {})
    mean_r2 = mean_r2_values.get("mean", 0.0)
    r2_warning = (
        f"Low mean R2 ({mean_r2:.3f}) — P&L trajectory is highly non-linear."
        if mean_r2 < 0.1
        else None
    )

    # --- Fill quality ---
    volumes = fill_decomp.get("volumes", {})
    fvf = fill_decomp.get("fill_vs_fair", {})

    fill_quality_short: Optional[dict[str, Any]] = None
    if volumes.get("taker_fill_count", 0) + volumes.get("maker_fill_count", 0) > 0:
        fill_quality_short = {
            "mean_fill_vs_fair_emerald": _safe_get(fvf, "EMERALDS", "mean"),
            "mean_fill_vs_fair_tomato": _safe_get(fvf, "TOMATOES", "mean"),
            "passive_fill_rate": volumes.get("passive_fill_rate", 0.0),
            "taker_fill_count": volumes.get("taker_fill_count", 0),
            "maker_fill_count": volumes.get("maker_fill_count", 0),
        }

    # --- Inventory ---
    inventory = aggregate_inventory_analysis(session_ledgers, session_summaries)
    mean_spread = estimate_mean_spread(session_ledgers)

    inventory_short: dict[str, Any] = {
        "mean_end_position_emerald": _safe_get(inventory, "end_position", "EMERALDS", "mean"),
        "mean_end_position_tomato": _safe_get(inventory, "end_position", "TOMATOES", "mean"),
        "inventory_drag_tomato": inventory.get("inventory_drag", {}).get("TOMATOES"),
        "mean_reversion_half_life_tomato": (
            inventory.get("mean_reversion", {}).get("TOMATOES", {}).get("mean_half_life")
        ),
    }

    # --- Drawdown ---
    drawdown = aggregate_drawdowns(session_ledgers)
    drawdown_short: Optional[dict[str, Any]] = None
    if drawdown["session_count"] > 0:
        drawdown_short = {
            "mean_max_drawdown": drawdown["mean_max_drawdown"],
            "p95_max_drawdown": drawdown["p95_max_drawdown"],
            "mean_recovery_ticks": drawdown["mean_recovery_ticks"],
            "unrecovered_rate": drawdown["unrecovered_rate"],
        }

    # --- Concentration (all sessions) ---
    pnl_values = [s.get("totalPnl", s.get("total_pnl", 0.0)) for s in session_summaries_list]
    concentration = compute_pnl_concentration(pnl_values)

    # --- PnL at risk ---
    pnl_at_risk = compute_pnl_at_risk(session_summaries, mean_spread)

    # --- Kill / Promote ---
    kill = compute_kill(overall.get("totalPnl", {}), confidence)
    promote = compute_promote(overall.get("totalPnl", {}), confidence, drawdown)

    # --- Correlation ---
    corr = overall.get("emeraldTomatoCorrelation", overall.get("emerald_tomato_correlation", 0.0))

    # --- Diagnosis ---
    diagnosis = generate_diagnosis(
        overall.get("totalPnl", {}), fill_decomp, inventory, drawdown, confidence
    )

    # --- Candidate identity ---
    candidate = {
        "strategy_path": strategy_path or meta.get("algorithmPath", ""),
        "session_count": session_count,
        "seed": meta.get("seed", 0),
        "fv_mode": meta.get("fvMode", ""),
        "trade_mode": meta.get("tradeMode", ""),
        "tomato_support": meta.get("tomatoSupport", ""),
        "sample_sessions": sample_count,
    }

    # --- Build candidate_id ---
    # Prefer content hash of actual strategy file for true code identity.
    # Fall back to metadata hash if the file is missing or no path given.
    strategy_file = Path(candidate["strategy_path"]) if candidate["strategy_path"] else None
    if strategy_file and strategy_file.is_file():
        file_bytes = strategy_file.read_bytes()
        candidate_id = hashlib.sha256(file_bytes).hexdigest()[:16]
        candidate_id_method = "content_hash"
    else:
        id_string = f"{candidate['strategy_path']}:{candidate['seed']}:{session_count}"
        candidate_id = hashlib.sha256(id_string.encode()).hexdigest()[:16]
        candidate_id_method = "metadata_fallback"

    # =======================================================================
    # PACKET SHORT
    # =======================================================================
    packet_short = {
        "candidate_id": candidate_id,
        "candidate_id_method": candidate_id_method,
        "candidate": candidate,
        "confidence": confidence,
        "confidence_reason": confidence_reason,
        "warnings": warnings,
        "pnl": {
            "mean": total_dist["mean"],
            "std": total_dist["std"],
            "sharpe_like": total_dist["sharpe_like"],
            "sortino_like": total_dist["sortino_like"],
            "p05": total_dist["p05"],
            "p50": total_dist["p50"],
            "p95": total_dist["p95"],
            "positive_rate": total_dist["positive_rate"],
            "skewness": total_dist["skewness"],
        },
        "per_product": {
            "emerald": {
                "mean": emerald_dist["mean"],
                "std": emerald_dist["std"],
                "sharpe_like": emerald_dist["sharpe_like"],
            },
            "tomato": {
                "mean": tomato_dist["mean"],
                "std": tomato_dist["std"],
                "sharpe_like": tomato_dist["sharpe_like"],
            },
        },
        "correlation": corr,
        "trend": {
            "mean_slope_per_step": mean_slope,
            "mean_r2": mean_r2,
            "r2_warning": r2_warning,
        },
        "fill_quality": fill_quality_short,
        "inventory": inventory_short,
        "drawdown": drawdown_short,
        "concentration": concentration,
        "end_of_session_risk": pnl_at_risk,
        "kill": kill,
        "promote": promote,
        "diagnosis": diagnosis,
    }

    # =======================================================================
    # PACKET FULL
    # =======================================================================
    packet_full = {
        **packet_short,
        "pnl_full": {
            "total": total_dist,
            "emerald": emerald_dist,
            "tomato": tomato_dist,
        },
        "per_product_full": dashboard.get("products", {}),
        "trend_fits": trend_fits,
        "aggregate_trend_fits": dashboard.get("aggregateTrendFits", {}),
        "fill_quality_full": fill_decomp,
        "inventory_full": inventory,
        "drawdown_full": drawdown,
        "regime_analysis": regime_summary,
        "correlation_full": {
            "emerald_tomato": corr,
            "scatter_fit": dashboard.get("scatterFit", {}),
        },
        "histograms": dashboard.get("histograms", {}),
        "normal_fits": dashboard.get("normalFits", {}),
        "generator_model": dashboard.get("generatorModel", {}),
        "sessions": session_summaries_list,
        "top_sessions": dashboard.get("topSessions", []),
        "bottom_sessions": dashboard.get("bottomSessions", []),
    }

    return {"short": packet_short, "full": packet_full}
