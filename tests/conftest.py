"""Shared test fixtures — synthetic backtest output matching Rust CSV conventions."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest


PRODUCTS = ["EMERALDS", "TOMATOES"]
TICKS = 100  # small session for tests
TIMESTAMP_STEP = 100
EMERALD_FAIR = 10000.0
TOMATO_FAIR_START = 5000.0


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create a synthetic backtest output directory with 2 sample sessions."""
    # --- session_summary.csv (3 sessions, comma-delimited) ---
    summary_path = tmp_path / "session_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "session_id", "total_pnl", "emerald_pnl", "tomato_pnl",
            "emerald_position", "tomato_position", "emerald_cash", "tomato_cash",
            "total_slope_per_step", "total_r2",
            "emerald_slope_per_step", "emerald_r2",
            "tomato_slope_per_step", "tomato_r2",
        ])
        writer.writeheader()
        for sid in range(3):
            writer.writerow({
                "session_id": sid,
                "total_pnl": 100.0 + sid * 50,
                "emerald_pnl": 60.0 + sid * 20,
                "tomato_pnl": 40.0 + sid * 30,
                "emerald_position": 5 - sid,
                "tomato_position": -3 + sid * 2,
                "emerald_cash": -(10000 * (5 - sid)),
                "tomato_cash": -(5000 * (-3 + sid * 2)),
                "total_slope_per_step": 0.01 * (sid + 1),
                "total_r2": 0.5 + sid * 0.1,
                "emerald_slope_per_step": 0.005 * (sid + 1),
                "emerald_r2": 0.4 + sid * 0.1,
                "tomato_slope_per_step": 0.005 * (sid + 1),
                "tomato_r2": 0.3 + sid * 0.15,
            })

    # --- run_summary.csv ---
    run_path = tmp_path / "run_summary.csv"
    with run_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "session_id", "day", "total_pnl", "emerald_pnl", "tomato_pnl",
            "total_slope_per_step", "total_r2",
            "emerald_slope_per_step", "emerald_r2",
            "tomato_slope_per_step", "tomato_r2",
        ])
        writer.writeheader()
        for sid in range(3):
            writer.writerow({
                "session_id": sid, "day": -2,
                "total_pnl": 100.0 + sid * 50,
                "emerald_pnl": 60.0 + sid * 20,
                "tomato_pnl": 40.0 + sid * 30,
                "total_slope_per_step": 0.01,
                "total_r2": 0.5,
                "emerald_slope_per_step": 0.005,
                "emerald_r2": 0.4,
                "tomato_slope_per_step": 0.005,
                "tomato_r2": 0.3,
            })

    # --- Sample sessions (2 of 3 get full trace data) ---
    for sid in range(2):
        _write_sample_session(tmp_path, sid)

    return tmp_path


def _write_sample_session(output_dir: Path, session_id: int) -> None:
    """Write trace, price, and trade CSVs for one synthetic session."""
    day = -2
    round_dir = output_dir / "sessions" / f"session_{session_id:05d}" / "round0"
    round_dir.mkdir(parents=True)

    # Random walk for tomato fair value
    import random
    rng = random.Random(42 + session_id)
    tomato_fairs = [TOMATO_FAIR_START]
    for _ in range(TICKS - 1):
        tomato_fairs.append(tomato_fairs[-1] + rng.gauss(0, 1.0))

    # --- Trace CSV (semicolon-delimited) ---
    trace_path = round_dir / f"trace_round_0_day_{day}.csv"
    emerald_pos = 0
    tomato_pos = 0
    emerald_cash = 0.0
    tomato_cash = 0.0
    with trace_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "day", "timestamp", "product", "fair_value", "position", "cash", "mtm_pnl",
        ], delimiter=";")
        writer.writeheader()
        for tick in range(TICKS):
            ts = tick * TIMESTAMP_STEP
            # Simulate simple position accumulation
            if tick % 10 == 5:
                emerald_pos = min(emerald_pos + 1, 10)
                emerald_cash -= EMERALD_FAIR
            if tick % 15 == 7:
                tomato_pos = min(tomato_pos + 1, 10)
                tomato_cash -= tomato_fairs[tick]

            writer.writerow({
                "day": day, "timestamp": ts, "product": "EMERALDS",
                "fair_value": EMERALD_FAIR,
                "position": emerald_pos,
                "cash": emerald_cash,
                "mtm_pnl": emerald_cash + emerald_pos * EMERALD_FAIR,
            })
            writer.writerow({
                "day": day, "timestamp": ts, "product": "TOMATOES",
                "fair_value": tomato_fairs[tick],
                "position": tomato_pos,
                "cash": tomato_cash,
                "mtm_pnl": tomato_cash + tomato_pos * tomato_fairs[tick],
            })

    # --- Price CSV (semicolon-delimited) ---
    price_path = round_dir / f"prices_round_0_day_{day}.csv"
    with price_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "day", "timestamp", "product",
            "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2",
            "bid_price_3", "bid_volume_3",
            "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2",
            "ask_price_3", "ask_volume_3",
            "mid_price", "profit_and_loss",
        ], delimiter=";")
        writer.writeheader()
        for tick in range(TICKS):
            ts = tick * TIMESTAMP_STEP
            # EMERALDS: tight spread around 10000
            writer.writerow({
                "day": day, "timestamp": ts, "product": "EMERALDS",
                "bid_price_1": 9998, "bid_volume_1": 20,
                "bid_price_2": 9997, "bid_volume_2": 15,
                "bid_price_3": 9996, "bid_volume_3": 10,
                "ask_price_1": 10002, "ask_volume_1": 20,
                "ask_price_2": 10003, "ask_volume_2": 15,
                "ask_price_3": 10004, "ask_volume_3": 10,
                "mid_price": 10000.0,
                "profit_and_loss": 0.0,
            })
            # TOMATOES: spread around fair value
            tf = tomato_fairs[tick]
            writer.writerow({
                "day": day, "timestamp": ts, "product": "TOMATOES",
                "bid_price_1": int(tf) - 2, "bid_volume_1": 15,
                "bid_price_2": int(tf) - 3, "bid_volume_2": 10,
                "bid_price_3": int(tf) - 4, "bid_volume_3": 5,
                "ask_price_1": int(tf) + 2, "ask_volume_1": 15,
                "ask_price_2": int(tf) + 3, "ask_volume_2": 10,
                "ask_price_3": int(tf) + 4, "ask_volume_3": 5,
                "mid_price": tf,
                "profit_and_loss": 0.0,
            })

    # --- Trade CSV (semicolon-delimited) ---
    trade_path = round_dir / f"trades_round_0_day_{day}.csv"
    with trade_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "buyer", "seller", "symbol", "currency", "price", "quantity",
        ], delimiter=";")
        writer.writeheader()

        for tick in range(TICKS):
            ts = tick * TIMESTAMP_STEP
            # Strategy taker buy of EMERALDS every 10 ticks
            if tick % 10 == 5:
                writer.writerow({
                    "timestamp": ts,
                    "buyer": "SUBMISSION", "seller": "BOT",
                    "symbol": "EMERALDS", "currency": "XIRECS",
                    "price": 10002.0, "quantity": 1,
                })
            # Strategy maker sell of TOMATOES every 15 ticks (bot taker buys from us)
            if tick % 15 == 7:
                writer.writerow({
                    "timestamp": ts,
                    "buyer": "BOT_TAKER", "seller": "SUBMISSION",
                    "symbol": "TOMATOES", "currency": "XIRECS",
                    "price": int(tomato_fairs[tick]) + 1, "quantity": 1,
                })
            # Strategy maker buy of EMERALDS every 20 ticks (bot taker sells to us)
            if tick % 20 == 3:
                writer.writerow({
                    "timestamp": ts,
                    "buyer": "SUBMISSION", "seller": "BOT_TAKER",
                    "symbol": "EMERALDS", "currency": "XIRECS",
                    "price": 9999.0, "quantity": 2,
                })
            # Bot-vs-bot trade (no strategy involvement)
            if tick % 8 == 0:
                writer.writerow({
                    "timestamp": ts,
                    "buyer": "BOT_TAKER", "seller": "BOT_MAKER",
                    "symbol": "EMERALDS", "currency": "XIRECS",
                    "price": 10001.0, "quantity": 3,
                })

    return


@pytest.fixture
def dashboard_dict() -> dict:
    """Minimal dashboard dict matching MonteCarloDashboard shape."""
    def dist(mean: float, std: float, n: int = 100) -> dict:
        sharpe = mean / std if std > 0 else 0.0
        return {
            "count": n, "mean": mean, "std": std,
            "min": mean - 3 * std, "max": mean + 3 * std,
            "p01": mean - 2.3 * std, "p05": mean - 1.65 * std,
            "p10": mean - 1.28 * std, "p25": mean - 0.67 * std,
            "p50": mean, "p75": mean + 0.67 * std,
            "p90": mean + 1.28 * std, "p95": mean + 1.65 * std,
            "p99": mean + 2.3 * std,
            "positiveRate": 0.72, "negativeRate": 0.28, "zeroRate": 0.0,
            "var95": mean - 1.65 * std, "cvar95": mean - 2.0 * std,
            "var99": mean - 2.3 * std, "cvar99": mean - 2.5 * std,
            "meanConfidenceLow95": mean - 1.96 * std / math.sqrt(n),
            "meanConfidenceHigh95": mean + 1.96 * std / math.sqrt(n),
            "sharpeLike": sharpe,
            "sortinoLike": sharpe * 1.2,
            "skewness": 0.1,
        }

    sessions = [
        {
            "sessionId": i,
            "totalPnl": 100.0 + i * 50,
            "emeraldPnl": 60.0 + i * 20,
            "tomatoPnl": 40.0 + i * 30,
            "emeraldPosition": 5 - i % 3,
            "tomatoPosition": -3 + (i % 3) * 2,
            "emeraldCash": -(10000 * (5 - i % 3)),
            "tomatoCash": -(5000 * (-3 + (i % 3) * 2)),
            "totalSlopePerStep": 0.01,
            "totalR2": 0.5,
            "emeraldSlopePerStep": 0.005,
            "emeraldR2": 0.4,
            "tomatoSlopePerStep": 0.005,
            "tomatoR2": 0.3,
        }
        for i in range(100)
    ]

    return {
        "kind": "monte_carlo_dashboard",
        "meta": {
            "algorithmPath": "/test/strategy.py",
            "sessionCount": 100,
            "fvMode": "simulate",
            "tradeMode": "simulate",
            "tomatoSupport": "quarter",
            "seed": 42,
            "sampleSessions": 2,
        },
        "overall": {
            "totalPnl": dist(150.0, 80.0),
            "emeraldPnl": dist(80.0, 40.0),
            "tomatoPnl": dist(70.0, 60.0),
            "emeraldTomatoCorrelation": 0.15,
        },
        "trendFits": {
            "total": {
                "profitability": dist(0.01, 0.005),
                "stability": dist(0.5, 0.2),
            },
        },
        "scatterFit": {
            "slope": 0.3, "intercept": 20.0, "r2": 0.15,
            "correlation": 0.15, "line": [], "diagnosis": "weak positive",
        },
        "normalFits": {
            "totalPnl": {"mean": 150.0, "std": 80.0, "r2": 0.95, "line": []},
            "emeraldPnl": {"mean": 80.0, "std": 40.0, "r2": 0.97, "line": []},
            "tomatoPnl": {"mean": 70.0, "std": 60.0, "r2": 0.90, "line": []},
        },
        "generatorModel": {},
        "products": {},
        "histograms": {},
        "sessions": sessions,
        "runs": [],
        "topSessions": sessions[-5:],
        "bottomSessions": sessions[:5],
        "samplePaths": [],
    }
