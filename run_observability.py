#!/usr/bin/env python3
"""End-to-end Observability V1 pipeline.

Usage:
    python run_observability.py <backtest_output_dir> [--dashboard dashboard.json]

Reads the backtest outputs (session_summary.csv, sample session CSVs, and
optionally a pre-built dashboard.json), builds the event ledger, runs all
analytics, assembles the research packet, and stores it.

If no dashboard.json is provided, the pipeline builds one from the CSV outputs
using the upstream build_dashboard function.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the upstream backtester is importable
ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "_upstream_backtest" / "backtester"
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(ROOT))

from analytics.fill_decomposition import aggregate_fill_decomposition
from analytics.regime_analysis import summarize_regimes
from engine.event_ledger import build_event_ledger
from engine.research_packet import build_packet
from memory.store import PacketStore


def load_or_build_dashboard(output_dir: Path, dashboard_path: Path | None) -> dict:
    """Load dashboard from JSON file or build from CSV outputs."""
    if dashboard_path and dashboard_path.exists():
        with dashboard_path.open() as f:
            return json.load(f)

    # Try to find dashboard.json in the output directory
    default_path = output_dir / "dashboard.json"
    if default_path.exists():
        with default_path.open() as f:
            return json.load(f)

    # Derive true session count from session_summary.csv (one row per MC session)
    summary_csv = output_dir / "session_summary.csv"
    if summary_csv.exists():
        import csv
        with summary_csv.open() as f:
            session_count = sum(1 for _ in csv.reader(f)) - 1  # subtract header
    else:
        # Last resort: count sample-session dirs (known undercount)
        session_count = len(list((output_dir / "sessions").iterdir())) if (output_dir / "sessions").exists() else 0
        print(
            f"Warning: session_summary.csv not found. "
            f"Using sample-session directory count ({session_count}) as session_count — "
            "this may undercount the true Monte Carlo session count.",
            file=sys.stderr,
        )

    # Build from upstream if available
    try:
        from prosperity3bt.monte_carlo import build_dashboard
        strategy_path = output_dir / "strategy.py"  # placeholder
        return build_dashboard(output_dir, strategy_path, session_count, {})
    except (ImportError, Exception) as exc:
        print(f"Warning: could not build dashboard: {exc}", file=sys.stderr)
        print("Proceeding with minimal dashboard stub.", file=sys.stderr)
        return {
            "kind": "monte_carlo_dashboard",
            "meta": {"sessionCount": session_count},
            "overall": {},
            "sessions": [],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Observability V1 pipeline")
    parser.add_argument("output_dir", type=Path, help="Backtest output directory")
    parser.add_argument("--dashboard", type=Path, default=None, help="Path to dashboard.json")
    parser.add_argument("--strategy-path", default="", help="Strategy file path for metadata")
    parser.add_argument("--parent-run-id", default=None, help="Parent run ID for lineage tracking")
    parser.add_argument("--db-path", type=Path, default=None, help="SQLite DB path for packet store")
    parser.add_argument("--output-json", type=Path, default=None, help="Write packet short to JSON file")
    parser.add_argument("--ticks-per-session", type=int, default=None, help="Ticks per session (auto-detected if omitted)")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if not output_dir.exists():
        print(f"Error: output directory does not exist: {output_dir}", file=sys.stderr)
        return 1

    # --- Step 1: Load dashboard ---
    print(f"Loading dashboard from {output_dir}...")
    dashboard = load_or_build_dashboard(output_dir, args.dashboard)

    # --- Step 2: Build event ledger ---
    print("Building event ledger from sample sessions...")
    event_ledger = build_event_ledger(output_dir)
    sample_count = event_ledger["provenance"]["sample_count"]
    print(f"  Found {sample_count} sample session(s) with trace data.")

    # --- Step 3: Fill decomposition ---
    print("Running fill decomposition...")
    fill_decomp = aggregate_fill_decomposition(event_ledger["session_ledgers"])
    total_fills = fill_decomp["provenance"]["total_strategy_fills"]
    print(f"  Analyzed {total_fills} strategy fills.")

    # --- Step 4: Regime analysis ---
    print("Running regime analysis...")
    regime_summary = summarize_regimes(event_ledger["session_ledgers"])

    # --- Step 5: Build research packet ---
    print("Assembling research packet...")
    packet = build_packet(
        dashboard=dashboard,
        event_ledger=event_ledger,
        fill_decomp=fill_decomp,
        regime_summary=regime_summary,
        strategy_path=args.strategy_path,
        ticks_per_session=args.ticks_per_session,
    )

    short = packet["short"]
    full = packet["full"]

    # --- Step 6: Store ---
    print("Storing research packet...")
    store = PacketStore(args.db_path)
    run_id = store.store(short, full, parent_run_id=args.parent_run_id)
    store.close()
    print(f"  Stored as run_id={run_id}")

    # --- Step 7: Output ---
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as f:
            json.dump(short, f, indent=2, default=str)
        print(f"  Wrote packet short to {args.output_json}")

    # --- Print summary ---
    print("\n=== PACKET SHORT SUMMARY ===")
    print(f"Candidate ID:  {short['candidate_id']}")
    print(f"Confidence:    {short['confidence']}")
    print(f"Mean P&L:      {short['pnl']['mean']:.2f}")
    print(f"Std P&L:       {short['pnl']['std']:.2f}")
    print(f"Sharpe-like:   {short['pnl']['sharpe_like']:.4f}")
    print(f"Positive rate: {short['pnl']['positive_rate']:.2%}")
    if short.get("fill_quality"):
        fq = short["fill_quality"]
        print(f"Passive fill:  {fq['passive_fill_rate']:.2%}")
        print(f"Taker fills:   {fq['taker_fill_count']}")
        print(f"Maker fills:   {fq['maker_fill_count']}")
    if short.get("drawdown"):
        dd = short["drawdown"]
        print(f"Mean max DD:   {dd['mean_max_drawdown']:.2f}")
    eff = short.get("efficiency", {})
    if eff.get("pnl_per_fill") is not None:
        print(f"PnL/fill:      {eff['pnl_per_fill']:.4f}")
    if eff.get("pnl_per_tick") is not None:
        print(f"PnL/tick:      {eff['pnl_per_tick']:.4f}")
    print(f"Kill:          {short['kill']['recommended']} (strength={short['kill']['strength']})")
    print(f"Promote:       {short['promote']['recommended']} (strength={short['promote']['strength']})")
    print(f"\nDiagnosis: {short['diagnosis']}")

    if short["warnings"]:
        print(f"\nWarnings ({len(short['warnings'])}):")
        for w in short["warnings"]:
            print(f"  - {w}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
