#!/usr/bin/env python3
"""CLI for the Mechanics Probe Engine.

Usage examples:

    # Run all probes on one backtest output directory
    python run_mechanics.py tmp/bank/73474_output

    # Run all probes on multiple directories
    python run_mechanics.py tmp/bank/73474_output tmp/gen2/results/ah03_output

    # Run only one probe family
    python run_mechanics.py tmp/bank/73474_output --family passive_fill

    # Run only one probe
    python run_mechanics.py tmp/bank/73474_output --probe pf01_spread_vs_maker_edge

    # Run on one product only
    python run_mechanics.py tmp/bank/73474_output --product TOMATOES

    # Custom output directory
    python run_mechanics.py tmp/bank/73474_output --out tmp/mechanics_results

    # List available probes
    python run_mechanics.py --list
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from mechanics.runner import ProbeRunner
from mechanics.report import write_json_report, write_markdown_report
from mechanics.probe_spec import list_probes, list_families

# Trigger probe registration
import mechanics.probes  # noqa: F401


def cmd_list() -> int:
    """List all available probes."""
    families = list_families()
    print(f"Available probe families: {', '.join(families)}\n")
    for fam in families:
        probes = list_probes(family=fam)
        print(f"  {fam} ({len(probes)} probes):")
        for p in probes:
            print(f"    {p.probe_id}")
            print(f"      {p.title}")
            print(f"      Hypothesis: {p.hypothesis}")
            print()
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run probes and generate reports."""
    output_dirs = [Path(d).resolve() for d in args.output_dirs]

    # Validate directories
    for d in output_dirs:
        if not d.exists():
            print(f"Error: directory does not exist: {d}", file=sys.stderr)
            return 1
        sessions_dir = d / "sessions"
        if not sessions_dir.exists():
            print(f"Error: no sessions/ subdirectory in {d}", file=sys.stderr)
            return 1

    runner = ProbeRunner(output_dirs)

    # Load data
    print(f"Loading session data from {len(output_dirs)} director{'y' if len(output_dirs) == 1 else 'ies'}...")
    summary = runner.summary()
    print(f"  Loaded {summary['total_sessions']} sessions")
    print(f"  Available probes: {len(summary['available_probes'])}")
    print()

    if summary["total_sessions"] == 0:
        print("Error: no session data found.", file=sys.stderr)
        return 1

    # Determine what to run
    products = [args.product] if args.product else None
    families = [args.family] if args.family else None

    if args.probe:
        # Single probe mode
        prod_list = products or ["EMERALDS", "TOMATOES"]
        results = []
        for p in prod_list:
            print(f"Running {args.probe} on {p}...")
            result = runner.run_probe(args.probe, p)
            results.append(result)
            _print_result(result)
    else:
        # Batch mode
        if families:
            print(f"Running family: {families[0]}...")
        else:
            print("Running all probes...")

        results = runner.run_all(products=products, families=families)
        for r in results:
            _print_result(r)

    # Write reports
    out_dir = Path(args.out).resolve() if args.out else output_dirs[0] / "mechanics"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "mechanics_report.json"
    md_path = out_dir / "mechanics_report.md"

    write_json_report(results, json_path, summary)
    write_markdown_report(results, md_path, summary)

    print(f"\nReports written:")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")

    # Print verdict summary
    verdicts = {}
    for r in results:
        verdicts[r.verdict] = verdicts.get(r.verdict, 0) + 1
    print(f"\nVerdict summary: {verdicts}")

    return 0


def _print_result(r) -> None:
    """Print a one-line result summary."""
    icons = {"supported": "+", "refuted": "-", "inconclusive": "?", "insufficient_data": "!"}
    icon = icons.get(r.verdict, "?")
    conf = r.confidence[0].upper()
    print(f"  [{icon}] {r.probe_id:40s} {r.product:10s} {r.verdict:20s} [{conf}] {r.detail[:80]}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mechanics Probe Engine — run targeted probes on backtest session data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("output_dirs", nargs="*", type=str, help="Backtest output directories with sessions/ subdirectory")
    parser.add_argument("--list", action="store_true", help="List all available probes and exit")
    parser.add_argument("--family", type=str, default=None, help="Run only this probe family")
    parser.add_argument("--probe", type=str, default=None, help="Run only this specific probe ID")
    parser.add_argument("--product", type=str, default=None, choices=["EMERALDS", "TOMATOES"], help="Run on one product only")
    parser.add_argument("--out", type=str, default=None, help="Output directory for reports")

    args = parser.parse_args()

    if args.list:
        return cmd_list()

    if not args.output_dirs:
        parser.error("Provide at least one output directory, or use --list.")

    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
