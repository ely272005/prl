#!/usr/bin/env python3
"""CLI for the Discovery Engine — regime + bot weakness discovery.

Usage examples:

    # Full discovery on one backtest output + bank comparison
    python run_discovery.py tmp/bank/73474_output --bank tmp/bank

    # With probe results from a previous mechanics run
    python run_discovery.py tmp/bank/73474_output --bank tmp/bank \
        --probes tmp/bank/73474_output/mechanics/mechanics_report.json

    # One product only
    python run_discovery.py tmp/bank/73474_output --bank tmp/bank --product TOMATOES

    # Multiple output directories (merges sessions)
    python run_discovery.py tmp/bank/73474_output tmp/bank/73479_output --bank tmp/bank

    # Custom output location and card limit
    python run_discovery.py tmp/bank/73474_output --bank tmp/bank \
        --out tmp/discovery_results --max-cards 10

    # Session-only analysis (no bank comparison)
    python run_discovery.py tmp/bank/73474_output
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from discovery.scanner import DiscoveryScanner
from discovery.report import write_json_report, write_markdown_report


def cmd_run(args: argparse.Namespace) -> int:
    output_dirs = [Path(d).resolve() for d in args.output_dirs]

    for d in output_dirs:
        if not d.exists():
            print(f"Error: directory does not exist: {d}", file=sys.stderr)
            return 1

    bank_dir = Path(args.bank).resolve() if args.bank else None
    probe_path = Path(args.probes).resolve() if args.probes else None
    products = [args.product] if args.product else None

    scanner = DiscoveryScanner(
        output_dirs=output_dirs,
        bank_dir=bank_dir,
        probe_report_path=probe_path,
        products=products,
    )

    # Load and summarize
    print(f"Loading session data from {len(output_dirs)} director{'y' if len(output_dirs) == 1 else 'ies'}...")
    summary = scanner.summary()
    print(f"  Sessions: {summary.get('session_count', 0)}")
    if summary.get("packet_count"):
        print(f"  Packets:  {summary['packet_count']} "
              f"({summary.get('winner_count', 0)} promoted, "
              f"{summary.get('loser_count', 0)} rejected)")
    if summary.get("probe_result_count"):
        print(f"  Probe results: {summary['probe_result_count']}")
    print()

    if summary.get("session_count", 0) == 0:
        print("Error: no session data found.", file=sys.stderr)
        return 1

    # Run discovery
    max_cards = args.max_cards
    print(f"Running discovery (max {max_cards} cards)...")
    cards = scanner.discover(max_cards=max_cards)
    print(f"  Generated {len(cards)} alpha cards")
    print()

    # Print card summary
    for card in cards:
        conf = card.confidence[0].upper()
        cat_short = card.category[:12]
        print(f"  [{conf}] {card.card_id:5s} {cat_short:14s} {', '.join(card.products):12s} {card.title[:70]}")
    print()

    # Write reports
    out_dir = Path(args.out).resolve() if args.out else output_dirs[0] / "discovery"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "discovery_report.json"
    md_path = out_dir / "discovery_report.md"

    comparison = scanner.run_comparison()
    write_json_report(cards, json_path, summary, comparison)
    write_markdown_report(cards, md_path, summary, comparison)

    print(f"Reports written:")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")

    # Confidence breakdown
    by_conf: dict[str, int] = {}
    for c in cards:
        by_conf[c.confidence] = by_conf.get(c.confidence, 0) + 1
    print(f"\nConfidence: {by_conf}")

    by_cat: dict[str, int] = {}
    for c in cards:
        by_cat[c.category] = by_cat.get(c.category, 0) + 1
    print(f"Categories: {by_cat}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discovery Engine — regime + weakness discovery for alpha card generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "output_dirs", nargs="+", type=str,
        help="Backtest output directories with sessions/ subdirectory",
    )
    parser.add_argument(
        "--bank", type=str, default=None,
        help="Bank directory containing *_packet.json files for comparison",
    )
    parser.add_argument(
        "--probes", type=str, default=None,
        help="Path to mechanics_report.json for probe-driven discovery",
    )
    parser.add_argument(
        "--product", type=str, default=None,
        choices=["EMERALDS", "TOMATOES"],
        help="Run discovery for one product only",
    )
    parser.add_argument(
        "--max-cards", type=int, default=20,
        help="Maximum number of alpha cards to generate (default: 20)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output directory for reports",
    )

    args = parser.parse_args()
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
