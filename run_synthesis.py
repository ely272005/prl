#!/usr/bin/env python3
"""CLI for the Strategy Synthesis Engine — alpha cards to strategy tasks.

Usage examples:

    # Full synthesis from discovery report + bank
    python run_synthesis.py tmp/bank/73474_output/discovery/discovery_report.json \
        --bank tmp/bank

    # Top 5 tasks only
    python run_synthesis.py tmp/bank/73474_output/discovery/discovery_report.json \
        --bank tmp/bank --max-tasks 5

    # Product-focused batch
    python run_synthesis.py tmp/bank/73474_output/discovery/discovery_report.json \
        --bank tmp/bank --product EMERALDS

    # Balanced batch mode (diverse mix)
    python run_synthesis.py tmp/bank/73474_output/discovery/discovery_report.json \
        --bank tmp/bank --mode balanced

    # Control-attack mode (exploit + controls paired)
    python run_synthesis.py tmp/bank/73474_output/discovery/discovery_report.json \
        --bank tmp/bank --mode control_attack

    # Single parent focus
    python run_synthesis.py tmp/bank/73474_output/discovery/discovery_report.json \
        --bank tmp/bank --mode single_parent --parent-id 73474

    # Emit briefs only (one .md file per task)
    python run_synthesis.py tmp/bank/73474_output/discovery/discovery_report.json \
        --bank tmp/bank --briefs-only

    # Custom output location
    python run_synthesis.py tmp/bank/73474_output/discovery/discovery_report.json \
        --bank tmp/bank --out tmp/synthesis_results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from discovery.comparison import load_packets_from_bank
from synthesis.converter import convert_cards_to_tasks
from synthesis.batch import build_batch, BATCH_MODES
from synthesis.report import (
    write_json_report,
    write_markdown_report,
    write_brief_files,
)


def _load_cards(report_path: Path) -> list[dict]:
    """Load alpha cards from a discovery report JSON."""
    with report_path.open() as f:
        report = json.load(f)

    cards = report.get("alpha_cards", [])
    if not cards:
        # Try cards_by_category as fallback
        for cat_cards in report.get("cards_by_category", {}).values():
            cards.extend(cat_cards)

    return cards


def _load_parents(bank_dir: Path) -> list[dict]:
    """Load parent strategies from bank packets."""
    packets = load_packets_from_bank(bank_dir)
    return packets


def cmd_run(args: argparse.Namespace) -> int:
    report_path = Path(args.discovery_report).resolve()
    if not report_path.exists():
        print(f"Error: discovery report not found: {report_path}", file=sys.stderr)
        return 1

    bank_dir = Path(args.bank).resolve() if args.bank else None
    if bank_dir and not bank_dir.exists():
        print(f"Error: bank directory not found: {bank_dir}", file=sys.stderr)
        return 1

    # Load inputs
    print(f"Loading discovery report: {report_path}")
    cards = _load_cards(report_path)
    print(f"  Alpha cards: {len(cards)}")

    parents = []
    if bank_dir:
        print(f"Loading parent strategies from: {bank_dir}")
        parents = _load_parents(bank_dir)
        print(f"  Parent candidates: {len(parents)}")
    print()

    if not cards:
        print("Error: no alpha cards found in the report.", file=sys.stderr)
        return 1

    if not parents:
        print("Warning: no parent strategies — tasks will use default parent.", file=sys.stderr)
        print()

    # Filter by product if requested
    if args.product:
        cards = [c for c in cards if args.product in c.get("products", [])]
        print(f"  After product filter ({args.product}): {len(cards)} cards")
        if not cards:
            print(f"Error: no cards found for product {args.product}.", file=sys.stderr)
            return 1

    # Convert cards to tasks
    max_tasks = args.max_tasks
    print(f"Converting {len(cards)} cards to tasks (max {max_tasks})...")
    tasks = convert_cards_to_tasks(cards, parents, max_tasks=max_tasks)
    print(f"  Generated {len(tasks)} strategy tasks")
    print()

    # Print task summary
    for t in tasks:
        pri = t.priority[0].upper()
        products = ", ".join(t.product_scope)
        print(f"  [{pri}] {t.task_id:5s} {t.task_type:22s} {products:12s} {t.title[:60]}")
    print()

    # Build batch
    mode = args.mode
    kwargs = {"max_tasks": max_tasks}
    if mode == "single_parent":
        if not args.parent_id:
            print("Error: --parent-id required for single_parent mode.", file=sys.stderr)
            return 1
        kwargs["parent_id"] = args.parent_id
    elif mode == "product_focus":
        if not args.product:
            print("Error: --product required for product_focus mode.", file=sys.stderr)
            return 1
        kwargs["product"] = args.product

    print(f"Building batch (mode: {mode})...")
    batch = build_batch(mode, tasks, parents, **kwargs)
    print(f"  Batch: {batch.batch_id} — {batch.title}")
    print(f"  Tasks: {len(batch.tasks)} exploit/defend + {len(batch.controls)} controls")
    if batch.overlap_warnings:
        for w in batch.overlap_warnings:
            print(f"  WARNING: {w}")
    print()

    # Output
    out_dir = Path(args.out).resolve() if args.out else report_path.parent.parent / "synthesis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.briefs_only:
        # Write only individual brief files
        brief_dir = out_dir / "briefs"
        written = write_brief_files(batch, brief_dir, cards)
        print(f"Brief files written to: {brief_dir}")
        for p in written:
            print(f"  {p.name}")
    else:
        # Write full reports
        json_path = out_dir / "synthesis_report.json"
        md_path = out_dir / "synthesis_report.md"
        brief_dir = out_dir / "briefs"

        write_json_report(batch, json_path, cards)
        write_markdown_report(batch, md_path, cards)
        written = write_brief_files(batch, brief_dir, cards)

        print(f"Reports written:")
        print(f"  JSON:   {json_path}")
        print(f"  MD:     {md_path}")
        print(f"  Briefs: {brief_dir}/ ({len(written)} files)")

    # Stats
    by_type: dict[str, int] = {}
    by_priority: dict[str, int] = {}
    for t in batch.tasks:
        by_type[t.task_type] = by_type.get(t.task_type, 0) + 1
        by_priority[t.priority] = by_priority.get(t.priority, 0) + 1

    print(f"\nTask types: {by_type}")
    print(f"Priorities: {by_priority}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strategy Synthesis Engine — alpha cards to strategy tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "discovery_report", type=str,
        help="Path to discovery_report.json from the Discovery Engine",
    )
    parser.add_argument(
        "--bank", type=str, default=None,
        help="Bank directory containing *_packet.json files (parent candidates)",
    )
    parser.add_argument(
        "--mode", type=str, default="top_priority",
        choices=list(BATCH_MODES),
        help="Batch construction mode (default: top_priority)",
    )
    parser.add_argument(
        "--product", type=str, default=None,
        choices=["EMERALDS", "TOMATOES"],
        help="Focus on a single product",
    )
    parser.add_argument(
        "--parent-id", type=str, default=None,
        help="Parent case ID for single_parent mode",
    )
    parser.add_argument(
        "--max-tasks", type=int, default=8,
        help="Maximum number of exploit/defend tasks (default: 8)",
    )
    parser.add_argument(
        "--briefs-only", action="store_true",
        help="Only write individual brief files, skip JSON/MD reports",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output directory for reports",
    )

    args = parser.parse_args()
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
