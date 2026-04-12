#!/usr/bin/env python3
"""CLI for the Adjudication Engine — experiment results to verdicts and learnings.

Usage examples:

    # Adjudicate a full batch (synthesis report + candidate packets + bank)
    python run_adjudication.py \
        --synthesis tmp/synthesis/synthesis_report.json \
        --candidates tmp/results/ \
        --bank tmp/bank \
        --out tmp/adjudication

    # Adjudicate a single candidate vs its parent
    python run_adjudication.py \
        --candidate-packet tmp/results/mh07_packet.json \
        --parent-packet tmp/bank/73474_packet.json \
        --task-json tmp/synthesis/tasks/T001.json

    # Compare a candidate to the frontier
    python run_adjudication.py \
        --candidate-packet tmp/results/mh07_packet.json \
        --bank tmp/bank \
        --frontier-only

    # Update the frontier
    python run_adjudication.py \
        --synthesis tmp/synthesis/synthesis_report.json \
        --candidates tmp/results/ \
        --bank tmp/bank \
        --update-frontier

    # Show only rejected hypotheses
    python run_adjudication.py \
        --synthesis tmp/synthesis/synthesis_report.json \
        --candidates tmp/results/ \
        --bank tmp/bank \
        --filter-verdict reject

    # Show only validated mechanisms
    python run_adjudication.py \
        --synthesis tmp/synthesis/synthesis_report.json \
        --candidates tmp/results/ \
        --bank tmp/bank \
        --filter-hypothesis validated

    # Emit a Prosperity-GPT-ready summary of what to do next
    python run_adjudication.py \
        --synthesis tmp/synthesis/synthesis_report.json \
        --candidates tmp/results/ \
        --bank tmp/bank \
        --gpt-summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from discovery.comparison import load_packets_from_bank
from adjudication.verdicts import adjudicate_candidate
from adjudication.hypothesis import adjudicate_hypothesis
from adjudication.frontier import compute_frontier_updates
from adjudication.learnings import extract_batch_learnings
from adjudication.next_actions import recommend_next_actions, format_gpt_summary
from adjudication.report import write_all_reports
from adjudication.comparison import compare_pair, compare_to_frontier


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _load_packets_from_dir(d: Path) -> list[dict]:
    """Load *_packet.json files from a directory."""
    packets = []
    for p in sorted(d.glob("*_packet.json")):
        try:
            data = _load_json(p)
            # Normalize: some packets have packet_short at top level
            if "packet_short" not in data and "pnl" in data:
                data = {"packet_short": data, "_path": str(p)}
            data.setdefault("_path", str(p))
            data.setdefault("_case_id", p.stem.replace("_packet", ""))
            packets.append(data)
        except Exception as e:
            print(f"  Warning: could not load {p}: {e}", file=sys.stderr)
    return packets


def _find_parent(parent_id: str, bank_packets: list[dict]) -> dict | None:
    for p in bank_packets:
        pid = p.get("_case_id", p.get("case_id", ""))
        if pid == parent_id:
            return p
    return None


def _load_synthesis_data(synthesis_path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Load tasks, cards, and batch from a synthesis report."""
    report = _load_json(synthesis_path)
    batch = report.get("batch", {})
    tasks = batch.get("tasks", [])
    controls = batch.get("controls", [])
    cards = report.get("source_cards", [])
    return tasks + controls, cards, [batch]


def cmd_batch(args: argparse.Namespace) -> int:
    """Adjudicate a full batch."""
    synthesis_path = Path(args.synthesis).resolve()
    if not synthesis_path.exists():
        print(f"Error: synthesis report not found: {synthesis_path}", file=sys.stderr)
        return 1

    candidates_dir = Path(args.candidates).resolve() if args.candidates else None
    bank_dir = Path(args.bank).resolve() if args.bank else None

    # Load inputs
    print(f"Loading synthesis report: {synthesis_path}")
    tasks, cards, batches = _load_synthesis_data(synthesis_path)
    print(f"  Tasks: {len(tasks)}")

    bank_packets = []
    if bank_dir:
        print(f"Loading bank: {bank_dir}")
        bank_packets = load_packets_from_bank(bank_dir)
        print(f"  Bank packets: {len(bank_packets)}")

    candidate_packets = []
    if candidates_dir and candidates_dir.exists():
        print(f"Loading candidate packets: {candidates_dir}")
        candidate_packets = _load_packets_from_dir(candidates_dir)
        print(f"  Candidate packets: {len(candidate_packets)}")

    if not candidate_packets:
        print("Warning: no candidate packets found. Running in dry-run mode.", file=sys.stderr)
        # In dry-run, use bank packets as stand-in candidates for testing
        if bank_packets:
            candidate_packets = bank_packets[:3]
            print(f"  Using {len(candidate_packets)} bank packets as stand-in candidates.")

    # Build frontier from promoted bank packets
    frontier = [p for p in bank_packets if p.get("packet_short", p).get("promote", {}).get("recommended", False)]
    print(f"  Frontier: {len(frontier)} promoted candidates")
    print()

    # Match tasks to candidate packets (by index for now)
    # In practice, tasks would link to candidates by task_id in the filename
    task_count = min(len(tasks), len(candidate_packets))

    # Adjudicate each candidate
    print(f"Adjudicating {task_count} candidates...")
    candidate_verdicts = []
    hypothesis_verdicts = []

    for i in range(task_count):
        task = tasks[i]
        candidate = candidate_packets[i]
        parent_id = task.get("parent_id", "")
        parent = _find_parent(parent_id, bank_packets) or candidate  # fallback

        family_name = task.get("parent_family", "unknown")
        family_peers = [p for p in bank_packets if p.get("_family", "") == family_name]

        # Candidate verdict
        cv = adjudicate_candidate(
            candidate=candidate,
            parent=parent,
            task=task,
            frontier=frontier,
            family_peers=family_peers,
            family_name=family_name,
        )
        candidate_verdicts.append(cv)

        # Hypothesis verdict
        card = cards[i] if i < len(cards) else None
        hv = adjudicate_hypothesis(cv, task, card)
        hypothesis_verdicts.append(hv)

        # Print summary
        verdict = cv["verdict"]
        icon = {"frontier_challenger": "++", "escalate": " +", "keep": " =", "reject": " X",
                "suspect_simulator_gain": " ?", "control_success": "CS", "control_failure": "CF"}.get(verdict, "  ")
        print(f"  [{icon}] {cv.get('candidate_id', '?')[:12]:12s} {verdict:25s} PnL{cv['pnl_delta']:+6.0f} Sharpe{cv['sharpe_delta']:+6.2f}")

    print()

    # Frontier updates
    frontier_updates = compute_frontier_updates(
        current_frontier=frontier,
        candidate_verdicts=candidate_verdicts,
        all_packets=bank_packets + candidate_packets,
    )
    print(f"Frontier: {frontier_updates['frontier_size_before']} → {frontier_updates['frontier_size_after']}")
    if frontier_updates["additions"]:
        print(f"  Added: {', '.join(a['candidate_id'] for a in frontier_updates['additions'])}")
    if frontier_updates["retirements"]:
        print(f"  Retired: {', '.join(r['candidate_id'] for r in frontier_updates['retirements'])}")
    print()

    # Learnings
    learnings = extract_batch_learnings(candidate_verdicts, hypothesis_verdicts, tasks)
    print(f"Learnings: {learnings['summary']}")

    # Next actions
    next_actions = recommend_next_actions(
        candidate_verdicts, hypothesis_verdicts, learnings, frontier_updates,
    )
    print(f"Next actions: {len(next_actions)}")
    for action in next_actions[:5]:
        prio = action["priority"][0].upper()
        print(f"  [{prio}] {action['action_type']}: {action['detail'][:70]}")
    print()

    # Filter output if requested
    if args.filter_verdict:
        candidate_verdicts = [v for v in candidate_verdicts if v["verdict"] == args.filter_verdict]
        print(f"Filtered to {len(candidate_verdicts)} candidates with verdict '{args.filter_verdict}'")

    if args.filter_hypothesis:
        hypothesis_verdicts = [v for v in hypothesis_verdicts if v["outcome"] == args.filter_hypothesis]
        print(f"Filtered to {len(hypothesis_verdicts)} hypotheses with outcome '{args.filter_hypothesis}'")

    # GPT summary
    if args.gpt_summary:
        print("\n" + "=" * 60)
        print(format_gpt_summary(next_actions, learnings))
        print("=" * 60)

    # Write reports
    out_dir = Path(args.out).resolve() if args.out else synthesis_path.parent.parent / "adjudication"
    written = write_all_reports(
        out_dir, candidate_verdicts, hypothesis_verdicts,
        frontier_updates, learnings, next_actions,
    )
    print(f"\nReports written to {out_dir}:")
    for p in written:
        print(f"  {p.name}")

    return 0


def cmd_single(args: argparse.Namespace) -> int:
    """Adjudicate a single candidate vs its parent."""
    candidate = _load_json(Path(args.candidate_packet).resolve())
    parent = _load_json(Path(args.parent_packet).resolve())

    task = {}
    if args.task_json:
        task = _load_json(Path(args.task_json).resolve())

    # Normalize
    if "packet_short" not in candidate and "pnl" in candidate:
        candidate = {"packet_short": candidate}
    if "packet_short" not in parent and "pnl" in parent:
        parent = {"packet_short": parent}

    bank_packets = []
    frontier = []
    if args.bank:
        bank_dir = Path(args.bank).resolve()
        bank_packets = load_packets_from_bank(bank_dir)
        frontier = [p for p in bank_packets if p.get("packet_short", p).get("promote", {}).get("recommended", False)]

    if args.frontier_only:
        result = compare_to_frontier(candidate, frontier)
        print(json.dumps(_sanitize_for_print(result), indent=2))
        return 0

    cv = adjudicate_candidate(
        candidate=candidate,
        parent=parent,
        task=task,
        frontier=frontier,
    )

    _print_verdict(cv)
    return 0


def _print_verdict(cv: dict) -> None:
    print(f"Candidate: {cv.get('candidate_id', '?')}")
    print(f"Verdict:   {cv.get('verdict', '?')}")
    print(f"Reason:    {cv.get('reason', '?')}")
    print(f"PnL:       {cv.get('pnl_mean', 0):.0f} (delta {cv.get('pnl_delta', 0):+.0f})")
    print(f"Sharpe:    {cv.get('sharpe', 0):.2f} (delta {cv.get('sharpe_delta', 0):+.2f})")
    print(f"EMERALDS:  {cv.get('emerald_mean', 0):.0f} (delta {cv.get('emerald_delta', 0):+.0f})")
    print(f"TOMATOES:  {cv.get('tomato_mean', 0):.0f} (delta {cv.get('tomato_delta', 0):+.0f})")
    print(f"Mechanism: {cv.get('mechanism_interpretation', '?')}")
    print(f"Transfer:  {cv.get('transfer_risk', '?')}")
    print(f"Next:      {cv.get('recommended_next_action', '?')}")

    suspicion = cv.get("suspicion", {})
    if suspicion.get("flags"):
        print(f"Suspicion: {suspicion['suspicion_level']}")
        for f in suspicion["flags"]:
            print(f"  - {f.get('flag')}: {f.get('detail', '')}")

    pres = cv.get("preservation_audit", {})
    if pres.get("violations"):
        print(f"Preservation: {pres['verdict']}")
        for v in pres["violations"]:
            print(f"  - [{v.get('severity')}] {v.get('detail', '')}")


def _sanitize_for_print(obj):
    import math as _math
    if isinstance(obj, float):
        if _math.isnan(obj) or _math.isinf(obj):
            return None
        return round(obj, 4)
    if isinstance(obj, dict):
        return {k: _sanitize_for_print(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_print(v) for v in obj]
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Adjudication Engine — experiment results to verdicts and learnings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Batch mode arguments
    parser.add_argument("--synthesis", type=str, help="Path to synthesis_report.json")
    parser.add_argument("--candidates", type=str, help="Directory of candidate *_packet.json files")
    parser.add_argument("--bank", type=str, help="Bank directory with parent *_packet.json files")
    parser.add_argument("--out", type=str, help="Output directory for reports")

    # Single candidate mode
    parser.add_argument("--candidate-packet", type=str, help="Single candidate packet JSON")
    parser.add_argument("--parent-packet", type=str, help="Parent packet JSON for comparison")
    parser.add_argument("--task-json", type=str, help="Task JSON for constraint audit")

    # Frontier mode
    parser.add_argument("--frontier-only", action="store_true", help="Only compare to frontier")
    parser.add_argument("--update-frontier", action="store_true", help="Update frontier after adjudication")

    # Filters
    parser.add_argument("--filter-verdict", type=str, help="Show only candidates with this verdict")
    parser.add_argument("--filter-hypothesis", type=str, help="Show only hypotheses with this outcome")

    # Output modes
    parser.add_argument("--gpt-summary", action="store_true", help="Emit a Prosperity-GPT-ready summary")

    args = parser.parse_args()

    # Dispatch
    if args.candidate_packet and args.parent_packet:
        return cmd_single(args)
    elif args.synthesis:
        return cmd_batch(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
