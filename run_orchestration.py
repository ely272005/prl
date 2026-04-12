#!/usr/bin/env python3
"""CLI for the Campaign Orchestrator — plans, queues, and manages the research loop.

Usage examples:

    # Build a full run plan from Phase 5 adjudication output
    python run_orchestration.py plan \
        --adjudication tmp/adjudication \
        --bank tmp/bank \
        --out tmp/orchestration

    # Show current champions
    python run_orchestration.py champions \
        --bank tmp/bank

    # Show official-testing shortlist
    python run_orchestration.py official \
        --adjudication tmp/adjudication \
        --bank tmp/bank

    # Route candidates by verdict
    python run_orchestration.py route \
        --adjudication tmp/adjudication

    # Build Prosperity-GPT brief for a campaign
    python run_orchestration.py handoff \
        --adjudication tmp/adjudication \
        --bank tmp/bank \
        --out tmp/orchestration

    # Show campaign history
    python run_orchestration.py history \
        --history tmp/orchestration/campaign_history.json

    # Explain why a candidate is or is not in the official queue
    python run_orchestration.py explain \
        --adjudication tmp/adjudication \
        --bank tmp/bank \
        --candidate mh07
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from discovery.comparison import load_packets_from_bank
from orchestration.campaigns import create_campaigns_from_actions, reset_counter
from orchestration.routing import route_candidates, summarize_routing
from orchestration.champions import build_champion_table
from orchestration.redundancy import check_all_redundancy
from orchestration.allocation import allocate_budget
from orchestration.run_plan import build_run_plan
from orchestration.official_queue import build_official_queue, generate_official_memo
from orchestration.handoff import build_all_handoffs
from orchestration.history import load_history, summarize_recent, campaign_stats
from orchestration.report import write_all_reports


def _load_json(path: Path) -> dict | list:
    with path.open() as f:
        return json.load(f)


def _load_adjudication(adj_dir: Path) -> dict:
    """Load all Phase 5 outputs from the adjudication directory."""
    data = {}
    files = {
        "candidate_verdicts": "candidate_verdicts.json",
        "hypothesis_verdicts": "hypothesis_verdicts.json",
        "frontier_updates": "frontier_updates.json",
        "batch_learnings": "batch_learnings.json",
        "next_actions": "next_actions.json",
    }
    for key, fname in files.items():
        p = adj_dir / fname
        if p.exists():
            raw = _load_json(p)
            # Unwrap: some files have a wrapper with generated_at + the actual data
            if key == "candidate_verdicts":
                data[key] = raw.get("verdicts", raw) if isinstance(raw, dict) else raw
            elif key == "hypothesis_verdicts":
                data[key] = raw.get("verdicts", raw) if isinstance(raw, dict) else raw
            elif key == "next_actions":
                data[key] = raw.get("actions", raw) if isinstance(raw, dict) else raw
            else:
                data[key] = raw
        else:
            print(f"  Warning: {p} not found", file=sys.stderr)
            data[key] = [] if key in ("candidate_verdicts", "hypothesis_verdicts", "next_actions") else {}
    return data


def cmd_plan(args: argparse.Namespace) -> int:
    """Build a full run plan."""
    adj_dir = Path(args.adjudication).resolve()
    if not adj_dir.exists():
        print(f"Error: adjudication directory not found: {adj_dir}", file=sys.stderr)
        return 1

    print(f"Loading adjudication output: {adj_dir}")
    adj = _load_adjudication(adj_dir)

    candidate_verdicts = adj["candidate_verdicts"]
    hypothesis_verdicts = adj["hypothesis_verdicts"]
    frontier_updates = adj["frontier_updates"]
    learnings = adj["batch_learnings"]
    next_actions = adj["next_actions"]

    print(f"  Candidates: {len(candidate_verdicts)}")
    print(f"  Hypotheses: {len(hypothesis_verdicts)}")
    print(f"  Next actions: {len(next_actions)}")

    # Load bank
    bank_packets = []
    if args.bank:
        bank_dir = Path(args.bank).resolve()
        print(f"Loading bank: {bank_dir}")
        bank_packets = load_packets_from_bank(bank_dir)
        print(f"  Bank packets: {len(bank_packets)}")

    frontier_packets = [
        p for p in bank_packets
        if p.get("packet_short", p).get("promote", {}).get("recommended", False)
    ]
    print(f"  Frontier: {len(frontier_packets)} promoted candidates")
    print()

    # Budget
    budget = None
    if args.budget:
        budget = json.loads(args.budget)

    exploit_ratio = args.exploit_ratio if args.exploit_ratio else 0.70

    # Reset campaign counter
    reset_counter()

    # Build run plan
    print("Building run plan...")
    plan = build_run_plan(
        next_actions=next_actions,
        learnings=learnings,
        frontier_updates=frontier_updates,
        candidate_verdicts=candidate_verdicts,
        hypothesis_verdicts=hypothesis_verdicts,
        frontier_packets=frontier_packets,
        budget=budget,
        exploit_ratio=exploit_ratio,
    )

    campaigns = plan.get("campaigns", [])
    routing = plan.get("routing_decisions", [])
    redundancy = plan.get("redundancy", {})

    print(f"  Campaigns: {len(campaigns)}")
    for c in campaigns:
        ee = c.get("exploit_explore", "?")[:3]
        n = c.get("allocated_candidates", 0)
        icon = {"high": "[H]", "medium": "[M]", "low": "[L]"}.get(c.get("priority", ""), "[-]")
        print(f"    {icon} {c.get('title', '?')[:40]:40s}  {c.get('campaign_type', '?'):15s}  {ee}  {n} cand")

    print(f"\n  Summary: {plan.get('summary', '?')}")
    print()

    # Official queue
    print("Building official queue...")
    champions = build_champion_table(frontier_packets, frontier_updates)
    official = build_official_queue(candidate_verdicts, routing, champions)
    print(f"  Official queue: {len(official)} candidate(s)")
    for entry in official:
        print(f"    #{entry['rank']} {entry['candidate_id'][:14]:14s} {entry['role']:20s} score={entry['score']:.0f}")
    print()

    # Handoffs
    handoffs = build_all_handoffs(campaigns, learnings, champions)
    print(f"  Handoff briefs: {len(handoffs)}")

    # History
    history_path = Path(args.out).resolve() / "campaign_history.json" if args.out else None
    history = load_history(history_path) if history_path and history_path.exists() else {"campaigns": [], "updated_at": None}

    # Write reports
    out_dir = Path(args.out).resolve() if args.out else adj_dir.parent / "orchestration"
    written = write_all_reports(
        out_dir, plan, official, champions, routing, history, handoffs, learnings,
    )
    print(f"Reports written to {out_dir}:")
    for p in written:
        print(f"  {p.relative_to(out_dir)}")

    return 0


def cmd_champions(args: argparse.Namespace) -> int:
    """Show current champion table."""
    bank_packets = []
    if args.bank:
        bank_dir = Path(args.bank).resolve()
        bank_packets = load_packets_from_bank(bank_dir)

    frontier = [
        p for p in bank_packets
        if p.get("packet_short", p).get("promote", {}).get("recommended", False)
    ]

    if not frontier:
        print("No promoted candidates in bank. Champion table is empty.")
        return 0

    champions = build_champion_table(frontier)
    active = [c for c in champions.get("champions", []) if c["status"] == "active"]

    print("Champion Table")
    print("=" * 70)
    for c in active:
        print(
            f"  {c['role']:25s}  {c['candidate_id'][:14]:14s}  "
            f"PnL={c.get('pnl_mean', 0):6.0f}  "
            f"Sharpe={c.get('sharpe', 0):5.2f}  "
            f"{c.get('confidence', '?')}"
        )
    return 0


def cmd_official(args: argparse.Namespace) -> int:
    """Show official-testing shortlist."""
    adj_dir = Path(args.adjudication).resolve()
    adj = _load_adjudication(adj_dir)

    candidate_verdicts = adj["candidate_verdicts"]
    hypothesis_verdicts = adj["hypothesis_verdicts"]
    learnings = adj["batch_learnings"]

    routing = route_candidates(candidate_verdicts, hypothesis_verdicts, learnings)

    bank_packets = []
    if args.bank:
        bank_dir = Path(args.bank).resolve()
        bank_packets = load_packets_from_bank(bank_dir)

    frontier = [
        p for p in bank_packets
        if p.get("packet_short", p).get("promote", {}).get("recommended", False)
    ]
    champions = build_champion_table(frontier)

    max_slots = args.max_official if args.max_official else 3
    official = build_official_queue(candidate_verdicts, routing, champions, max_slots)

    if not official:
        print("No candidates ready for official testing.")
        return 0

    print(generate_official_memo(official, champions))
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    """Route candidates by verdict."""
    adj_dir = Path(args.adjudication).resolve()
    adj = _load_adjudication(adj_dir)

    candidate_verdicts = adj["candidate_verdicts"]
    hypothesis_verdicts = adj["hypothesis_verdicts"]
    learnings = adj["batch_learnings"]

    decisions = route_candidates(candidate_verdicts, hypothesis_verdicts, learnings)
    print(summarize_routing(decisions))
    return 0


def cmd_handoff(args: argparse.Namespace) -> int:
    """Build Prosperity-GPT handoff briefs."""
    adj_dir = Path(args.adjudication).resolve()
    adj = _load_adjudication(adj_dir)

    learnings = adj["batch_learnings"]
    next_actions = adj["next_actions"]
    frontier_updates = adj["frontier_updates"]
    candidate_verdicts = adj["candidate_verdicts"]

    bank_packets = []
    if args.bank:
        bank_dir = Path(args.bank).resolve()
        bank_packets = load_packets_from_bank(bank_dir)

    frontier = [
        p for p in bank_packets
        if p.get("packet_short", p).get("promote", {}).get("recommended", False)
    ]
    champions = build_champion_table(frontier)

    reset_counter()
    campaigns = create_campaigns_from_actions(
        next_actions, learnings, frontier_updates, candidate_verdicts,
    )
    handoffs = build_all_handoffs(campaigns, learnings, champions)

    out_dir = Path(args.out).resolve() if args.out else adj_dir.parent / "orchestration" / "prosperity_handoff"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ho in handoffs:
        cid = ho.get("campaign_id", "unknown").replace("-", "_")
        path = out_dir / f"{cid}_brief.md"
        with path.open("w") as f:
            f.write(ho.get("brief_markdown", ""))
        print(f"  Written: {path.name}")

    print(f"\n{len(handoffs)} handoff brief(s) written to {out_dir}")
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    """Show campaign history."""
    path = Path(args.history).resolve()
    if not path.exists():
        print("No campaign history found.")
        return 0

    history = load_history(path)
    stats = campaign_stats(history)

    print(f"Campaign History: {stats['total']} campaign(s)")
    print(f"  Budget used: {stats.get('total_budget_used', 0)} candidates")
    print(f"  Worth it: {stats.get('worth_budget_yes', 0)} yes / {stats.get('worth_budget_no', 0)} no / {stats.get('worth_budget_unknown', 0)} unknown")
    print()
    print(summarize_recent(history, n=5))
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    """Explain why a candidate is or is not in the official queue."""
    adj_dir = Path(args.adjudication).resolve()
    adj = _load_adjudication(adj_dir)

    candidate_verdicts = adj["candidate_verdicts"]
    hypothesis_verdicts = adj["hypothesis_verdicts"]
    learnings = adj["batch_learnings"]
    target = args.candidate

    # Find the candidate
    cv = None
    for v in candidate_verdicts:
        cid = v.get("candidate_id", "")
        if cid == target or cid.startswith(target) or target in cid:
            cv = v
            break

    if cv is None:
        print(f"Candidate '{target}' not found in adjudication verdicts.")
        return 1

    # Route it
    routing = route_candidates(candidate_verdicts, hypothesis_verdicts, learnings)
    decision = None
    for d in routing:
        if d.get("candidate_id") == cv.get("candidate_id"):
            decision = d
            break

    print(f"Candidate: {cv.get('candidate_id', '?')}")
    print(f"Verdict:   {cv.get('verdict', '?')}")
    print(f"PnL:       {cv.get('pnl_mean', 0):.0f} (delta {cv.get('pnl_delta', 0):+.0f})")
    print(f"Sharpe:    {cv.get('sharpe', 0):.2f} (delta {cv.get('sharpe_delta', 0):+.2f})")
    print(f"Confidence: {cv.get('confidence', '?')}")
    print(f"Transfer:  {cv.get('transfer_risk', '?')}")
    print()

    if decision:
        print(f"Routing:   {decision.get('action', '?')}")
        print(f"Official:  {'YES' if decision.get('official_eligible') else 'NO'}")
        print(f"Reason:    {decision.get('reason', '?')}")
        if decision.get("dead_zone_warning"):
            print(f"Warning:   {decision['dead_zone_warning']}")
    else:
        print("Routing:   Not found in routing decisions.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Campaign Orchestrator — plans, queues, and manages the research loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # plan
    p_plan = subparsers.add_parser("plan", help="Build a full run plan from Phase 5 output")
    p_plan.add_argument("--adjudication", required=True, help="Adjudication output directory")
    p_plan.add_argument("--bank", help="Bank directory with parent packets")
    p_plan.add_argument("--out", help="Output directory for reports")
    p_plan.add_argument("--budget", help='Budget JSON, e.g. \'{"total_local_candidates": 20}\'')
    p_plan.add_argument("--exploit-ratio", type=float, help="Exploit ratio (default 0.70)")

    # champions
    p_champs = subparsers.add_parser("champions", help="Show current champion table")
    p_champs.add_argument("--bank", required=True, help="Bank directory")

    # official
    p_off = subparsers.add_parser("official", help="Show official-testing shortlist")
    p_off.add_argument("--adjudication", required=True, help="Adjudication output directory")
    p_off.add_argument("--bank", help="Bank directory")
    p_off.add_argument("--max-official", type=int, help="Max official test slots (default 3)")

    # route
    p_route = subparsers.add_parser("route", help="Route candidates by verdict")
    p_route.add_argument("--adjudication", required=True, help="Adjudication output directory")

    # handoff
    p_handoff = subparsers.add_parser("handoff", help="Build Prosperity-GPT handoff briefs")
    p_handoff.add_argument("--adjudication", required=True, help="Adjudication output directory")
    p_handoff.add_argument("--bank", help="Bank directory")
    p_handoff.add_argument("--out", help="Output directory for briefs")

    # history
    p_hist = subparsers.add_parser("history", help="Show campaign history")
    p_hist.add_argument("--history", required=True, help="Path to campaign_history.json")

    # explain
    p_explain = subparsers.add_parser("explain", help="Explain a candidate's queue status")
    p_explain.add_argument("--adjudication", required=True, help="Adjudication output directory")
    p_explain.add_argument("--bank", help="Bank directory")
    p_explain.add_argument("--candidate", required=True, help="Candidate ID to explain")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "plan": cmd_plan,
        "champions": cmd_champions,
        "official": cmd_official,
        "route": cmd_route,
        "handoff": cmd_handoff,
        "history": cmd_history,
        "explain": cmd_explain,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
