"""Official-testing queue — ranks candidates for manual submission to IMC.

This is the most important decision layer in Phase 6.
It answers: "Which candidates should we upload to the official server, in what
order, and why?"

Each recommendation includes:
  - candidate_id, parent, mechanism, family
  - local metrics and packet quality
  - risk notes and transfer-risk assessment
  - why it is being recommended
  - whether it is a main bet or control bet
  - explicit role assignment

Does NOT interact with the competition website. Only produces recommendations.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


OFFICIAL_ROLES = (
    "safest_challenger",     # Best risk-adjusted, highest confidence
    "highest_ceiling",       # Highest mean or Sharpe, higher transfer risk
    "control_candidate",     # Near-twin or calibration for A/B comparison
    "calibration_anchor",    # Reproduction of known-good for baseline
    "champion_replacement",  # Intended to replace current champion
)

QUEUE_STATUSES = ("queued", "submitted", "confirmed", "rejected", "withdrawn")


def build_official_queue(
    candidate_verdicts: list[dict[str, Any]],
    routing_decisions: list[dict[str, Any]],
    champions: dict[str, Any] | None = None,
    max_slots: int = 3,
) -> list[dict[str, Any]]:
    """Build a ranked official-testing queue.

    Parameters
    ----------
    candidate_verdicts : list[dict]
        Full verdicts from Phase 5.
    routing_decisions : list[dict]
        From route_candidates() — identifies official-eligible candidates.
    champions : dict, optional
        Current champion table for comparison.
    max_slots : int
        Maximum official test slots available.
    """
    # Collect official-eligible candidates
    eligible_ids = {
        r["candidate_id"]
        for r in routing_decisions
        if r.get("official_eligible")
    }

    eligible = [
        cv for cv in candidate_verdicts
        if cv.get("candidate_id") in eligible_ids
    ]

    if not eligible:
        return []

    # Score and rank
    scored = [(cv, _rank_score(cv)) for cv in eligible]
    scored.sort(key=lambda x: -x[1])

    # Build queue entries
    queue: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()

    for rank_idx, (cv, score) in enumerate(scored[:max_slots]):
        role = _assign_official_role(cv, rank_idx, champions)
        queue.append({
            "rank": rank_idx + 1,
            "candidate_id": cv.get("candidate_id", "?"),
            "parent_id": cv.get("parent_id", "?"),
            "family": cv.get("family", "unknown"),
            "mechanism": cv.get("attribution", {}).get("dominant_mechanism", "?"),
            "hypothesis": cv.get("source_hypothesis", "?"),
            "local_metrics": {
                "pnl_mean": cv.get("pnl_mean", 0),
                "sharpe": cv.get("sharpe", 0),
                "positive_rate": cv.get("positive_rate", 0),
                "p05": cv.get("p05", 0),
                "p50": cv.get("p50", 0),
                "confidence": cv.get("confidence", "LOW"),
            },
            "packet_quality": _assess_packet_quality(cv),
            "transfer_risk": cv.get("transfer_risk", "?"),
            "risk_notes": _collect_risk_notes(cv),
            "reason": _explain_recommendation(cv, role),
            "is_main_bet": role in ("safest_challenger", "highest_ceiling", "champion_replacement"),
            "is_control": role in ("control_candidate", "calibration_anchor"),
            "role": role,
            "score": round(score, 2),
            "verdict": cv.get("verdict", "?"),
            "queued_at": now,
            "status": "queued",
        })

    return queue


def generate_official_memo(
    queue: list[dict[str, Any]],
    champions: dict[str, Any] | None = None,
) -> str:
    """Generate a concise memo for the official-testing shortlist.

    This memo is designed to help a human decide what to upload.
    """
    if not queue:
        return "## Official Testing Queue\n\nNo candidates ready for official testing."

    lines = [
        "## Official Testing Queue",
        "",
        f"**{len(queue)} candidate(s)** recommended for official evaluation.",
        "",
    ]

    # Summary table
    lines.append("| Rank | Candidate | Role | PnL | Sharpe | Confidence | Risk |")
    lines.append("|------|-----------|------|-----|--------|------------|------|")
    for entry in queue:
        m = entry.get("local_metrics", {})
        lines.append(
            f"| {entry['rank']} "
            f"| {entry['candidate_id'][:14]} "
            f"| {entry['role']} "
            f"| {m.get('pnl_mean', 0):.0f} "
            f"| {m.get('sharpe', 0):.2f} "
            f"| {m.get('confidence', '?')} "
            f"| {entry.get('packet_quality', '?')} |"
        )
    lines.append("")

    # Detailed entries
    for entry in queue:
        m = entry.get("local_metrics", {})
        bet_type = "Main bet" if entry.get("is_main_bet") else "Control"
        lines.append(f"### #{entry['rank']} — {entry['candidate_id']} ({bet_type})")
        lines.append("")
        lines.append(f"**Role:** {entry['role']}")
        lines.append(f"**Family:** {entry['family']} | **Parent:** {entry['parent_id']}")
        lines.append(f"**Mechanism:** {entry['mechanism']}")
        lines.append(f"**Hypothesis:** {entry['hypothesis']}")
        lines.append("")
        lines.append(
            f"**Metrics:** PnL={m.get('pnl_mean', 0):.0f}, "
            f"Sharpe={m.get('sharpe', 0):.2f}, "
            f"pos_rate={m.get('positive_rate', 0):.0%}, "
            f"p05={m.get('p05', 0):.0f}"
        )
        lines.append(f"**Packet quality:** {entry.get('packet_quality', '?')}")
        lines.append(f"**Transfer risk:** {entry.get('transfer_risk', '?')}")
        lines.append("")
        lines.append(f"**Why:** {entry['reason']}")
        lines.append("")

        risk_notes = entry.get("risk_notes", [])
        if risk_notes:
            lines.append("**Risk notes:**")
            for note in risk_notes:
                lines.append(f"  - {note}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Recommendation
    safest = [e for e in queue if e["role"] == "safest_challenger"]
    ceiling = [e for e in queue if e["role"] == "highest_ceiling"]
    controls = [e for e in queue if e.get("is_control")]

    lines.append("### Recommendation")
    lines.append("")
    if safest:
        lines.append(f"**Safest bet:** {safest[0]['candidate_id']} — lowest transfer risk, cleanest packet.")
    if ceiling:
        lines.append(f"**Highest ceiling:** {ceiling[0]['candidate_id']} — best local metrics but higher risk.")
    if controls:
        cids = ", ".join(c["candidate_id"] for c in controls)
        lines.append(f"**Control(s):** {cids} — for A/B comparison on official server.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ranking and scoring
# ---------------------------------------------------------------------------

def _rank_score(cv: dict[str, Any]) -> float:
    """Score a candidate for official queue ranking.

    Higher = more worthy of an official test slot.
    """
    score = 0.0

    # Verdict weight
    verdict = cv.get("verdict", "?")
    if verdict == "frontier_challenger":
        score += 100
    elif verdict == "escalate":
        score += 50

    # Confidence
    conf = cv.get("confidence", "LOW")
    score += {"HIGH": 30, "MEDIUM": 15, "LOW": 0}.get(conf, 0)

    # Transfer risk
    transfer = cv.get("transfer_risk", "")
    if "Low transfer risk" in transfer:
        score += 20
    elif "risk" in transfer.lower():
        score -= 15

    # Sharpe (capped at 20 for scoring)
    score += min(cv.get("sharpe", 0), 20)

    # Positive rate
    score += (cv.get("positive_rate", 0) or 0) * 30

    # p05 bonus (tail risk)
    p05 = cv.get("p05", 0) or 0
    if p05 > 0:
        score += 10
    elif p05 < -1000:
        score -= 10

    # Suspicion penalty
    susp_level = cv.get("suspicion", {}).get("suspicion_level", "clean")
    score -= {"clean": 0, "low": 5, "medium": 25, "high": 60}.get(susp_level, 0)

    # Promote gate bonus
    if cv.get("promote_recommended"):
        score += 15

    return score


def _assign_official_role(
    cv: dict[str, Any],
    rank_idx: int,
    champions: dict[str, Any] | None,
) -> str:
    """Assign an official-testing role based on candidate profile."""
    verdict = cv.get("verdict", "?")
    conf = cv.get("confidence", "LOW")
    transfer = cv.get("transfer_risk", "")

    # First candidate: safest or highest ceiling based on profile
    if rank_idx == 0:
        if conf == "HIGH" and "Low transfer risk" in transfer:
            return "safest_challenger"
        return "highest_ceiling"

    # Check if this is a champion replacement candidate
    if champions:
        active = [
            c for c in champions.get("champions", [])
            if c.get("status") == "active" and c.get("role") == "overall_champion"
        ]
        if active:
            champ = active[0]
            if cv.get("sharpe", 0) > champ.get("sharpe", 0) and cv.get("pnl_mean", 0) > champ.get("pnl_mean", 0):
                return "champion_replacement"

    # Lower-confidence or higher-risk candidates
    if conf != "HIGH" or "risk" in transfer.lower():
        return "highest_ceiling"

    # Controls
    if verdict == "keep" or rank_idx >= 2:
        return "control_candidate"

    return "safest_challenger"


def _assess_packet_quality(cv: dict[str, Any]) -> str:
    """Assess overall packet quality as a label."""
    conf = cv.get("confidence", "LOW")
    pos_rate = cv.get("positive_rate", 0) or 0
    susp = cv.get("suspicion", {}).get("suspicion_level", "clean")

    if conf == "HIGH" and pos_rate >= 0.65 and susp in ("clean", "low"):
        return "strong"
    if conf in ("HIGH", "MEDIUM") and pos_rate >= 0.50:
        return "moderate"
    return "weak"


def _collect_risk_notes(cv: dict[str, Any]) -> list[str]:
    """Collect all risk-related notes for a candidate."""
    notes: list[str] = []

    # Suspicion flags
    for flag in cv.get("suspicion", {}).get("flags", []):
        notes.append(f"Suspicion: {flag.get('flag', '?')} — {flag.get('detail', '')}")

    # Preservation violations
    for viol in cv.get("preservation_audit", {}).get("violations", []):
        notes.append(f"Preservation: [{viol.get('severity', '?')}] {viol.get('detail', '')}")

    # Transfer risk
    tr = cv.get("transfer_risk", "")
    if tr and "Low transfer risk" not in tr:
        notes.append(f"Transfer: {tr}")

    # Confidence
    if cv.get("confidence") == "LOW":
        notes.append("LOW confidence — insufficient sessions for reliable comparison.")

    return notes


def _explain_recommendation(cv: dict[str, Any], role: str) -> str:
    """Explain why this candidate is in the official queue."""
    verdict = cv.get("verdict", "?")
    pnl = cv.get("pnl_mean", 0)
    sharpe = cv.get("sharpe", 0)
    conf = cv.get("confidence", "?")
    mechanism = cv.get("attribution", {}).get("dominant_mechanism", "?")

    if role == "safest_challenger":
        return (
            f"Safest official candidate: {verdict} with {conf} confidence, "
            f"PnL={pnl:.0f}, Sharpe={sharpe:.2f}. "
            f"Clean packet, low transfer risk."
        )
    if role == "highest_ceiling":
        return (
            f"Highest upside: PnL={pnl:.0f}, Sharpe={sharpe:.2f} "
            f"via {mechanism}. Higher transfer risk but best local metrics."
        )
    if role == "champion_replacement":
        return (
            f"Beats current champion on both PnL and Sharpe. "
            f"Candidate for champion replacement."
        )
    if role == "control_candidate":
        return (
            f"Control candidate for A/B comparison on official server. "
            f"PnL={pnl:.0f}, Sharpe={sharpe:.2f}."
        )
    if role == "calibration_anchor":
        return f"Calibration anchor: known-good reproduction for baseline."

    return f"{verdict} candidate: PnL={pnl:.0f}, Sharpe={sharpe:.2f}."
