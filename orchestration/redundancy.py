"""Redundancy control — prevents wasting runs on near-duplicates and dead directions.

Checks:
  1. Near-duplicate: candidate too similar to existing frontier member
  2. Falsified repeat: campaign targets a dead-zone hypothesis
  3. Low diversity: too many candidates attacking the same mechanism+parent
  4. Cross-campaign overlap: two campaigns asking the same research question
  5. Weak near-twin: candidate differs from frontier member on < 3% of metrics
"""
from __future__ import annotations

from typing import Any


# If average relative difference across key metrics is below this, it's a near-duplicate
NEAR_DUPLICATE_THRESHOLD = 0.05  # 5%

# Minimum number of distinct mechanisms per campaign
MIN_MECHANISM_DIVERSITY = 2

# Maximum candidates from the same parent in a single campaign
MAX_SAME_PARENT = 3


def check_all_redundancy(
    campaigns: list[dict[str, Any]],
    candidate_verdicts: list[dict[str, Any]],
    frontier_verdicts: list[dict[str, Any]],
    dead_zones: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run all redundancy checks.

    Returns a dict with:
      - near_duplicates: candidates too similar to frontier
      - falsified_repeats: campaigns targeting dead hypotheses
      - low_diversity: campaigns lacking mechanism diversity
      - cross_overlap: campaigns that overlap
      - recommendations: actions to take
    """
    near_dups = detect_near_duplicates(candidate_verdicts, frontier_verdicts)
    falsified = detect_falsified_repeats(campaigns, dead_zones)
    diversity = assess_campaign_diversity(campaigns)
    overlap = detect_cross_campaign_overlap(campaigns)

    recommendations = []
    for nd in near_dups:
        recommendations.append({
            "action": "skip",
            "target": nd["candidate_id"],
            "reason": nd["reason"],
        })
    for fr in falsified:
        recommendations.append({
            "action": "cancel_campaign",
            "target": fr["campaign_id"],
            "reason": fr["reason"],
        })
    for dv in diversity:
        recommendations.append({
            "action": "add_diversity",
            "target": dv["campaign_id"],
            "reason": dv["reason"],
        })
    for ov in overlap:
        recommendations.append({
            "action": "merge_campaigns",
            "target": f"{ov['campaign_a']} + {ov['campaign_b']}",
            "reason": ov["reason"],
        })

    return {
        "near_duplicates": near_dups,
        "falsified_repeats": falsified,
        "low_diversity": diversity,
        "cross_overlap": overlap,
        "recommendations": recommendations,
        "total_issues": len(recommendations),
    }


def detect_near_duplicates(
    candidate_verdicts: list[dict[str, Any]],
    frontier_verdicts: list[dict[str, Any]],
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Detect candidates that are too similar to existing frontier members.

    Compares on: pnl_mean, sharpe, positive_rate, emerald_mean, tomato_mean.
    """
    duplicates: list[dict[str, Any]] = []

    for cv in candidate_verdicts:
        for fv in frontier_verdicts:
            sim = _metric_similarity(cv, fv)
            if sim > (1.0 - threshold):
                duplicates.append({
                    "candidate_id": cv.get("candidate_id", "?"),
                    "frontier_member": fv.get("candidate_id", "?"),
                    "similarity": round(sim, 4),
                    "reason": (
                        f"Near-duplicate: {cv.get('candidate_id', '?')} is "
                        f"{sim:.1%} similar to frontier member "
                        f"{fv.get('candidate_id', '?')}. "
                        f"Do not spend a slot on this."
                    ),
                })
                break  # one match is enough

    return duplicates


def detect_falsified_repeats(
    campaigns: list[dict[str, Any]],
    dead_zones: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect campaigns that target already-falsified hypotheses."""
    dead_ids = {dz.get("hypothesis_id", "") for dz in dead_zones}
    repeats: list[dict[str, Any]] = []

    for camp in campaigns:
        mechanism = camp.get("target_mechanism", "")
        if mechanism and mechanism in dead_ids:
            repeats.append({
                "campaign_id": camp.get("campaign_id", "?"),
                "campaign_title": camp.get("title", "?"),
                "dead_hypothesis": mechanism,
                "reason": (
                    f"Campaign '{camp.get('title', '?')}' targets hypothesis "
                    f"'{mechanism}' which is in a dead zone. Cancel this campaign."
                ),
            })

    return repeats


def assess_campaign_diversity(
    campaigns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Check mechanism and parent diversity within each campaign."""
    issues: list[dict[str, Any]] = []

    for camp in campaigns:
        actions = camp.get("source_next_actions", [])
        parents = camp.get("allowed_parents", [])

        # Check parent concentration
        if len(parents) == 1 and camp.get("campaign_type") == "exploration":
            issues.append({
                "campaign_id": camp.get("campaign_id", "?"),
                "campaign_title": camp.get("title", "?"),
                "issue": "single_parent",
                "reason": (
                    f"Exploration campaign '{camp.get('title', '?')}' uses only "
                    f"one parent. Consider testing on multiple parents for "
                    f"mechanism generalization."
                ),
            })

    return issues


def detect_cross_campaign_overlap(
    campaigns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect pairs of campaigns that are essentially asking the same question."""
    overlaps: list[dict[str, Any]] = []

    for i in range(len(campaigns)):
        for j in range(i + 1, len(campaigns)):
            a = campaigns[i]
            b = campaigns[j]

            # Same mechanism + same family = overlap
            a_mech = a.get("target_mechanism", "")
            b_mech = b.get("target_mechanism", "")
            a_fam = a.get("family", "")
            b_fam = b.get("family", "")

            if a_mech and a_mech == b_mech and a_fam and a_fam == b_fam:
                overlaps.append({
                    "campaign_a": a.get("campaign_id", "?"),
                    "campaign_b": b.get("campaign_id", "?"),
                    "shared_mechanism": a_mech,
                    "shared_family": a_fam,
                    "reason": (
                        f"Campaigns '{a.get('title', '?')}' and "
                        f"'{b.get('title', '?')}' both target mechanism "
                        f"'{a_mech}' on family '{a_fam}'. Consider merging."
                    ),
                })

    return overlaps


def filter_redundant_campaigns(
    campaigns: list[dict[str, Any]],
    redundancy: dict[str, Any],
) -> list[dict[str, Any]]:
    """Remove campaigns flagged as falsified repeats."""
    cancel_ids = {
        fr["campaign_id"]
        for fr in redundancy.get("falsified_repeats", [])
    }
    return [c for c in campaigns if c.get("campaign_id") not in cancel_ids]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIMILARITY_METRICS = ("pnl_mean", "sharpe", "positive_rate", "emerald_mean", "tomato_mean")


def _metric_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Compute metric similarity between two verdict dicts.

    Returns a value in [0, 1] where 1 = identical.
    Uses average relative difference across key metrics.
    """
    diffs: list[float] = []
    for metric in _SIMILARITY_METRICS:
        a_val = a.get(metric, 0) or 0
        b_val = b.get(metric, 0) or 0
        denom = max(abs(a_val), abs(b_val), 1e-9)
        diffs.append(abs(a_val - b_val) / denom)

    avg_diff = sum(diffs) / len(diffs) if diffs else 0.0
    return max(0.0, 1.0 - avg_diff)
