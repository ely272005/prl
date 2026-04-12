"""Budget-aware planning and exploit/explore allocation.

Manages:
  - exploit vs explore ratio (default 70/30)
  - per-campaign budget sizing
  - total local run budget and official test budget
  - campaign prioritization within each category

The allocator prevents the loop from becoming:
  - only tiny descendants of the champion (all exploit)
  - only wild exploration with no convergence (all explore)
"""
from __future__ import annotations

from typing import Any

from orchestration.campaigns import CAMPAIGN_TYPE_PROPERTIES


# Default budget limits
DEFAULT_BUDGET = {
    "total_local_candidates": 20,
    "total_official_tests": 3,
    "max_campaigns": 6,
}

# Campaign type default sizes
CAMPAIGN_TYPE_SIZES = {
    "confirmation": 3,
    "exploration": 6,
    "official_gate": 2,
    "calibration": 4,
    "champion_defense": 2,
    "control_batch": 3,
}

# Default exploit/explore ratio
DEFAULT_EXPLOIT_RATIO = 0.70


def allocate_budget(
    campaigns: list[dict[str, Any]],
    budget: dict[str, int] | None = None,
    exploit_ratio: float = DEFAULT_EXPLOIT_RATIO,
) -> dict[str, Any]:
    """Allocate budget across campaigns.

    Returns an allocation plan with:
      - campaigns: updated with allocated_candidates
      - exploit_budget / explore_budget
      - official_budget
      - overflow notes
    """
    budget = budget or dict(DEFAULT_BUDGET)
    total_local = budget.get("total_local_candidates", 20)
    total_official = budget.get("total_official_tests", 3)
    max_campaigns = budget.get("max_campaigns", 6)

    # Classify campaigns
    exploit_campaigns, explore_campaigns = split_exploit_explore(campaigns)

    # Calculate budget splits
    exploit_budget = int(total_local * exploit_ratio)
    explore_budget = total_local - exploit_budget

    # Size and allocate within each category
    exploit_allocated = _allocate_category(exploit_campaigns, exploit_budget)
    explore_allocated = _allocate_category(explore_campaigns, explore_budget)

    # Overflow: if one category has leftover, give to the other
    exploit_used = sum(c.get("allocated_candidates", 0) for c in exploit_allocated)
    explore_used = sum(c.get("allocated_candidates", 0) for c in explore_allocated)
    overflow_notes: list[str] = []

    exploit_remaining = exploit_budget - exploit_used
    explore_remaining = explore_budget - explore_used

    if exploit_remaining > 0 and explore_allocated:
        # Give exploit overflow to explore
        _distribute_overflow(explore_allocated, exploit_remaining)
        overflow_notes.append(
            f"Reallocated {exploit_remaining} exploit slots to explore campaigns."
        )
    elif explore_remaining > 0 and exploit_allocated:
        _distribute_overflow(exploit_allocated, explore_remaining)
        overflow_notes.append(
            f"Reallocated {explore_remaining} explore slots to exploit campaigns."
        )

    # Trim to max campaigns
    all_allocated = exploit_allocated + explore_allocated
    all_allocated.sort(key=lambda c: _priority_order(c.get("priority", "low")))

    if len(all_allocated) > max_campaigns:
        trimmed = all_allocated[:max_campaigns]
        dropped = all_allocated[max_campaigns:]
        overflow_notes.append(
            f"Dropped {len(dropped)} lowest-priority campaign(s) to fit max_campaigns={max_campaigns}."
        )
        all_allocated = trimmed

    # Compute totals
    total_allocated = sum(c.get("allocated_candidates", 0) for c in all_allocated)

    return {
        "campaigns": all_allocated,
        "budget": budget,
        "exploit_ratio": exploit_ratio,
        "exploit_budget": exploit_budget,
        "explore_budget": explore_budget,
        "exploit_campaigns": len(exploit_allocated),
        "explore_campaigns": len(explore_allocated),
        "total_allocated": total_allocated,
        "total_local": total_local,
        "official_budget": total_official,
        "overflow_notes": overflow_notes,
    }


def split_exploit_explore(
    campaigns: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split campaigns into exploit and explore categories."""
    exploit = []
    explore = []
    for c in campaigns:
        ee = c.get("exploit_explore", "exploit")
        if ee == "explore":
            explore.append(c)
        else:
            exploit.append(c)
    return exploit, explore


def size_campaign(campaign_type: str, priority: str = "medium") -> int:
    """Determine default candidate count for a campaign type."""
    base = CAMPAIGN_TYPE_SIZES.get(campaign_type, 4)
    if priority == "high":
        return base + 1
    if priority == "low":
        return max(base - 1, 1)
    return base


# ---------------------------------------------------------------------------
# Internal allocation
# ---------------------------------------------------------------------------

def _allocate_category(
    campaigns: list[dict[str, Any]],
    budget: int,
) -> list[dict[str, Any]]:
    """Allocate budget within a category, prioritizing high-priority campaigns."""
    # Sort by priority
    sorted_camps = sorted(
        campaigns,
        key=lambda c: _priority_order(c.get("priority", "low")),
    )

    remaining = budget
    allocated = []

    for camp in sorted_camps:
        camp = dict(camp)  # copy
        ct = camp.get("campaign_type", "exploration")
        desired = size_campaign(ct, camp.get("priority", "medium"))

        # Respect campaign's own budget if set
        camp_max = camp.get("budget", {}).get("max_candidates", desired)
        desired = min(desired, camp_max)

        if remaining <= 0:
            camp["allocated_candidates"] = 0
            camp["allocation_note"] = "Budget exhausted — campaign deferred."
        elif remaining < desired:
            camp["allocated_candidates"] = remaining
            camp["allocation_note"] = f"Partial allocation: {remaining}/{desired} candidates."
            remaining = 0
        else:
            camp["allocated_candidates"] = desired
            camp["allocation_note"] = f"Full allocation: {desired} candidates."
            remaining -= desired

        allocated.append(camp)

    return allocated


def _distribute_overflow(
    campaigns: list[dict[str, Any]],
    extra: int,
) -> None:
    """Distribute overflow budget to campaigns that can use more."""
    for camp in campaigns:
        if extra <= 0:
            break
        allocated = camp.get("allocated_candidates", 0)
        desired = camp.get("budget", {}).get("max_candidates", 10)
        room = desired - allocated
        if room > 0:
            give = min(room, extra)
            camp["allocated_candidates"] = allocated + give
            extra -= give


def _priority_order(priority: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(priority, 9)
