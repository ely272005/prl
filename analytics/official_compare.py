"""Official-vs-local calibration comparison.

Accepts an official IMC result JSON and a local Packet Short,
produces a compact comparison dict for calibration analysis.

Official JSON structure (observed from IMC Prosperity 4):
  - profit: float (total PnL)
  - activitiesLog: semicolon-delimited string with per-tick order book + PnL
  - graphLog: semicolon-delimited "timestamp;value" PnL trajectory
  - status: "FINISHED"
  - round: int
"""
from __future__ import annotations

from typing import Any, Optional


def parse_official_result(official: dict[str, Any]) -> dict[str, Any]:
    """Extract structured data from raw official JSON.

    Returns dict with:
      - total_pnl
      - per_product_pnl (if derivable from activitiesLog)
      - official_ticks (from graphLog or activitiesLog)
      - status
    """
    total_pnl = official.get("profit", 0.0)
    status = official.get("status", "UNKNOWN")

    # Parse per-product PnL from activitiesLog
    per_product_pnl: dict[str, float] = {}
    official_ticks = 0

    activities_raw = official.get("activitiesLog", "")
    if activities_raw:
        lines = activities_raw.strip().split("\n")
        if len(lines) > 1:
            header = lines[0].split(";")
            prod_idx = _col_index(header, "product")
            pnl_idx = _col_index(header, "profit_and_loss")
            ts_idx = _col_index(header, "timestamp")

            if prod_idx is not None and pnl_idx is not None:
                # Track last PnL per product (final tick)
                last_pnl: dict[str, float] = {}
                timestamps: set[int] = set()
                for line in lines[1:]:
                    parts = line.split(";")
                    if len(parts) <= max(
                        x for x in [prod_idx, pnl_idx, ts_idx] if x is not None
                    ):
                        continue
                    product = parts[prod_idx]
                    pnl = float(parts[pnl_idx])
                    last_pnl[product] = pnl
                    if ts_idx is not None:
                        timestamps.add(int(parts[ts_idx]))

                per_product_pnl = last_pnl
                official_ticks = len(timestamps)

    # Fallback tick count from graphLog
    if official_ticks == 0:
        graph_raw = official.get("graphLog", "")
        if graph_raw:
            graph_lines = graph_raw.strip().split("\n")
            # Subtract header
            official_ticks = max(0, len(graph_lines) - 1)

    return {
        "total_pnl": total_pnl,
        "per_product_pnl": per_product_pnl,
        "official_ticks": official_ticks,
        "status": status,
    }


def compare_official_vs_local(
    official: dict[str, Any],
    packet_short: dict[str, Any],
) -> dict[str, Any]:
    """Compare an official IMC result against a local Packet Short.

    Args:
        official: Raw official JSON dict (with profit, activitiesLog, etc.)
        packet_short: Local Packet Short dict.

    Returns:
        Comparison dict with ratios, sign-flip detection, and warnings.
    """
    parsed = parse_official_result(official)
    official_pnl = parsed["total_pnl"]
    official_ticks = parsed["official_ticks"]
    official_products = parsed["per_product_pnl"]

    local_mean = packet_short.get("pnl", {}).get("mean", 0.0)
    local_std = packet_short.get("pnl", {}).get("std", 0.0)
    local_ticks = (packet_short.get("scale") or {}).get("ticks_per_session")

    warnings: list[str] = []

    # --- Time-normalized comparison ---
    time_ratio: Optional[float] = None
    normalized_local_pnl: Optional[float] = None

    if local_ticks and official_ticks and local_ticks > 0:
        time_ratio = local_ticks / official_ticks
        normalized_local_pnl = local_mean / time_ratio
    else:
        warnings.append(
            "Cannot time-normalize: missing ticks_per_session or official_ticks."
        )

    # --- Raw ratio ---
    raw_ratio: Optional[float] = None
    if official_pnl != 0:
        raw_ratio = local_mean / official_pnl
    else:
        warnings.append("Official PnL is zero — ratio undefined.")

    # --- Normalized ratio ---
    normalized_ratio: Optional[float] = None
    if normalized_local_pnl is not None and official_pnl != 0:
        normalized_ratio = normalized_local_pnl / official_pnl

    # --- Per-product comparison ---
    per_product: dict[str, dict[str, Any]] = {}
    local_per_product = packet_short.get("per_product", {})
    product_map = {
        "EMERALDS": "emerald",
        "TOMATOES": "tomato",
    }

    for official_name, local_key in product_map.items():
        off_pnl = official_products.get(official_name)
        loc_pnl = (local_per_product.get(local_key) or {}).get("mean")

        if off_pnl is None or loc_pnl is None:
            continue

        normalized_loc: Optional[float] = None
        if time_ratio and time_ratio > 0:
            normalized_loc = loc_pnl / time_ratio

        sign_flip = False
        if off_pnl != 0 and normalized_loc is not None:
            sign_flip = (off_pnl > 0) != (normalized_loc > 0)

        if sign_flip:
            warnings.append(
                f"Official/local sign flip on {official_name}: "
                f"official={off_pnl:.1f}, local(normalized)={normalized_loc:.1f}."
            )

        entry: dict[str, Any] = {
            "official_pnl": off_pnl,
            "local_mean_pnl": loc_pnl,
            "normalized_local_pnl": normalized_loc,
            "sign_flip": sign_flip,
        }
        if off_pnl != 0 and normalized_loc is not None:
            entry["normalized_ratio"] = normalized_loc / off_pnl
        per_product[official_name] = entry

    # --- Standard warnings ---
    warnings.append(
        "Official result is a single run; local is a Monte Carlo ensemble."
    )
    if raw_ratio is not None and abs(raw_ratio) > 3.0:
        warnings.append(
            f"Raw local/official ratio is {raw_ratio:.1f}x — "
            "absolute PnL not directly comparable without time normalization."
        )

    return {
        "official_pnl": official_pnl,
        "local_mean_pnl": local_mean,
        "local_std_pnl": local_std,
        "official_ticks": official_ticks,
        "local_ticks_per_session": local_ticks,
        "time_ratio": time_ratio,
        "normalized_local_pnl": normalized_local_pnl,
        "raw_ratio": raw_ratio,
        "normalized_ratio": normalized_ratio,
        "per_product": per_product,
        "warnings": warnings,
    }


def _col_index(header: list[str], name: str) -> Optional[int]:
    """Find column index by name, case-insensitive."""
    for i, col in enumerate(header):
        if col.strip().lower() == name.lower():
            return i
    return None
