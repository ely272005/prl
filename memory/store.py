"""Research Packet Store — persistent storage for research packets.

SQLite-backed store indexed by candidate_id and run_id.
Supports parent lookup for comparisons across runs.

Schema:
  packets table:
    - candidate_id TEXT (content hash of strategy+seed+sessions)
    - run_id TEXT (unique per store call, auto-generated)
    - parent_run_id TEXT (optional, for linking iterations)
    - strategy_path TEXT
    - created_at TEXT (ISO 8601)
    - confidence TEXT
    - mean_pnl REAL
    - sharpe_like REAL
    - session_count INTEGER
    - packet_short TEXT (JSON)
    - packet_full TEXT (JSON)
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


DEFAULT_DB_PATH = Path.home() / ".prosperity_machine" / "research_packets.db"


class PacketStore:
    """SQLite-backed research packet store."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS packets (
                run_id TEXT PRIMARY KEY,
                candidate_id TEXT NOT NULL,
                parent_run_id TEXT,
                strategy_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                confidence TEXT NOT NULL,
                mean_pnl REAL NOT NULL,
                sharpe_like REAL NOT NULL,
                session_count INTEGER NOT NULL,
                packet_short TEXT NOT NULL,
                packet_full TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_candidate_id ON packets(candidate_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON packets(created_at)
        """)
        self._conn.commit()

    def store(
        self,
        packet_short: dict[str, Any],
        packet_full: dict[str, Any],
        parent_run_id: Optional[str] = None,
    ) -> str:
        """Store a research packet and return the generated run_id."""
        run_id = uuid.uuid4().hex[:16]
        candidate_id = packet_short.get("candidate_id", "unknown")
        strategy_path = packet_short.get("candidate", {}).get("strategy_path", "")
        confidence = packet_short.get("confidence", "LOW")
        mean_pnl = _sanitize_float(packet_short.get("pnl", {}).get("mean", 0.0))
        sharpe_like = _sanitize_float(packet_short.get("pnl", {}).get("sharpe_like", 0.0))
        session_count = packet_short.get("candidate", {}).get("session_count", 0)
        created_at = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            """
            INSERT INTO packets
                (run_id, candidate_id, parent_run_id, strategy_path, created_at,
                 confidence, mean_pnl, sharpe_like, session_count,
                 packet_short, packet_full)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                candidate_id,
                parent_run_id,
                strategy_path,
                created_at,
                confidence,
                mean_pnl,
                sharpe_like,
                session_count,
                _dumps(packet_short),
                _dumps(packet_full),
            ),
        )
        self._conn.commit()
        return run_id

    def get_by_run_id(self, run_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a packet by run_id."""
        row = self._conn.execute(
            "SELECT * FROM packets WHERE run_id = ?", (run_id,)
        ).fetchone()
        return _row_to_dict(row) if row else None

    def get_by_candidate_id(self, candidate_id: str) -> list[dict[str, Any]]:
        """Retrieve all packets for a candidate, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM packets WHERE candidate_id = ? ORDER BY created_at DESC",
            (candidate_id,),
        ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def get_latest(self, limit: int = 10) -> list[dict[str, Any]]:
        """Retrieve the most recent packets."""
        rows = self._conn.execute(
            "SELECT * FROM packets ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def get_parent(self, run_id: str) -> Optional[dict[str, Any]]:
        """Follow parent_run_id link to retrieve the parent packet."""
        row = self._conn.execute(
            "SELECT parent_run_id FROM packets WHERE run_id = ?", (run_id,)
        ).fetchone()
        if not row or not row["parent_run_id"]:
            return None
        return self.get_by_run_id(row["parent_run_id"])

    def list_candidates(self) -> list[dict[str, Any]]:
        """List distinct candidates with their latest run info."""
        rows = self._conn.execute("""
            SELECT candidate_id, strategy_path, MAX(created_at) as latest,
                   COUNT(*) as run_count
            FROM packets
            GROUP BY candidate_id
            ORDER BY latest DESC
        """).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _sanitize_value(obj: Any) -> Any:
    """Recursively replace NaN/inf with None in nested dicts/lists."""
    if isinstance(obj, float):
        if obj != obj or abs(obj) == float("inf"):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_value(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _dumps(obj: Any) -> str:
    """JSON-serialize with NaN/inf -> null."""
    return json.dumps(_sanitize_value(obj))


def _sanitize_float(value: Any) -> float:
    """Convert NaN/inf to 0.0 for SQLite REAL columns."""
    if isinstance(value, float) and (value != value or abs(value) == float("inf")):
        return 0.0
    return float(value)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["packet_short"] = json.loads(d["packet_short"])
    d["packet_full"] = json.loads(d["packet_full"])
    return d


def _json_default(obj: Any) -> Any:
    """Handle non-serializable types in packet JSON."""
    if isinstance(obj, float) and (obj != obj):  # NaN check
        return None
    if isinstance(obj, float) and abs(obj) == float("inf"):
        return None
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
