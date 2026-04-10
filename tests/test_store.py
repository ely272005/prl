"""Tests for memory/store.py."""
from __future__ import annotations

from pathlib import Path

from memory.store import PacketStore


def _make_packet(mean_pnl: float = 100.0) -> tuple[dict, dict]:
    short = {
        "candidate_id": "abc123",
        "candidate": {
            "strategy_path": "/test/strategy.py",
            "session_count": 100,
        },
        "confidence": "MEDIUM",
        "pnl": {"mean": mean_pnl, "sharpe_like": 0.5},
    }
    full = {**short, "extra": "data"}
    return short, full


class TestPacketStore:
    def test_store_and_retrieve(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        with PacketStore(db_path) as store:
            short, full = _make_packet()
            run_id = store.store(short, full)
            assert len(run_id) == 16

            retrieved = store.get_by_run_id(run_id)
            assert retrieved is not None
            assert retrieved["packet_short"]["candidate_id"] == "abc123"
            assert retrieved["packet_full"]["extra"] == "data"

    def test_get_by_candidate_id(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        with PacketStore(db_path) as store:
            short1, full1 = _make_packet(100.0)
            short2, full2 = _make_packet(200.0)
            store.store(short1, full1)
            store.store(short2, full2)

            results = store.get_by_candidate_id("abc123")
            assert len(results) == 2
            # Newest first
            assert results[0]["mean_pnl"] == 200.0

    def test_parent_lookup(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        with PacketStore(db_path) as store:
            short1, full1 = _make_packet(100.0)
            parent_id = store.store(short1, full1)

            short2, full2 = _make_packet(200.0)
            child_id = store.store(short2, full2, parent_run_id=parent_id)

            parent = store.get_parent(child_id)
            assert parent is not None
            assert parent["run_id"] == parent_id
            assert parent["mean_pnl"] == 100.0

    def test_get_latest(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        with PacketStore(db_path) as store:
            for i in range(5):
                short, full = _make_packet(float(i))
                store.store(short, full)
            results = store.get_latest(3)
            assert len(results) == 3

    def test_list_candidates(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        with PacketStore(db_path) as store:
            short, full = _make_packet()
            store.store(short, full)
            candidates = store.list_candidates()
            assert len(candidates) == 1
            assert candidates[0]["candidate_id"] == "abc123"
            assert candidates[0]["run_count"] == 1

    def test_missing_run_id(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        with PacketStore(db_path) as store:
            assert store.get_by_run_id("nonexistent") is None

    def test_nan_handling(self, tmp_path: Path):
        """NaN and inf values should be stored as null in JSON."""
        db_path = tmp_path / "test.db"
        with PacketStore(db_path) as store:
            short = {
                "candidate_id": "nan_test",
                "candidate": {"strategy_path": "/test.py", "session_count": 10},
                "confidence": "LOW",
                "pnl": {"mean": float("nan"), "sharpe_like": float("inf")},
            }
            full = {**short}
            run_id = store.store(short, full)
            retrieved = store.get_by_run_id(run_id)
            assert retrieved is not None
            # NaN/inf should be serialized as None
            assert retrieved["packet_short"]["pnl"]["mean"] is None
            assert retrieved["packet_short"]["pnl"]["sharpe_like"] is None
