"""Tests for run_observability.py — dashboard fallback session count."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

# Ensure run_observability is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_observability import load_or_build_dashboard


class TestLoadOrBuildDashboard:
    def test_fallback_uses_session_summary_count(self, tmp_path):
        """When no dashboard.json exists, session_count comes from session_summary.csv rows."""
        # Write session_summary.csv with 50 sessions (not sample dirs)
        summary = tmp_path / "session_summary.csv"
        with summary.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["session_id", "total_pnl"])
            writer.writeheader()
            for i in range(50):
                writer.writerow({"session_id": i, "total_pnl": 100.0 + i})

        # Only create 3 sample session dirs (fewer than the 50 total)
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        for i in range(3):
            (sessions_dir / f"session_{i:05d}").mkdir()

        # No dashboard.json, no upstream build_dashboard → falls back to stub
        result = load_or_build_dashboard(tmp_path, None)

        # The stub should carry the correct session count (50), not the sample dir count (3)
        assert result["meta"]["sessionCount"] == 50

    def test_fallback_no_summary_csv_warns(self, tmp_path, capsys):
        """When neither dashboard.json nor session_summary.csv exists, warns about undercount."""
        # Only sample dirs, no session_summary.csv
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        for i in range(3):
            (sessions_dir / f"session_{i:05d}").mkdir()

        result = load_or_build_dashboard(tmp_path, None)

        # Should use the sample dir count as last resort
        assert result["meta"]["sessionCount"] == 3

        # Should have emitted a warning to stderr
        captured = capsys.readouterr()
        assert "session_summary.csv not found" in captured.err
        assert "undercount" in captured.err

    def test_fallback_empty_dir(self, tmp_path):
        """When output dir has nothing, session_count is 0."""
        result = load_or_build_dashboard(tmp_path, None)
        assert result["meta"]["sessionCount"] == 0

    def test_dashboard_json_preferred(self, tmp_path):
        """When dashboard.json exists, it's loaded directly (no fallback)."""
        import json
        dashboard = {"kind": "monte_carlo_dashboard", "meta": {"sessionCount": 999}}
        (tmp_path / "dashboard.json").write_text(json.dumps(dashboard))

        result = load_or_build_dashboard(tmp_path, None)
        assert result["meta"]["sessionCount"] == 999

    def test_explicit_dashboard_path(self, tmp_path):
        """When --dashboard path is given, it takes precedence."""
        import json
        dashboard = {"kind": "test", "meta": {"sessionCount": 42}}
        explicit = tmp_path / "custom_dashboard.json"
        explicit.write_text(json.dumps(dashboard))

        result = load_or_build_dashboard(tmp_path, explicit)
        assert result["meta"]["sessionCount"] == 42
