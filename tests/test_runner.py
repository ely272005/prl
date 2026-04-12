"""Tests for ProbeRunner and report generation."""
import json
import math
import pytest
from pathlib import Path

from mechanics.runner import ProbeRunner
from mechanics.report import build_json_report, build_markdown_report, write_json_report, write_markdown_report
from mechanics.probe_spec import ProbeResult


@pytest.fixture
def runner(output_dir):
    """ProbeRunner pointed at the synthetic output_dir."""
    return ProbeRunner([output_dir])


class TestProbeRunner:
    def test_loads_sessions(self, runner):
        summary = runner.summary()
        assert summary["total_sessions"] == 2  # conftest creates 2 sessions
        assert len(summary["available_probes"]) >= 10

    def test_run_all(self, runner):
        results = runner.run_all()
        assert len(results) >= 20  # 10 probes x 2 products
        for r in results:
            assert isinstance(r, ProbeResult)

    def test_run_single_probe(self, runner):
        r = runner.run_probe("pf01_spread_vs_maker_edge", "EMERALDS")
        assert isinstance(r, ProbeResult)
        assert r.probe_id == "pf01_spread_vs_maker_edge"
        assert r.product == "EMERALDS"

    def test_run_family(self, runner):
        results = runner.run_family("passive_fill")
        assert len(results) == 6  # 3 probes x 2 products
        assert all(r.family == "passive_fill" for r in results)

    def test_run_single_product(self, runner):
        results = runner.run_all(products=["TOMATOES"])
        assert all(r.product == "TOMATOES" for r in results)

    def test_run_family_single_product(self, runner):
        results = runner.run_family("taking", products=["EMERALDS"])
        assert len(results) == 3  # 3 taking probes x 1 product
        assert all(r.product == "EMERALDS" for r in results)


class TestReportGeneration:
    def test_json_report_structure(self, runner):
        results = runner.run_all()
        report = build_json_report(results, runner.summary())
        assert "generated_at" in report
        assert "summary" in report
        assert report["summary"]["total_probes_run"] == len(results)
        assert "results_by_family" in report
        assert "passive_fill" in report["results_by_family"]

    def test_json_report_serializable(self, runner):
        results = runner.run_all()
        report = build_json_report(results, runner.summary())
        # Should not raise
        serialized = json.dumps(report, default=str)
        assert len(serialized) > 100

    def test_markdown_report_not_empty(self, runner):
        results = runner.run_all()
        md = build_markdown_report(results, runner.summary())
        assert "# Mechanics Probe Report" in md
        assert "Verdict Summary" in md
        assert len(md) > 500

    def test_write_json_report(self, runner, tmp_path):
        results = runner.run_all()
        path = tmp_path / "test_report.json"
        write_json_report(results, path, runner.summary())
        assert path.exists()
        with path.open() as f:
            data = json.load(f)
        assert data["summary"]["total_probes_run"] == len(results)

    def test_write_markdown_report(self, runner, tmp_path):
        results = runner.run_all()
        path = tmp_path / "test_report.md"
        write_markdown_report(results, path, runner.summary())
        assert path.exists()
        content = path.read_text()
        assert "# Mechanics Probe Report" in content


class TestNaNHandling:
    def test_nan_in_metrics_serializes(self):
        """NaN values should become null in JSON output."""
        r = ProbeResult(
            probe_id="test", family="test", title="T", hypothesis="H",
            product="EMERALDS", dataset="d", sample_size={},
            metrics={"val": math.nan, "nested": {"x": math.nan}},
            verdict="inconclusive", confidence="low", detail="d",
        )
        report = build_json_report([r])
        serialized = json.dumps(report, default=str)
        parsed = json.loads(serialized)
        # NaN should have been replaced with None -> null in JSON
        result_metrics = parsed["results_by_family"]["test"][0]["metrics"]
        assert result_metrics["val"] is None
        assert result_metrics["nested"]["x"] is None
