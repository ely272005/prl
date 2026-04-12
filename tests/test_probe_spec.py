"""Tests for probe spec, result, and registry."""
import pytest

from mechanics.probe_spec import (
    ProbeSpec,
    ProbeResult,
    Probe,
    register_probe,
    get_probe,
    list_probes,
    list_families,
    _REGISTRY,
)


class TestProbeSpec:
    def test_spec_creation(self):
        spec = ProbeSpec(
            probe_id="test_probe",
            family="test",
            title="A test probe",
            hypothesis="This is a test",
            required_data=("traces",),
            metrics_produced=("foo",),
        )
        assert spec.probe_id == "test_probe"
        assert spec.family == "test"
        assert spec.required_data == ("traces",)

    def test_spec_is_frozen(self):
        spec = ProbeSpec("id", "fam", "title", "hyp", ("traces",), ("m",))
        with pytest.raises(AttributeError):
            spec.probe_id = "changed"


class TestProbeResult:
    def test_result_to_dict(self):
        r = ProbeResult(
            probe_id="test",
            family="test",
            title="Test",
            hypothesis="hyp",
            product="EMERALDS",
            dataset="test_data",
            sample_size={"sessions": 2},
            metrics={"foo": 1.0},
            verdict="supported",
            confidence="high",
            detail="It works.",
            warnings=["small sample"],
        )
        d = r.to_dict()
        assert d["probe_id"] == "test"
        assert d["metrics"]["foo"] == 1.0
        assert d["warnings"] == ["small sample"]

    def test_result_default_warnings(self):
        r = ProbeResult(
            probe_id="t", family="t", title="t", hypothesis="h",
            product="P", dataset="d", sample_size={}, metrics={},
            verdict="inconclusive", confidence="low", detail="d",
        )
        assert r.warnings == []


class TestRegistry:
    def test_registered_probes_exist(self):
        """After importing mechanics.probes, the registry should be populated."""
        import mechanics.probes  # noqa: F401
        probes = list_probes()
        assert len(probes) >= 10  # 3 + 3 + 2 + 2 = 10

    def test_list_families(self):
        import mechanics.probes  # noqa: F401
        families = list_families()
        assert "passive_fill" in families
        assert "taking" in families
        assert "inventory" in families
        assert "danger_zone" in families

    def test_get_probe(self):
        import mechanics.probes  # noqa: F401
        probe = get_probe("pf01_spread_vs_maker_edge")
        assert isinstance(probe, Probe)
        assert probe.spec.probe_id == "pf01_spread_vs_maker_edge"

    def test_get_unknown_probe_raises(self):
        with pytest.raises(KeyError, match="Unknown probe"):
            get_probe("nonexistent_probe_999")

    def test_list_probes_by_family(self):
        import mechanics.probes  # noqa: F401
        passive = list_probes(family="passive_fill")
        assert len(passive) == 3
        assert all(s.family == "passive_fill" for s in passive)

    def test_register_probe_validates(self):
        """register_probe should reject classes without a ProbeSpec."""
        with pytest.raises(TypeError, match="must define a ProbeSpec"):
            @register_probe
            class BadProbe(Probe):
                def run(self, session_ledgers, product, dataset_label=""):
                    pass
