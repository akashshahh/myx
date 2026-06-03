"""Tests for data/degradation.py."""
from __future__ import annotations

import numpy as np
import pytest

from data.degradation import (
    ISSUE_DIMENSIONS,
    NUM_DIMENSIONS,
    DegradationSpec,
    apply_chain,
    build_plugin,
    chain_to_label_vec,
    degrade,
    random_degradation_chain,
    sample_degradation,
    specs_to_jsonable,
)


SR = 44100
DURATION_S = 2.0


@pytest.fixture
def stereo_sine() -> np.ndarray:
    """2-second 440 Hz stereo sine at 44.1 kHz, float32, shape (2, N)."""
    t = np.linspace(0, DURATION_S, int(SR * DURATION_S), endpoint=False)
    mono = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return np.stack([mono, mono], axis=0)


@pytest.fixture
def mono_sine() -> np.ndarray:
    t = np.linspace(0, DURATION_S, int(SR * DURATION_S), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_issue_dimensions_match_spec():
    expected = [
        "low_excess",
        "low_mid_mud",
        "mid_balance",
        "presence_lack",
        "harshness",
        "over_compression",
        "loudness_deficit",
        "dynamic_range_issue",
    ]
    assert ISSUE_DIMENSIONS == expected
    assert NUM_DIMENSIONS == 8


def test_degradation_spec_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown degradation"):
        DegradationSpec(name="nonsense", params={}, severity=0.5)


@pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0, -1.0])
def test_degradation_spec_rejects_out_of_range_severity(bad):
    with pytest.raises(ValueError, match="severity"):
        DegradationSpec(name="low_excess", params={}, severity=bad)


# ---------------------------------------------------------------------------
# Per-degradation: sample, build, apply yields finite same-shape audio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ISSUE_DIMENSIONS)
def test_each_degradation_renders_cleanly(name, stereo_sine):
    rng = np.random.default_rng(42)
    spec = sample_degradation(name, rng)
    assert spec.name == name
    assert 0.0 <= spec.severity <= 1.0
    plugin = build_plugin(spec)
    assert plugin is not None
    out = apply_chain(stereo_sine, SR, [spec])
    assert out.shape == stereo_sine.shape
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))


def test_apply_chain_handles_mono(mono_sine):
    rng = np.random.default_rng(0)
    chain = [sample_degradation("low_excess", rng)]
    out = apply_chain(mono_sine, SR, chain)
    assert out.shape == mono_sine.shape
    assert np.all(np.isfinite(out))


def test_apply_chain_empty_returns_copy(stereo_sine):
    out = apply_chain(stereo_sine, SR, [])
    assert out.shape == stereo_sine.shape
    np.testing.assert_array_equal(out, stereo_sine)
    assert out is not stereo_sine  # actually a copy


# ---------------------------------------------------------------------------
# Label vector
# ---------------------------------------------------------------------------


def test_label_vec_shape_and_range():
    rng = np.random.default_rng(1)
    chain = random_degradation_chain(rng)
    vec = chain_to_label_vec(chain)
    assert vec.shape == (NUM_DIMENSIONS,)
    assert vec.dtype == np.float32
    assert np.all(vec >= 0.0)
    assert np.all(vec <= 1.0)


def test_label_vec_marks_exactly_named_dimensions():
    rng = np.random.default_rng(7)
    chain = [
        sample_degradation("harshness", rng),
        sample_degradation("over_compression", rng),
    ]
    vec = chain_to_label_vec(chain)
    nonzero_idx = np.where(vec > 0)[0]
    expected_idx = {ISSUE_DIMENSIONS.index(n) for n in ("harshness", "over_compression")}
    assert set(nonzero_idx.tolist()) == expected_idx


def test_label_vec_max_pools_duplicates():
    """If two specs share a name, the larger severity wins."""
    low = DegradationSpec(
        name="low_excess", params={"freq": 80.0, "gain_db": 3.0, "q": 0.7071}, severity=0.2
    )
    high = DegradationSpec(
        name="low_excess", params={"freq": 80.0, "gain_db": 9.0, "q": 0.7071}, severity=0.9
    )
    vec = chain_to_label_vec([low, high])
    assert vec[ISSUE_DIMENSIONS.index("low_excess")] == pytest.approx(0.9, abs=1e-6)


# ---------------------------------------------------------------------------
# Random chain semantics
# ---------------------------------------------------------------------------


def test_random_chain_is_deterministic_with_seed():
    a = random_degradation_chain(np.random.default_rng(123))
    b = random_degradation_chain(np.random.default_rng(123))
    assert [s.name for s in a] == [s.name for s in b]
    for x, y in zip(a, b):
        assert x.params == y.params
        assert x.severity == pytest.approx(y.severity)


def test_random_chain_returns_distinct_names():
    for seed in range(20):
        chain = random_degradation_chain(np.random.default_rng(seed))
        names = [s.name for s in chain]
        assert len(names) == len(set(names)), f"seed {seed}: duplicates {names}"
        assert 1 <= len(names) <= 3
        assert all(n in ISSUE_DIMENSIONS for n in names)


def test_random_chain_rejects_invalid_n_range():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        random_degradation_chain(rng, n_min=0, n_max=3)
    with pytest.raises(ValueError):
        random_degradation_chain(rng, n_min=2, n_max=1)
    with pytest.raises(ValueError):
        random_degradation_chain(rng, n_min=1, n_max=NUM_DIMENSIONS + 1)


# ---------------------------------------------------------------------------
# Severity normalization: low/high gain magnitudes hit ~0 / ~1
# ---------------------------------------------------------------------------


class _FixedRNG:
    """Stub Generator that returns a fixed uniform() value, so we can pin a
    sampler to its min or max parameter bound."""

    def __init__(self, value: float):
        self._value = value

    def uniform(self, lo, hi):
        if self._value == 0.0:
            return float(lo)
        if self._value == 1.0:
            return float(hi)
        return float(lo + (hi - lo) * self._value)

    def random(self):
        return self._value

    def integers(self, lo, hi):
        # uniform integer in [lo, hi)
        return int(lo + (hi - lo) * self._value) if hi > lo else int(lo)

    def choice(self, a, size, replace):
        # not used in these tests
        raise NotImplementedError


@pytest.mark.parametrize("name", ISSUE_DIMENSIONS)
def test_severity_hits_unit_endpoints(name):
    # The two param-range endpoints should map to severities {0, 1} in some
    # order. Some degradations (presence_lack, loudness_deficit,
    # dynamic_range_issue) invert the relationship — what matters is that
    # severity covers the full unit interval, not the direction of the mapping.
    s_lo = sample_degradation(name, _FixedRNG(0.0)).severity
    s_hi = sample_degradation(name, _FixedRNG(1.0)).severity
    severities = sorted([s_lo, s_hi])
    assert severities[0] == pytest.approx(0.0, abs=1e-6), f"{name}: low end {s_lo=} {s_hi=}"
    assert severities[1] == pytest.approx(1.0, abs=1e-6), f"{name}: high end {s_lo=} {s_hi=}"


# ---------------------------------------------------------------------------
# Audible-change sanity: degraded audio differs meaningfully from input
# ---------------------------------------------------------------------------


def test_degraded_audio_differs_from_input(stereo_sine):
    rng = np.random.default_rng(99)
    out, _, _ = degrade(stereo_sine, SR, rng)
    assert out.shape == stereo_sine.shape
    # Some difference, but not silence/explosion.
    diff = np.abs(out - stereo_sine).mean()
    assert diff > 1e-4, "degraded output too similar to input"
    assert np.max(np.abs(out)) < 4.0, "degraded output suspiciously loud"


def test_degrade_returns_consistent_triplet(stereo_sine):
    rng = np.random.default_rng(2)
    out, labels, chain = degrade(stereo_sine, SR, rng)
    assert out.shape == stereo_sine.shape
    assert labels.shape == (NUM_DIMENSIONS,)
    assert 1 <= len(chain) <= 3
    # Label nonzero exactly where chain has entries.
    nz = {ISSUE_DIMENSIONS[i] for i in np.where(labels > 0)[0]}
    assert nz == {s.name for s in chain}


def test_specs_to_jsonable_roundtrip():
    rng = np.random.default_rng(5)
    chain = random_degradation_chain(rng)
    j = specs_to_jsonable(chain)
    import json
    s = json.dumps(j)  # must not raise
    parsed = json.loads(s)
    assert len(parsed) == len(chain)
    for entry, spec in zip(parsed, chain):
        assert entry["name"] == spec.name
        assert entry["severity"] == pytest.approx(spec.severity)
        assert set(entry["params"].keys()) == set(spec.params.keys())
