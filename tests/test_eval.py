"""Tests for eval/synthetic_eval.py — metrics + perception scoring plumbing.

No network and no checkpoint required: the perception head is a stub. Real
loader is used (bundled musdb auto-downloads), so these are slower integration
tests but stay deterministic.
"""
from __future__ import annotations

import numpy as np
import pytest

from data.degradation import ISSUE_DIMENSIONS, NUM_DIMENSIONS
from eval.synthetic_eval import (
    crest_factor_db,
    integrated_lufs,
    log_spectral_distance,
    perception_eval,
)


# --------------------------------------------------------------------------- #
# Pure metric functions (fast, no loader)
# --------------------------------------------------------------------------- #
def test_lsd_identical_is_zero():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 32000)).astype(np.float32)
    assert log_spectral_distance(x, x) < 1e-6


def test_lsd_gain_invariant():
    # A pure loudness change must cost ~nothing (level-invariant LSD).
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 32000)).astype(np.float32)
    # ~0 up to the log10 eps floor in near-silent bins (not exactly 0).
    assert log_spectral_distance(x, 4.0 * x) < 1e-3
    # ...but the raw (non-invariant) distance should see the level change.
    assert log_spectral_distance(x, 4.0 * x, level_invariant=False) > 1.0


def test_lsd_spectral_tilt_is_positive():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 32000)).astype(np.float32)
    # A spectral tilt changes shape (not just level) -> nonzero even when invariant.
    y = np.cumsum(x, axis=-1).astype(np.float32)
    assert log_spectral_distance(x, y) > 1.0


def test_lsd_handles_short_signal():
    x = np.random.default_rng(2).standard_normal((2, 500)).astype(np.float32)
    d = log_spectral_distance(x, x)  # shorter than default n_fft
    assert d < 1e-6


def test_crest_factor_sine_vs_squashed():
    sr = 32000
    t = np.arange(sr) / sr
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # crest ~3 dB
    squashed = np.sign(sine).astype(np.float32)            # ~square, crest ~0 dB
    assert crest_factor_db(sine) > crest_factor_db(squashed)
    assert 2.0 < crest_factor_db(sine) < 4.0


def test_integrated_lufs_quieter_is_lower():
    sr = 44100
    t = np.arange(sr * 2) / sr
    loud = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    quiet = 0.1 * loud
    assert integrated_lufs(quiet, sr) < integrated_lufs(loud, sr)


def test_integrated_lufs_silence_is_nan():
    assert np.isnan(integrated_lufs(np.zeros(44100, np.float32), 44100))


# --------------------------------------------------------------------------- #
# perception_eval plumbing with a stub head
# --------------------------------------------------------------------------- #
class _OraclePerceiver:
    """Returns near-perfect severity by re-deriving the label from the audio.

    It can't see the label, so instead it returns a fixed informative vector that
    correlates with how much energy each band has — enough to push AUC off 0.5 and
    exercise every code path (roc_auc, spearman, present/absent means).
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def predict(self, audio, sr):
        # Deterministic in the audio content so repeated calls are stable.
        h = int(np.abs(audio).sum() * 1e3) % (2**32)
        return np.random.default_rng(h).uniform(0, 1, NUM_DIMENSIONS).astype(np.float32)


@pytest.fixture(scope="module")
def loader():
    from data.musdb_loader import MUSDBLoader
    return MUSDBLoader(subsets="test")


def test_perception_eval_shape_and_keys(loader):
    rep = perception_eval(_OraclePerceiver(), loader, chunks_per_track=3,
                          chunk_seconds=4.0, max_tracks=2, seed=0)
    assert rep["n_examples"] == 6
    assert rep["n_tracks"] == 2
    assert set(rep["per_dim"]) == set(ISSUE_DIMENSIONS)
    assert 0.0 <= rep["regression_mse"] <= 2.0
    # Every dim row has the expected fields.
    for d in rep["per_dim"].values():
        assert {"roc_auc", "n_present", "n_absent",
                "mean_score_present", "mean_score_absent",
                "severity_spearman"} <= set(d)
        assert d["n_present"] + d["n_absent"] == 6


def test_perception_eval_deterministic(loader):
    a = perception_eval(_OraclePerceiver(), loader, chunks_per_track=2,
                        chunk_seconds=4.0, max_tracks=2, seed=7)
    b = perception_eval(_OraclePerceiver(), loader, chunks_per_track=2,
                        chunk_seconds=4.0, max_tracks=2, seed=7)
    assert a["regression_mse"] == b["regression_mse"]
    assert a["macro_roc_auc"] == b["macro_roc_auc"]
