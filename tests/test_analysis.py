"""Tests for the DSP analysis surface."""
from __future__ import annotations

import json

import numpy as np
import pytest

from agent.analysis import analyze

SR = 44100


def _stereo_tone(freq=440.0, dur=2.0, amp=0.3):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False, dtype=np.float32)
    tone = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([tone, tone])


def test_analyze_is_json_serializable():
    feats = analyze(_stereo_tone(), SR)
    json.dumps(feats)  # raises if not serializable


def test_top_level_keys():
    feats = analyze(_stereo_tone(), SR)
    assert set(feats) >= {"loudness", "dynamics", "spectral", "stereo", "detectors"}


def test_loudness_fields():
    feats = analyze(_stereo_tone(), SR)
    L = feats["loudness"]
    assert {"integrated_lufs", "short_term_lufs_max", "lra", "sample_peak_dbfs", "true_peak_dbtp"} <= set(L)
    # a 0.3-amplitude sine peaks at ~-10.5 dBFS; just assert it's sane (negative, not -inf)
    assert -20 < L["sample_peak_dbfs"] <= 0.5


def test_mono_input_flagged():
    t = np.linspace(0, 1.0, SR, endpoint=False, dtype=np.float32)
    mono = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    feats = analyze(mono, SR)
    assert feats["stereo"]["is_mono"] is True


def test_resonance_peak_detects_injected_tone():
    # A loud narrow tone at 1 kHz on top of broadband noise should surface as a peak.
    rng = np.random.default_rng(0)
    dur = 3.0
    n = int(SR * dur)
    t = np.arange(n) / SR
    noise = 0.05 * rng.standard_normal(n).astype(np.float32)
    tone = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    sig = np.stack([noise + tone, noise + tone])
    feats = analyze(sig, SR)
    peaks = feats["detectors"]["narrow_resonance_peaks"]
    assert peaks, "expected at least one resonance peak"
    assert any(abs(p["hz"] - 1000) < 60 for p in peaks)


def test_dc_offset_detected():
    t = np.linspace(0, 1.0, SR, endpoint=False, dtype=np.float32)
    tone = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.1).astype(np.float32)  # +0.1 DC
    feats = analyze(np.stack([tone, tone]), SR)
    assert feats["detectors"]["dc_offset"][0] == pytest.approx(0.1, abs=0.02)


def test_empty_audio_raises():
    with pytest.raises(ValueError):
        analyze(np.zeros((2, 0), dtype=np.float32), SR)
