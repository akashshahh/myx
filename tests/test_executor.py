"""Tests for the mastering executor (Pydantic chain validation + pedalboard render)."""
from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from agent.executor import (
    OP_TYPES,
    MasteringChain,
    chain_to_jsonable,
    render,
)

SR = 44100


@pytest.fixture
def stereo() -> np.ndarray:
    t = np.linspace(0, 1.0, SR, endpoint=False, dtype=np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 220 * t)
    return np.stack([tone, tone * 0.9])


def test_eight_op_types():
    assert len(OP_TYPES) == 8


def test_empty_chain_is_copy(stereo):
    out = render(stereo, SR, MasteringChain(steps=[]))
    assert out.shape == stereo.shape
    np.testing.assert_allclose(out, stereo, atol=1e-6)


def test_all_ops_parse_and_render(stereo):
    chain = MasteringChain(
        steps=[
            {"type": "low_shelf", "freq_hz": 100, "gain_db": 3},
            {"type": "high_shelf", "freq_hz": 8000, "gain_db": -2},
            {"type": "peak_eq", "freq_hz": 3000, "gain_db": -4, "q": 3.0},
            {"type": "highpass", "freq_hz": 30},
            {"type": "lowpass", "freq_hz": 16000},
            {"type": "compressor", "threshold_db": -18, "ratio": 3.0},
            {"type": "limiter", "threshold_db": -1.0},
            {"type": "gain", "gain_db": 2},
        ]
    )
    assert len(chain.steps) == 8
    out = render(stereo, SR, chain)
    assert out.shape == stereo.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_out_of_range_param_rejected():
    with pytest.raises(ValidationError):
        MasteringChain(steps=[{"type": "gain", "gain_db": 999}])
    with pytest.raises(ValidationError):
        MasteringChain(steps=[{"type": "highpass", "freq_hz": 5000}])  # >500 cap


def test_unknown_op_rejected():
    with pytest.raises(ValidationError):
        MasteringChain(steps=[{"type": "reverb", "wet": 0.5}])


def test_chain_jsonable_roundtrip():
    chain = MasteringChain(steps=[{"type": "gain", "gain_db": -3}])
    js = chain_to_jsonable(chain)
    assert js == [{"type": "gain", "gain_db": -3.0}]
    # rebuild from json
    rebuilt = MasteringChain(steps=js)
    assert rebuilt.steps[0].gain_db == -3.0
