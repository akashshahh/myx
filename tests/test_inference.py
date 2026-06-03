"""Tests for whole-track perception inference (uses a random-init model — fast)."""
from __future__ import annotations

import numpy as np
import pytest

from data.degradation import ISSUE_DIMENSIONS, NUM_DIMENSIONS
from perception.inference import PerceptionInference, issue_vector_to_dict
from perception.model import build_perception_model

SR = 44100


@pytest.fixture(scope="module")
def infer() -> PerceptionInference:
    model = build_perception_model(checkpoint_path=None, freeze_backbone=True)
    return PerceptionInference(model, window_seconds=6.0)


def _stereo_noise(dur=8.0):
    rng = np.random.default_rng(0)
    n = int(SR * dur)
    return (0.1 * rng.standard_normal((2, n))).astype(np.float32)


def test_predict_shape_and_range(infer):
    vec = infer.predict(_stereo_noise(), SR)
    assert vec.shape == (NUM_DIMENSIONS,)
    assert np.all(vec >= 0) and np.all(vec <= 1)


def test_predict_dict_keys(infer):
    d = infer.predict_dict(_stereo_noise(), SR)
    assert list(d.keys()) == ISSUE_DIMENSIONS
    assert all(0.0 <= v <= 1.0 for v in d.values())


def test_short_clip_still_predicts(infer):
    # clip shorter than one window must still yield a vector
    vec = infer.predict(_stereo_noise(dur=2.0), SR)
    assert vec.shape == (NUM_DIMENSIONS,)


def test_issue_vector_to_dict():
    vec = np.linspace(0, 1, NUM_DIMENSIONS, dtype=np.float32)
    d = issue_vector_to_dict(vec)
    assert list(d.keys()) == ISSUE_DIMENSIONS
    assert d[ISSUE_DIMENSIONS[0]] == 0.0
