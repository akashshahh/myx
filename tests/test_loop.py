"""Tests for the iterative mastering loop — offline (fake reasoner, random model)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from agent.executor import MasteringChain
from agent.loop import MAX_ITERATIONS, master
from agent.reasoner import MasteringDecision, Reasoner
from perception.inference import PerceptionInference
from perception.model import build_perception_model

SR = 44100


class FakeReasoner(Reasoner):
    """Yields a scripted sequence of decisions, ignoring the LLM entirely."""

    def __init__(self, decisions):
        super().__init__(api_key="test")
        self._decisions = list(decisions)
        self.seen_iterations = []

    def decide(self, issue_vector, features, target_lufs=-14.0, iteration=0, history=None):
        self.seen_iterations.append(iteration)
        return self._decisions[min(iteration, len(self._decisions) - 1)]


@pytest.fixture(scope="module")
def perception() -> PerceptionInference:
    model = build_perception_model(checkpoint_path=None, freeze_backbone=True)
    return PerceptionInference(model, window_seconds=6.0)


def _audio(dur=8.0):
    rng = np.random.default_rng(1)
    return (0.1 * rng.standard_normal((2, int(SR * dur)))).astype(np.float32)


def test_single_pass_then_stop(perception):
    decision = MasteringDecision(
        reasoning="boost air, done",
        chain=MasteringChain(steps=[{"type": "high_shelf", "freq_hz": 10000, "gain_db": 3}]),
        needs_another_pass=False,
    )
    reasoner = FakeReasoner([decision])
    res = master(_audio(), SR, perception, reasoner)
    assert len(res.iterations) == 1
    assert res.audio.shape[0] == 2
    assert "completion" in res.stopped_reason
    # trace must be JSON-serializable
    json.dumps(res.trace())


def test_caps_at_max_iterations(perception):
    always_more = MasteringDecision(
        reasoning="keep going",
        chain=MasteringChain(steps=[{"type": "gain", "gain_db": 1}]),
        needs_another_pass=True,
    )
    reasoner = FakeReasoner([always_more])
    res = master(_audio(), SR, perception, reasoner)
    assert len(res.iterations) == MAX_ITERATIONS
    assert "max_iterations" in res.stopped_reason


def test_empty_chain_breaks(perception):
    noop = MasteringDecision(
        reasoning="already good",
        chain=MasteringChain(steps=[]),
        needs_another_pass=True,  # says continue, but empty chain should still break
    )
    reasoner = FakeReasoner([noop])
    res = master(_audio(), SR, perception, reasoner)
    # needs_another_pass=False path not hit, but empty-chain guard stops after 1
    assert len(res.iterations) == 1
    assert "empty chain" in res.stopped_reason


def test_perception_delta_recorded(perception):
    decision = MasteringDecision(
        reasoning="x",
        chain=MasteringChain(steps=[{"type": "gain", "gain_db": -6}]),
        needs_another_pass=False,
    )
    res = master(_audio(), SR, perception, FakeReasoner([decision]))
    delta = res.iterations[0].perception_delta
    assert delta is not None
    assert set(delta.keys()) == set(res.iterations[0].issue_vector.keys())
