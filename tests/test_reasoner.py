"""Tests for the LLM reasoner — fully offline via a patched _raw_completion."""
from __future__ import annotations

import json

from agent.reasoner import MasteringDecision, Reasoner


class ScriptedReasoner(Reasoner):
    """Returns canned completions from a queue; records call count."""

    def __init__(self, responses):
        super().__init__(api_key="test-key")
        self._responses = list(responses)
        self.calls = 0

    def _raw_completion(self, messages):
        self.calls += 1
        return self._responses.pop(0)


VALID = json.dumps({
    "reasoning": "Track is a touch dull and 6 dB under target; add air and gain.",
    "chain": {"steps": [
        {"type": "high_shelf", "freq_hz": 10000, "gain_db": 2.0},
        {"type": "gain", "gain_db": 6.0},
    ]},
    "needs_another_pass": True,
})

FEATURES = {"loudness": {"integrated_lufs": -20.0}}
ISSUE_VEC = {"low_excess": 0.1, "harshness": 0.2}


def test_valid_json_parses():
    r = ScriptedReasoner([VALID])
    d = r.decide(ISSUE_VEC, FEATURES)
    assert isinstance(d, MasteringDecision)
    assert len(d.chain.steps) == 2
    assert d.needs_another_pass is True
    assert r.calls == 1


def test_markdown_fenced_json_tolerated():
    r = ScriptedReasoner(["```json\n" + VALID + "\n```"])
    d = r.decide(ISSUE_VEC, FEATURES)
    assert len(d.chain.steps) == 2


def test_retry_on_bad_then_good():
    r = ScriptedReasoner(["not json at all", VALID])
    d = r.decide(ISSUE_VEC, FEATURES)
    assert r.calls == 2
    assert len(d.chain.steps) == 2


def test_fallback_after_two_failures():
    r = ScriptedReasoner(["garbage", "still garbage"])
    d = r.decide(ISSUE_VEC, FEATURES)
    assert r.calls == 2
    assert d.chain.steps == []  # safe no-op
    assert d.needs_another_pass is False


def test_invalid_op_value_triggers_repair():
    bad = json.dumps({
        "reasoning": "x",
        "chain": {"steps": [{"type": "gain", "gain_db": 999}]},  # out of range
        "needs_another_pass": False,
    })
    r = ScriptedReasoner([bad, VALID])
    d = r.decide(ISSUE_VEC, FEATURES)
    assert r.calls == 2
    assert len(d.chain.steps) == 2
