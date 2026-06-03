"""Iterative self-listening mastering loop (cap 3 iterations).

Each iteration: analyze the *current* audio (DSP) + run the perception model
(learned issue vector) -> ask the LLM for a mastering chain -> render it ->
re-measure. The new perception vector becomes feedback for the next round. The
loop stops when the LLM says `needs_another_pass=False` or the cap is hit.

Returns the mastered audio plus a full trace (every iteration's inputs,
decision, and perception delta) so the API/frontend can show the reasoning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from agent.analysis import analyze
from agent.executor import MasteringChain, chain_to_jsonable, render
from agent.reasoner import MasteringDecision, Reasoner
from perception.inference import PerceptionInference

MAX_ITERATIONS = 3
DEFAULT_TARGET_LUFS = -14.0


@dataclass
class IterationRecord:
    iteration: int
    issue_vector: dict
    features: dict
    reasoning: str
    chain: list
    needs_another_pass: bool
    perception_delta: Optional[dict] = None  # filled after re-measuring


@dataclass
class MasteringResult:
    audio: np.ndarray
    sr: int
    target_lufs: float
    iterations: list[IterationRecord] = field(default_factory=list)
    stopped_reason: str = ""

    def trace(self) -> dict:
        """JSON-serializable trace for the API response / frontend."""
        return {
            "target_lufs": self.target_lufs,
            "stopped_reason": self.stopped_reason,
            "iterations": [
                {
                    "iteration": r.iteration,
                    "issue_vector": r.issue_vector,
                    "features": r.features,
                    "reasoning": r.reasoning,
                    "chain": r.chain,
                    "needs_another_pass": r.needs_another_pass,
                    "perception_delta": r.perception_delta,
                }
                for r in self.iterations
            ],
        }


def _delta(before: dict, after: dict) -> dict:
    return {k: round(after[k] - before[k], 4) for k in before}


def master(
    audio: np.ndarray,
    sr: int,
    perception: PerceptionInference,
    reasoner: Reasoner,
    target_lufs: float = DEFAULT_TARGET_LUFS,
    max_iterations: int = MAX_ITERATIONS,
) -> MasteringResult:
    """Run the iterative mastering loop on `audio` (channels, samples) float32."""
    current = np.asarray(audio, dtype=np.float32)
    result = MasteringResult(audio=current, sr=sr, target_lufs=target_lufs)
    history: list[dict] = []

    for i in range(max_iterations):
        features = analyze(current, sr)
        issue_vector = perception.predict_dict(current, sr)

        decision: MasteringDecision = reasoner.decide(
            issue_vector=issue_vector,
            features=features,
            target_lufs=target_lufs,
            iteration=i,
            history=history,
        )

        chain_json = chain_to_jsonable(decision.chain)
        record = IterationRecord(
            iteration=i,
            issue_vector=issue_vector,
            features=features,
            reasoning=decision.reasoning,
            chain=chain_json,
            needs_another_pass=decision.needs_another_pass,
        )

        # Apply the chain (no-op chain just copies).
        processed = render(current, sr, decision.chain)

        # Re-measure perception to record the effect of this pass.
        after_vector = perception.predict_dict(processed, sr)
        record.perception_delta = _delta(issue_vector, after_vector)
        result.iterations.append(record)

        history.append({
            "iteration": i,
            "issue_vector": {k: round(v, 3) for k, v in issue_vector.items()},
            "chain": chain_json,
            "perception_delta": record.perception_delta,
            "reasoning": decision.reasoning,
        })

        current = processed
        result.audio = current

        if not decision.needs_another_pass:
            result.stopped_reason = f"LLM signalled completion after iteration {i}"
            break
        if not decision.chain.steps:
            result.stopped_reason = f"empty chain at iteration {i}; nothing left to do"
            break
    else:
        result.stopped_reason = f"hit max_iterations ({max_iterations})"

    return result
