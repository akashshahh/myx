"""LLM reasoner — turns perception + DSP analysis into a mastering chain.

Talks to OpenRouter (OpenAI-compatible). The system prompt frames the 8-dim
`issue_vector` as a *high-confidence learned* signal while explicitly inviting
the model to diagnose problems OUTSIDE those 8 dimensions (sibilance, narrow
resonances, sub rumble, stereo width, dullness, pumping, DC offset, ...) by
reading the rich `features` dict. Output is strict JSON, validated with
Pydantic; one repair retry on parse failure, then a safe no-op fallback so the
caller never crashes.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from agent.executor import OP_TYPES, MasteringChain
from data.degradation import ISSUE_DIMENSIONS

DEFAULT_MODEL = "google/gemini-2.5-flash"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class MasteringDecision(BaseModel):
    """The LLM's per-iteration verdict."""

    reasoning: str = Field(description="Brief explanation of the diagnosis and chosen moves.")
    chain: MasteringChain = Field(default_factory=MasteringChain)
    needs_another_pass: bool = Field(
        default=False,
        description="True if another listen/process iteration is warranted.",
    )


_OP_DOC = """\
Allowed step types (exactly these 8 — `type` plus the listed params):
  low_shelf   {freq_hz 20-20000, gain_db -24..24, q}         broad low-end tilt
  high_shelf  {freq_hz 20-20000, gain_db -24..24, q}         broad top-end tilt (air/dullness)
  peak_eq     {freq_hz 20-20000, gain_db -24..24, q 0.1-12}  surgical boost/cut (resonances, de-ess, mud)
  highpass    {freq_hz 10-500}                                remove sub rumble / DC
  lowpass     {freq_hz 1000-20000}                            tame harsh top / noise
  compressor  {threshold_db -60..0, ratio 1-20, attack_ms, release_ms}  glue / control dynamics
  limiter     {threshold_db -24..0, release_ms}               peak control / loudness ceiling
  gain        {gain_db -24..24}                               level trim toward target LUFS
Process order matters: corrective EQ/cleanup first, dynamics, then gain/limiter last."""


def _system_prompt(target_lufs: float) -> str:
    return f"""You are a mastering engineer. You receive measurements of a track and decide on a \
mastering processing chain to improve it toward a target of {target_lufs:.1f} LUFS integrated, \
clean spectral balance, controlled-but-not-crushed dynamics, and a true-peak ceiling near -1 dBTP.

INPUTS you are given each iteration:
- issue_vector: 8 learned severity scores in [0,1] from a model trained to detect these specific \
problems: {", ".join(ISSUE_DIMENSIONS)}. Treat these as HIGH-CONFIDENCE detections.
- features: a rich DSP measurement dict. Use it to diagnose problems OUTSIDE the 8 learned \
dimensions too — sibilance (detectors.sibilance_index_db), narrow resonances \
(detectors.narrow_resonance_peaks), sub rumble (detectors.sub_rumble_db), DC offset, noise floor, \
stereo correlation/width (stereo.*), dullness or harshness (spectral.*), pumping/over-limiting \
(dynamics.*, loudness.lra). You are free to act on anything you see, not just the 8 dimensions.
- target_lufs, iteration index, and history of prior decisions + how the issue_vector changed.

{_OP_DOC}

Be conservative and surgical: prefer a few targeted moves over many large ones. If the track is \
already close to target and balanced, return an empty chain and needs_another_pass=false.

Respond with ONLY a JSON object of this exact shape (no markdown, no prose outside JSON):
{{"reasoning": "<one short paragraph>", "chain": {{"steps": [{{"type": "...", ...}}]}}, \
"needs_another_pass": <true|false>}}"""


def _user_message(issue_vector: dict, features: dict, target_lufs: float,
                  iteration: int, history: list) -> str:
    payload = {
        "iteration": iteration,
        "target_lufs": target_lufs,
        "issue_vector": {k: round(float(v), 3) for k, v in issue_vector.items()},
        "features": features,
        "history": history,
    }
    return json.dumps(payload)


class Reasoner:
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.4,
    ):
        self.model = model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
        self.base_url = base_url
        self.temperature = temperature
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._client = None  # lazily created so tests that patch _raw_completion need no key

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            if not self._api_key:
                raise RuntimeError("OPENROUTER_API_KEY not set")
            self._client = OpenAI(base_url=self.base_url, api_key=self._api_key)
        return self._client

    def _raw_completion(self, messages: list[dict]) -> str:
        """Single LLM call returning raw assistant text. Patch this in tests."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    def decide(
        self,
        issue_vector: dict,
        features: dict,
        target_lufs: float = -14.0,
        iteration: int = 0,
        history: Optional[list] = None,
    ) -> MasteringDecision:
        history = history or []
        system = _system_prompt(target_lufs)
        user = _user_message(issue_vector, features, target_lufs, iteration, history)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        # First attempt, then one repair retry feeding the error back.
        raw = ""
        for attempt in range(2):
            try:
                raw = self._raw_completion(messages)
                return self._parse(raw)
            except (ValidationError, json.JSONDecodeError, ValueError) as err:
                if attempt == 0:
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Your previous response failed validation: {err}. "
                            "Reply again with ONLY the valid JSON object, no markdown."
                        ),
                    })
                    continue
                break

        # Fallback: do nothing rather than 500. Safe and auditable.
        return MasteringDecision(
            reasoning="LLM output could not be parsed; defaulting to no-op chain.",
            chain=MasteringChain(steps=[]),
            needs_another_pass=False,
        )

    @staticmethod
    def _parse(raw: str) -> MasteringDecision:
        text = raw.strip()
        # Tolerate ```json fences if the model adds them despite instructions.
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        data = json.loads(text)
        return MasteringDecision.model_validate(data)
