"""Mastering executor — 8 general-purpose pedalboard primitives.

The LLM emits a `MasteringChain` (an ordered list of typed steps); we validate
it with Pydantic, build a `pedalboard.Pedalboard` in order, and render at the
pipeline rate (44.1 kHz stereo float32). Param ranges are clamped by the schema
so a hallucinated value can never blow up the render or the speakers.

These 8 ops are intentionally general-purpose: together they cover not just the
8 trained issue dimensions but also problems the LLM diagnoses from the rich
DSP features (sub rumble -> highpass, sibilance/resonance -> narrow peak cut,
dullness -> high shelf, etc.). The op vocabulary stays fixed at 8.
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

import numpy as np
import pedalboard as pb
from pydantic import BaseModel, Field

PIPELINE_SR = 44100

# Shared field constraints
_Freq = Field(ge=20.0, le=20000.0, description="frequency in Hz")
_Gain = Field(ge=-24.0, le=24.0, description="gain in dB")
_Q = Field(default=0.7071, ge=0.1, le=10.0, description="filter Q / bandwidth")


class LowShelf(BaseModel):
    type: Literal["low_shelf"]
    freq_hz: float = _Freq
    gain_db: float = _Gain
    q: float = _Q

    def to_plugin(self) -> pb.Plugin:
        return pb.LowShelfFilter(cutoff_frequency_hz=self.freq_hz, gain_db=self.gain_db, q=self.q)


class HighShelf(BaseModel):
    type: Literal["high_shelf"]
    freq_hz: float = _Freq
    gain_db: float = _Gain
    q: float = _Q

    def to_plugin(self) -> pb.Plugin:
        return pb.HighShelfFilter(cutoff_frequency_hz=self.freq_hz, gain_db=self.gain_db, q=self.q)


class PeakEQ(BaseModel):
    type: Literal["peak_eq"]
    freq_hz: float = _Freq
    gain_db: float = _Gain
    q: float = Field(default=1.0, ge=0.1, le=12.0, description="filter Q (higher = narrower)")

    def to_plugin(self) -> pb.Plugin:
        return pb.PeakFilter(cutoff_frequency_hz=self.freq_hz, gain_db=self.gain_db, q=self.q)


class HighPass(BaseModel):
    type: Literal["highpass"]
    freq_hz: float = Field(ge=10.0, le=500.0, description="highpass cutoff in Hz (rumble removal)")

    def to_plugin(self) -> pb.Plugin:
        return pb.HighpassFilter(cutoff_frequency_hz=self.freq_hz)


class LowPass(BaseModel):
    type: Literal["lowpass"]
    freq_hz: float = Field(ge=1000.0, le=20000.0, description="lowpass cutoff in Hz (taming air/harshness)")

    def to_plugin(self) -> pb.Plugin:
        return pb.LowpassFilter(cutoff_frequency_hz=self.freq_hz)


class CompressorStep(BaseModel):
    type: Literal["compressor"]
    threshold_db: float = Field(ge=-60.0, le=0.0)
    ratio: float = Field(ge=1.0, le=20.0)
    attack_ms: float = Field(default=10.0, ge=0.1, le=200.0)
    release_ms: float = Field(default=100.0, ge=5.0, le=1000.0)

    def to_plugin(self) -> pb.Plugin:
        return pb.Compressor(
            threshold_db=self.threshold_db, ratio=self.ratio,
            attack_ms=self.attack_ms, release_ms=self.release_ms,
        )


class LimiterStep(BaseModel):
    type: Literal["limiter"]
    threshold_db: float = Field(ge=-24.0, le=0.0)
    release_ms: float = Field(default=100.0, ge=5.0, le=1000.0)

    def to_plugin(self) -> pb.Plugin:
        return pb.Limiter(threshold_db=self.threshold_db, release_ms=self.release_ms)


class GainStep(BaseModel):
    type: Literal["gain"]
    gain_db: float = _Gain

    def to_plugin(self) -> pb.Plugin:
        return pb.Gain(gain_db=self.gain_db)


ChainStep = Annotated[
    Union[
        LowShelf, HighShelf, PeakEQ, HighPass, LowPass,
        CompressorStep, LimiterStep, GainStep,
    ],
    Field(discriminator="type"),
]

# Names exposed to the LLM prompt / schema docs.
OP_TYPES: list[str] = [
    "low_shelf", "high_shelf", "peak_eq", "highpass", "lowpass",
    "compressor", "limiter", "gain",
]


class MasteringChain(BaseModel):
    """An ordered processing chain. Empty = no-op (valid: 'already mastered')."""

    steps: list[ChainStep] = Field(default_factory=list, max_length=12)


def _as_pb_audio(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio)
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    return np.ascontiguousarray(a)


def render(audio: np.ndarray, sr: int, chain: MasteringChain) -> np.ndarray:
    """Render `audio` (channels, samples) through the chain in order.

    Returns float32, same shape and sample rate. An empty chain is a copy."""
    a = _as_pb_audio(audio)
    if not chain.steps:
        return a.copy()
    board = pb.Pedalboard([s.to_plugin() for s in chain.steps])
    return board(a, float(sr))


def chain_to_jsonable(chain: MasteringChain) -> list[dict]:
    """Compact serialization for trace logs."""
    return [s.model_dump() for s in chain.steps]
