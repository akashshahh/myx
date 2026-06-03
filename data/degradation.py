"""Synthetic degradation library — the project's supervision source.

For each clean audio chunk, we randomly apply 1-3 "bad mix" pedalboard chains
(low-shelf boom, harsh resonance, over-compression, etc.) and emit an 8-dim
label vector recording which degradations were applied and how severely. The
perception model is trained to invert this map.

The same 8 degradation types correspond 1:1 to the perception model's output
dimensions (`ISSUE_DIMENSIONS`), so a label vector and a perception prediction
share an identical contract.

Public API:
    ISSUE_DIMENSIONS        — ordered list of 8 dim names
    DegradationSpec         — frozen dataclass: name, params, severity in [0,1]
    sample_degradation      — produce one random spec for a named degradation
    random_degradation_chain — sample 1-3 distinct degradations
    apply_chain             — render audio through a chain
    chain_to_label_vec      — chain -> (8,) float32 label vector
    degrade                 — one-shot: audio -> (degraded, label, chain)
    specs_to_jsonable       — for logging / sanity scripts
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pedalboard as pb


ISSUE_DIMENSIONS: list[str] = [
    "low_excess",
    "low_mid_mud",
    "mid_balance",
    "presence_lack",
    "harshness",
    "over_compression",
    "loudness_deficit",
    "dynamic_range_issue",
]
NUM_DIMENSIONS: int = len(ISSUE_DIMENSIONS)
_NAME_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(ISSUE_DIMENSIONS)}


@dataclass(frozen=True)
class DegradationSpec:
    name: str
    params: dict
    severity: float

    def __post_init__(self) -> None:
        if self.name not in _NAME_TO_IDX:
            raise ValueError(f"Unknown degradation: {self.name!r}")
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(
                f"severity must be in [0, 1], got {self.severity} for {self.name}"
            )


# ---------------------------------------------------------------------------
# Per-degradation samplers + builders
#
# Each sampler returns (params_dict, severity_in_unit_interval).
# Each builder takes that params_dict and returns a pedalboard Plugin.
# ---------------------------------------------------------------------------


def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def _norm(value: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 1.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


# low_excess: low-shelf boost in the sub/low region
def _sample_low_excess(rng):
    freq = _u(rng, 50.0, 120.0)
    gain_db = _u(rng, 4.0, 12.0)
    return {"freq": freq, "gain_db": gain_db, "q": 0.7071}, _norm(gain_db, 4.0, 12.0)


def _build_low_shelf(p):
    return pb.LowShelfFilter(
        cutoff_frequency_hz=p["freq"], gain_db=p["gain_db"], q=p["q"]
    )


# low_mid_mud: peak boost in 200-450 Hz mud zone
def _sample_low_mid_mud(rng):
    freq = _u(rng, 200.0, 450.0)
    gain_db = _u(rng, 4.0, 11.0)
    q = _u(rng, 0.7, 1.5)
    return {"freq": freq, "gain_db": gain_db, "q": q}, _norm(gain_db, 4.0, 11.0)


# mid_balance: peak boost OR cut in 700-1800 Hz (boxy/honky vs scooped)
def _sample_mid_balance(rng):
    freq = _u(rng, 700.0, 1800.0)
    mag = _u(rng, 4.0, 9.0)
    sign = -1.0 if rng.random() < 0.5 else 1.0
    gain_db = sign * mag
    q = _u(rng, 0.7, 1.5)
    return {"freq": freq, "gain_db": gain_db, "q": q}, _norm(mag, 4.0, 9.0)


def _build_peak(p):
    return pb.PeakFilter(
        cutoff_frequency_hz=p["freq"], gain_db=p["gain_db"], q=p["q"]
    )


# presence_lack: high-shelf CUT in the 3.5-5 kHz "presence" region
def _sample_presence_lack(rng):
    freq = _u(rng, 3500.0, 5000.0)
    gain_db = _u(rng, -8.0, -3.0)
    return {"freq": freq, "gain_db": gain_db, "q": 0.7071}, _norm(-gain_db, 3.0, 8.0)


def _build_high_shelf(p):
    return pb.HighShelfFilter(
        cutoff_frequency_hz=p["freq"], gain_db=p["gain_db"], q=p["q"]
    )


# harshness: narrow peak boost in 2.5-5.5 kHz (ice-pick region)
def _sample_harshness(rng):
    freq = _u(rng, 2500.0, 5500.0)
    gain_db = _u(rng, 5.0, 12.0)
    q = _u(rng, 1.5, 3.0)
    return {"freq": freq, "gain_db": gain_db, "q": q}, _norm(gain_db, 5.0, 12.0)


# over_compression: high-ratio compressor with low threshold and fast attack
def _sample_over_compression(rng):
    ratio = _u(rng, 10.0, 25.0)
    threshold_db = _u(rng, -40.0, -30.0)
    return (
        {
            "ratio": ratio,
            "threshold_db": threshold_db,
            "attack_ms": 2.0,
            "release_ms": 80.0,
        },
        _norm(ratio, 10.0, 25.0),
    )


def _build_compressor(p):
    return pb.Compressor(
        threshold_db=p["threshold_db"],
        ratio=p["ratio"],
        attack_ms=p["attack_ms"],
        release_ms=p["release_ms"],
    )


# loudness_deficit: simple level cut
def _sample_loudness_deficit(rng):
    gain_db = _u(rng, -20.0, -10.0)
    return {"gain_db": gain_db}, _norm(-gain_db, 10.0, 20.0)


def _build_gain(p):
    return pb.Gain(gain_db=p["gain_db"])


# dynamic_range_issue: aggressive limiter at -10 dB; shorter release = harsher
def _sample_dynamic_range_issue(rng):
    threshold_db = -10.0
    release_ms = _u(rng, 10.0, 100.0)
    severity = 1.0 - _norm(release_ms, 10.0, 100.0)
    return {"threshold_db": threshold_db, "release_ms": release_ms}, severity


def _build_limiter(p):
    return pb.Limiter(threshold_db=p["threshold_db"], release_ms=p["release_ms"])


_DISPATCH: dict[str, tuple[Callable, Callable]] = {
    "low_excess": (_sample_low_excess, _build_low_shelf),
    "low_mid_mud": (_sample_low_mid_mud, _build_peak),
    "mid_balance": (_sample_mid_balance, _build_peak),
    "presence_lack": (_sample_presence_lack, _build_high_shelf),
    "harshness": (_sample_harshness, _build_peak),
    "over_compression": (_sample_over_compression, _build_compressor),
    "loudness_deficit": (_sample_loudness_deficit, _build_gain),
    "dynamic_range_issue": (_sample_dynamic_range_issue, _build_limiter),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sample_degradation(name: str, rng: np.random.Generator) -> DegradationSpec:
    if name not in _DISPATCH:
        raise ValueError(f"Unknown degradation: {name!r}")
    sampler, _ = _DISPATCH[name]
    params, severity = sampler(rng)
    return DegradationSpec(name=name, params=params, severity=severity)


def build_plugin(spec: DegradationSpec) -> pb.Plugin:
    _, builder = _DISPATCH[spec.name]
    return builder(spec.params)


def random_degradation_chain(
    rng: np.random.Generator, n_min: int = 1, n_max: int = 3
) -> list[DegradationSpec]:
    """Sample N distinct degradations (N uniform in [n_min, n_max]) with random params."""
    if not 1 <= n_min <= n_max <= NUM_DIMENSIONS:
        raise ValueError(f"invalid n_min={n_min}, n_max={n_max}")
    n = int(rng.integers(n_min, n_max + 1))
    names = list(rng.choice(ISSUE_DIMENSIONS, size=n, replace=False))
    return [sample_degradation(name, rng) for name in names]


def _as_pb_audio(audio: np.ndarray) -> np.ndarray:
    """Pedalboard wants float32, contiguous, shape (samples,) mono or (channels, samples)."""
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    return np.ascontiguousarray(audio)


def apply_chain(
    audio: np.ndarray, sr: int, specs: list[DegradationSpec]
) -> np.ndarray:
    """Render `audio` through the given chain in order; returns same shape, float32."""
    if not specs:
        return _as_pb_audio(audio).copy()
    board = pb.Pedalboard([build_plugin(s) for s in specs])
    return board(_as_pb_audio(audio), float(sr))


def chain_to_label_vec(specs: list[DegradationSpec]) -> np.ndarray:
    """(8,) float32 vector. Max-pool severity across duplicates (no-op for distinct chains)."""
    vec = np.zeros(NUM_DIMENSIONS, dtype=np.float32)
    for spec in specs:
        idx = _NAME_TO_IDX[spec.name]
        if spec.severity > vec[idx]:
            vec[idx] = spec.severity
    return vec


def degrade(
    audio: np.ndarray, sr: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, list[DegradationSpec]]:
    """One-shot: sample a chain, apply it, return (degraded_audio, label_vec, chain)."""
    chain = random_degradation_chain(rng)
    degraded = apply_chain(audio, sr, chain)
    labels = chain_to_label_vec(chain)
    return degraded, labels, chain


def specs_to_jsonable(specs: list[DegradationSpec]) -> list[dict]:
    return [
        {"name": s.name, "severity": s.severity, "params": dict(s.params)}
        for s in specs
    ]
