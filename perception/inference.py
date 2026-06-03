"""Run the perception model over a full track and average to one 8-dim vector.

Pipeline audio lives at 44.1 kHz stereo float32 `(channels, samples)`. The PANNs
Cnn14 ingests 32 kHz MONO, so we downmix + resample, slice into fixed windows,
batch-predict, and mean-pool the per-window issue vectors into a single
`(8,)` severity estimate aligned to `ISSUE_DIMENSIONS`.
"""
from __future__ import annotations

import librosa
import numpy as np
import torch

from data.degradation import ISSUE_DIMENSIONS, NUM_DIMENSIONS
from perception.model import SAMPLE_RATE, PerceptionModel

DEFAULT_WINDOW_SECONDS = 10.0


def _to_mono_32k(audio: np.ndarray, sr: int) -> np.ndarray:
    """(channels, samples) or (samples,) at `sr` -> mono float32 at 32 kHz."""
    a = np.asarray(audio, dtype=np.float32)
    mono = a.mean(axis=0) if a.ndim == 2 else a
    if sr != SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=SAMPLE_RATE, res_type="soxr_hq")
    return np.ascontiguousarray(mono.astype(np.float32, copy=False))


def _windows(mono: np.ndarray, window_samples: int) -> np.ndarray:
    """Slice mono signal into (n_windows, window_samples). Pads the tail; always
    returns at least one window."""
    total = mono.shape[0]
    if total <= window_samples:
        pad = window_samples - total
        return np.pad(mono, (0, pad), mode="constant")[None, :]
    n = total // window_samples
    trimmed = mono[: n * window_samples]
    return trimmed.reshape(n, window_samples)


class PerceptionInference:
    """Wraps a loaded `PerceptionModel` for whole-track prediction."""

    def __init__(
        self,
        model: PerceptionModel,
        device: str | torch.device = "cpu",
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        batch_size: int = 8,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.window_samples = int(window_seconds * SAMPLE_RATE)
        self.batch_size = int(batch_size)

    @torch.no_grad()
    def predict(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Return a (8,) float32 issue vector in [0,1], aligned to ISSUE_DIMENSIONS."""
        mono = _to_mono_32k(audio, sr)
        windows = _windows(mono, self.window_samples)

        preds = []
        for i in range(0, len(windows), self.batch_size):
            batch = torch.from_numpy(windows[i : i + self.batch_size]).to(self.device)
            preds.append(self.model(batch).cpu().numpy())
        out = np.concatenate(preds, axis=0).mean(axis=0)
        return out.astype(np.float32)

    def predict_dict(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """Same as predict() but as a {dimension_name: severity} dict for the LLM."""
        vec = self.predict(audio, sr)
        return {name: float(vec[i]) for i, name in enumerate(ISSUE_DIMENSIONS)}


def issue_vector_to_dict(vec: np.ndarray) -> dict[str, float]:
    """Map a (8,) vector to {dimension_name: severity}."""
    if vec.shape[-1] != NUM_DIMENSIONS:
        raise ValueError(f"expected {NUM_DIMENSIONS} dims, got {vec.shape}")
    return {name: float(vec[i]) for i, name in enumerate(ISSUE_DIMENSIONS)}
