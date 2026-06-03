"""PyTorch Dataset yielding (degraded_audio_chunk, label_vector) pairs.

Each __getitem__ picks a random chunk of a real mixture, applies a fresh
random degradation chain, and returns the degraded chunk resampled to 32 kHz
MONO (the PANNs CNN14 ingest format) plus the (8,) float32 label vector.

In `deterministic=True` mode each idx returns the same (chunk, degradation)
across calls — used for validation.
"""
from __future__ import annotations

import numpy as np
import torch
import librosa
from torch.utils.data import Dataset

from data.degradation import NUM_DIMENSIONS, degrade
from data.musdb_loader import MUSDBLoader


PANNS_SR: int = 32000
# Default chunk = 6.0 s because the bundled musdb-7s tracks are 6.8 s. When
# we move to full MUSDB18-HQ (3-5 min tracks), pass chunk_seconds=10.0.
DEFAULT_CHUNK_SECONDS: float = 6.0


class DegradedAudioDataset(Dataset):
    def __init__(
        self,
        loader: MUSDBLoader,
        chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
        chunks_per_track: int = 8,
        target_sr: int = PANNS_SR,
        deterministic: bool = False,
        seed: int = 0,
    ):
        if len(loader) == 0:
            raise ValueError("loader has no tracks")
        self.loader = loader
        self.chunk_seconds = float(chunk_seconds)
        self.chunks_per_track = int(chunks_per_track)
        self.target_sr = int(target_sr)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self._n_tracks = len(loader)

    def __len__(self) -> int:
        return self._n_tracks * self.chunks_per_track

    def _rng(self, idx: int) -> np.random.Generator:
        if self.deterministic:
            return np.random.default_rng(self.seed * 1_000_003 + idx)
        # In a DataLoader worker, torch.initial_seed() differs per worker, so
        # different workers produce different degradations for the same idx.
        worker_entropy = int(torch.initial_seed() & 0xFFFFFFFF)
        return np.random.default_rng(worker_entropy ^ (self.seed + idx))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        rng = self._rng(idx)
        track_idx = idx // self.chunks_per_track
        audio, sr, _ = self.loader.get_mixture(track_idx)

        chunk_samples = int(self.chunk_seconds * sr)
        total = audio.shape[-1]
        if total >= chunk_samples:
            start = int(rng.integers(0, total - chunk_samples + 1))
            clean = audio[..., start : start + chunk_samples]
        else:
            # Track shorter than chunk — reflect-pad to length.
            clean = np.pad(audio, ((0, 0), (0, chunk_samples - total)), mode="reflect")

        degraded, label, _ = degrade(clean, sr, rng)

        # PANNs CNN14 wants 32 kHz mono
        mono = degraded.mean(axis=0) if degraded.ndim == 2 else degraded
        if sr != self.target_sr:
            mono = librosa.resample(
                mono, orig_sr=sr, target_sr=self.target_sr, res_type="soxr_hq"
            )
        mono = np.ascontiguousarray(mono.astype(np.float32, copy=False))

        return torch.from_numpy(mono), torch.from_numpy(label)
