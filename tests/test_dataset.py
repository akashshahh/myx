"""Smoke tests for MUSDBLoader + DegradedAudioDataset.

These hit the bundled musdb-7s samples (already downloaded into ~/MUSDB18/).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data.degradation import NUM_DIMENSIONS
from data.dataset import PANNS_SR, DegradedAudioDataset
from data.musdb_loader import MUSDBLoader


@pytest.fixture(scope="module")
def loader() -> MUSDBLoader:
    # Use train subset only; the bundled musdb-7s splits 100 train / 50 test
    return MUSDBLoader(subsets="train")


def test_loader_basic(loader):
    assert len(loader) > 0
    audio, sr, name = loader.get_mixture(0)
    assert audio.ndim == 2 and audio.shape[0] == 2, audio.shape
    assert audio.dtype == np.float32
    assert sr == 44100
    assert isinstance(name, str) and len(name) > 0


def test_loader_names_match_len(loader):
    assert len(loader.names) == len(loader)


def test_loader_min_duration_reasonable(loader):
    d = loader.min_duration()
    # Bundled samples are exactly 6.8 s; full MUSDB18 tracks are much longer.
    assert 6.0 <= d <= 600.0, d


def test_dataset_shapes(loader):
    ds = DegradedAudioDataset(
        loader, chunk_seconds=6.0, chunks_per_track=2, deterministic=True
    )
    assert len(ds) == 2 * len(loader)
    x, y = ds[0]
    expected = int(6.0 * PANNS_SR)
    assert x.dtype == torch.float32
    assert x.shape == (expected,)
    assert y.dtype == torch.float32
    assert y.shape == (NUM_DIMENSIONS,)
    assert torch.all(y >= 0.0) and torch.all(y <= 1.0)


def test_dataset_deterministic_repeat(loader):
    ds = DegradedAudioDataset(loader, deterministic=True, seed=7)
    x1, y1 = ds[5]
    x2, y2 = ds[5]
    torch.testing.assert_close(x1, x2)
    torch.testing.assert_close(y1, y2)


def test_dataset_indices_produce_different_outputs(loader):
    ds = DegradedAudioDataset(loader, chunks_per_track=4, deterministic=True, seed=7)
    x_a, _ = ds[0]
    x_b, _ = ds[len(ds) // 2]  # likely a different track
    assert not torch.allclose(x_a, x_b)


def test_dataset_index_out_of_range(loader):
    ds = DegradedAudioDataset(loader, chunks_per_track=1, deterministic=True)
    with pytest.raises(IndexError):
        _ = ds[len(ds)]


def test_dataloader_single_worker(loader):
    ds = DegradedAudioDataset(loader, chunks_per_track=1, deterministic=True)
    dl = DataLoader(ds, batch_size=4, num_workers=0, shuffle=False)
    batch_x, batch_y = next(iter(dl))
    assert batch_x.shape == (4, int(6.0 * PANNS_SR))
    assert batch_y.shape == (4, NUM_DIMENSIONS)
    assert torch.all(torch.isfinite(batch_x))


def test_dataset_handles_short_track(loader):
    """If chunk_seconds exceeds track length, dataset should reflect-pad."""
    ds = DegradedAudioDataset(
        loader, chunk_seconds=20.0, chunks_per_track=1, deterministic=True
    )
    x, _ = ds[0]
    assert x.shape == (int(20.0 * PANNS_SR),)
    assert torch.all(torch.isfinite(x))
