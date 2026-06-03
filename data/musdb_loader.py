"""MUSDB18 wrapper.

By default uses the small bundled 7-sec excerpts (~140 MB, auto-downloaded
on first call). Pass `root=` to point at full MUSDB18 / MUSDB18-HQ once you
have access.

The public surface is intentionally tiny: `len`, `names`, `get_mixture`,
`iter_mixtures`. Audio is returned as `(channels, samples)` float32 at the
file's native sample rate (44.1 kHz for both bundled and full MUSDB18).
"""
from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


def _ensure_env() -> None:
    """python.org Python lacks system CA bundle and Apple-Silicon ffmpeg.

    stempeg checks for ffmpeg on PATH at import time and musdb downloads over
    HTTPS, so this must run before any musdb-related import.
    """
    for p in ("/opt/homebrew/bin", "/usr/local/bin"):
        if p not in os.environ.get("PATH", ""):
            os.environ["PATH"] = p + ":" + os.environ.get("PATH", "")
    if "SSL_CERT_FILE" not in os.environ:
        try:
            import certifi
            os.environ["SSL_CERT_FILE"] = certifi.where()
        except ImportError:
            pass


class MUSDBLoader:
    """Wraps musdb.DB and exposes mixture audio as (channels, samples) float32.

    Args:
        root: Path to a full MUSDB18 / MUSDB18-HQ dataset. If None, uses the
            bundled 7-sec sample dataset (auto-downloaded on first use).
        subsets: 'train', 'test', or None for both.
        is_wav: True when pointing at MUSDB18-HQ (uncompressed WAV variant).
        cache_size: How many decoded mixtures to keep in an in-memory LRU cache.
            musdb re-decodes the .stem.mp4 on every `track.audio` access, which
            otherwise makes the cache hit rate ~0% under a shuffled DataLoader.
            The default (128) holds the entire bundled 7-sec set (~120 MB) so
            every chunk after the first decode is a cache hit. LOWER this for
            full MUSDB18-HQ (minutes-long tracks are ~85 MB each). Set 0/None to
            disable caching.
    """

    def __init__(
        self,
        root: Optional[str | Path] = None,
        subsets: Optional[str] = None,
        is_wav: bool = False,
        cache_size: Optional[int] = 128,
    ):
        _ensure_env()
        import musdb

        kwargs: dict = {}
        if root is None:
            kwargs["download"] = True
        else:
            kwargs["root"] = str(root)
            kwargs["is_wav"] = is_wav
        if subsets is not None:
            kwargs["subsets"] = subsets
        self._db = musdb.DB(**kwargs)
        self._cache_size = int(cache_size) if cache_size else 0
        self._cache: "OrderedDict[int, tuple[np.ndarray, int, str]]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._db)

    @property
    def names(self) -> list[str]:
        return [t.name for t in self._db]

    def get_mixture(self, idx: int) -> tuple[np.ndarray, int, str]:
        """Return (audio (channels, samples) float32, sample_rate, track_name).

        The returned array is shared with the LRU cache — treat it as read-only
        (slice/copy before mutating). The Dataset already does (it slices a view
        then renders through pedalboard, which allocates a fresh buffer)."""
        if self._cache_size and idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]

        track = self._db[idx]
        # musdb's .audio is (samples, channels); transpose for our (channels, samples) convention
        audio = np.ascontiguousarray(track.audio.T.astype(np.float32))
        entry = (audio, int(track.rate), track.name)

        if self._cache_size:
            self._cache[idx] = entry
            self._cache.move_to_end(idx)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return entry

    def iter_mixtures(self) -> Iterator[tuple[np.ndarray, int, str]]:
        for i in range(len(self)):
            yield self.get_mixture(i)

    def min_duration(self) -> float:
        """Length of the shortest track (seconds). Useful for chunk sizing."""
        return min(self._db[i].duration for i in range(len(self)))
