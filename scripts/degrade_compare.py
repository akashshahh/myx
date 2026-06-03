"""A/B compare each of the 8 degradations on real music.

Sine sweeps are bad listening material: most EQ/compression moves only show up
on broadband content (drums, vocals, full mix). This script applies each
degradation INDIVIDUALLY at MAX severity to a single real mixture, so you can
A/B them one at a time against the clean source.

Defaults to a random musdb-7s sample track (no file argument needed).

Run:
    python scripts/degrade_compare.py
    python scripts/degrade_compare.py --track-idx 12
    python scripts/degrade_compare.py --input path/to/mix.wav
    python scripts/degrade_compare.py --severity 0.5     # milder degradations
    python scripts/degrade_compare.py --out outputs/compare/

Output layout:
    outputs/compare/
      00_clean.wav
      01_low_excess.wav
      02_low_mid_mud.wav
      ...
      09_all_stacked_mild.wav
      summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# allow `python scripts/...` from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf

from data.degradation import (
    ISSUE_DIMENSIONS,
    apply_chain,
    sample_degradation,
    specs_to_jsonable,
)


# Ensure SSL works (python.org installer lacks system CAs) and ffmpeg is found
# (stempeg checks at import time via shutil.which, which reads PATH).
_extra_paths = ["/opt/homebrew/bin", "/usr/local/bin"]
os.environ["PATH"] = ":".join(_extra_paths + [os.environ.get("PATH", "")])
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass


class _FixedRNG:
    """Force every sampler to pick max-severity params.

    For some degradations the max-severity end is rng.uniform(...)==lo;
    for others it's ==hi. We pick the right endpoint based on a per-degradation
    flag (handled by the caller). For simplicity, callers can pass `value=1.0`
    or `value=0.0` to get the corresponding endpoint.
    """

    def __init__(self, value: float):
        self.value = value

    def uniform(self, lo, hi):
        return float(lo if self.value == 0.0 else hi if self.value == 1.0 else lo + (hi - lo) * self.value)

    def random(self):
        return float(self.value)

    def integers(self, lo, hi):
        return int(lo if self.value == 0.0 else (hi - 1) if self.value == 1.0 else lo + (hi - lo) * self.value)

    def choice(self, a, size, replace):
        raise NotImplementedError


# Per-degradation: which endpoint (0.0 or 1.0 of rng.uniform) yields max severity?
_MAX_SEVERITY_ENDPOINT = {
    "low_excess": 1.0,         # max gain = max severity
    "low_mid_mud": 1.0,
    "mid_balance": 1.0,
    "presence_lack": 0.0,      # most-negative gain = max severity (-5 dB)
    "harshness": 1.0,
    "over_compression": 1.0,   # max ratio = max severity
    "loudness_deficit": 0.0,   # most-negative gain (-15 dB) = max severity
    "dynamic_range_issue": 0.0,  # shortest release (10 ms) = max severity
}


def _max_severity_spec(name: str):
    endpoint = _MAX_SEVERITY_ENDPOINT[name]
    return sample_degradation(name, _FixedRNG(endpoint))


def _severity_spec(name: str, severity: float):
    """Interpolate between min and max endpoint to land at `severity`."""
    max_end = _MAX_SEVERITY_ENDPOINT[name]
    # If max is 1.0, value=severity lines up. If max is 0.0, invert.
    value = severity if max_end == 1.0 else 1.0 - severity
    return sample_degradation(name, _FixedRNG(value))


def _load_input(args) -> tuple[np.ndarray, int, str]:
    """Return (audio (channels, samples) float32, sr, label-for-summary)."""
    if args.input:
        path = Path(args.input).resolve()
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return np.ascontiguousarray(data.T), sr, path.name

    import musdb  # heavy import deferred

    mus = musdb.DB(download=True)
    if args.track_idx is None:
        rng = np.random.default_rng(args.seed)
        idx = int(rng.integers(0, len(mus)))
    else:
        idx = args.track_idx % len(mus)
    track = mus[idx]
    # musdb .audio is (samples, channels); convert to (channels, samples)
    audio = np.ascontiguousarray(track.audio.T.astype(np.float32))
    return audio, int(track.rate), f"musdb[{idx}] {track.name}"


def _normalize_peak(audio: np.ndarray, headroom_db: float = 0.5) -> np.ndarray:
    """Scale so peak sits at -headroom_db dBFS (only if it would clip)."""
    peak = float(np.max(np.abs(audio)))
    if peak <= 0:
        return audio
    target_peak = 10 ** (-headroom_db / 20.0)
    if peak > target_peak:
        return (audio / peak) * target_peak
    return audio


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    out = _normalize_peak(audio)
    if out.ndim == 2:
        sf.write(str(path), out.T, sr, subtype="FLOAT")
    else:
        sf.write(str(path), out, sr, subtype="FLOAT")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=None, help="Path to a WAV (default: random musdb-7s sample)")
    ap.add_argument("--track-idx", type=int, default=None, help="Index into musdb DB (default: random)")
    ap.add_argument("--seed", type=int, default=0, help="Seed when picking a random musdb track")
    ap.add_argument("--severity", type=float, default=1.0, help="0..1 — degradation strength (default: 1.0 = max)")
    ap.add_argument("--out", default=None, help="Output directory (default: outputs/compare/)")
    args = ap.parse_args()

    if not 0.0 <= args.severity <= 1.0:
        print("severity must be in [0, 1]", file=sys.stderr)
        return 2

    audio, sr, label = _load_input(args)
    outdir = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "outputs" / "compare"
    outdir.mkdir(parents=True, exist_ok=True)

    summary = {
        "source": label,
        "sample_rate": sr,
        "shape": list(audio.shape),
        "duration_s": audio.shape[-1] / sr,
        "severity": args.severity,
        "files": [],
    }

    # 0: clean reference
    clean_path = outdir / "00_clean.wav"
    _write_wav(clean_path, audio, sr)
    summary["files"].append({"file": clean_path.name, "kind": "clean"})
    print(f"clean reference -> {clean_path}")

    # 1..8: one degradation each, at requested severity
    for i, name in enumerate(ISSUE_DIMENSIONS, start=1):
        spec = _severity_spec(name, args.severity)
        out_audio = apply_chain(audio, sr, [spec])
        path = outdir / f"{i:02d}_{name}.wav"
        _write_wav(path, out_audio, sr)
        summary["files"].append(
            {"file": path.name, "kind": name, "spec": specs_to_jsonable([spec])[0]}
        )
        print(f"  {name:22s} severity={spec.severity:.2f}  -> {path.name}")

    # 9: all 8 stacked at a milder severity to hear "typical training" worst-case
    mild_severity = min(0.4, args.severity)
    stacked_chain = [_severity_spec(n, mild_severity) for n in ISSUE_DIMENSIONS]
    stacked = apply_chain(audio, sr, stacked_chain)
    stacked_path = outdir / "09_all_stacked_mild.wav"
    _write_wav(stacked_path, stacked, sr)
    summary["files"].append(
        {
            "file": stacked_path.name,
            "kind": "all_stacked_mild",
            "severity_each": mild_severity,
            "chain": specs_to_jsonable(stacked_chain),
        }
    )
    print(f"all-stacked-mild (each at severity={mild_severity}) -> {stacked_path}")

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {outdir / 'summary.json'}")

    print("\nListening order (open in your audio player as a playlist):")
    print(f"  open '{outdir}'")
    print("\nWhat to listen for at max severity:")
    print("  low_excess           : boomy/woofy sub-bass excess (50-120 Hz, +12 dB shelf)")
    print("  low_mid_mud          : muddy/boxy mids (200-450 Hz, +11 dB peak)")
    print("  mid_balance          : nasal/honky OR scooped (700-1800 Hz, ±9 dB)")
    print("  presence_lack        : dull, blanket-over-speakers (3.5-5 kHz, -8 dB shelf)")
    print("  harshness            : ice-pick, fatiguing top (2.5-5.5 kHz, +12 dB peak Q3)")
    print("  over_compression     : flat, no punch, pumping (ratio 25:1, thresh -40 dB)")
    print("  loudness_deficit     : just quiet (-20 dB gain)")
    print("  dynamic_range_issue  : crushed/distorted on transients (limiter, fast release)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
