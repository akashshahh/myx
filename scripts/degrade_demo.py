"""Sanity script for data/degradation.py.

Run with:
    python scripts/degrade_demo.py                     # generates a sine sweep
    python scripts/degrade_demo.py path/to/track.wav   # degrades a real file
    python scripts/degrade_demo.py track.wav --seed 7  # reproducible

Writes `<stem>_degraded.wav` and `<stem>_degraded.json` next to the input
(or under outputs/ for the synthesized sweep).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root with `python scripts/degrade_demo.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf

from data.degradation import degrade, specs_to_jsonable


SR_DEFAULT = 44100


def _synth_sweep(seconds: float = 6.0, sr: int = SR_DEFAULT) -> np.ndarray:
    """Log-frequency sweep 50 Hz -> 12 kHz, stereo (2, N), float32, peak -3 dBFS."""
    n = int(seconds * sr)
    t = np.linspace(0, seconds, n, endpoint=False)
    f0, f1 = 50.0, 12000.0
    # log sweep phase
    k = (f1 / f0) ** (1.0 / seconds)
    phase = 2 * np.pi * f0 * (k ** t - 1.0) / np.log(k)
    mono = (0.707 * np.sin(phase)).astype(np.float32)
    return np.stack([mono, mono], axis=0)


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load as (channels, samples) float32."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # soundfile returns (samples, channels) — transpose to (channels, samples)
    audio = np.ascontiguousarray(data.T)
    return audio, sr


def _save_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    # soundfile wants (samples, channels) for stereo or (samples,) for mono
    if audio.ndim == 2:
        sf.write(str(path), audio.T, sr, subtype="FLOAT")
    else:
        sf.write(str(path), audio, sr, subtype="FLOAT")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to a WAV/FLAC. If omitted, a 6-sec log sweep is synthesized.",
    )
    ap.add_argument("--seed", type=int, default=None, help="RNG seed (default: random)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.input is None:
        audio = _synth_sweep()
        sr = SR_DEFAULT
        outdir = Path(__file__).resolve().parents[1] / "outputs"
        outdir.mkdir(parents=True, exist_ok=True)
        clean_path = outdir / "sweep_clean.wav"
        _save_wav(clean_path, audio, sr)
        stem = outdir / "sweep"
        print(f"synthesized sine sweep -> {clean_path}")
    else:
        in_path = Path(args.input).resolve()
        if not in_path.exists():
            print(f"file not found: {in_path}", file=sys.stderr)
            return 2
        audio, sr = _load_wav(in_path)
        stem = in_path.with_suffix("")
        print(f"loaded {in_path}  sr={sr}  shape={audio.shape}")

    degraded, label, chain = degrade(audio, sr, rng)
    wav_out = stem.parent / f"{stem.name}_degraded.wav"
    json_out = stem.parent / f"{stem.name}_degraded.json"

    # Headroom-safe: peak-limit if degradation pushed past 0 dBFS
    peak = float(np.max(np.abs(degraded)))
    if peak > 1.0:
        degraded = degraded / peak * 0.99
        print(f"peak {peak:.3f} > 1.0; rescaled to 0.99 to avoid clipping in WAV")

    _save_wav(wav_out, degraded, sr)
    json_out.write_text(
        json.dumps(
            {
                "input": str(args.input) if args.input else "synthesized_sweep",
                "sample_rate": sr,
                "seed": args.seed,
                "label_vector": label.tolist(),
                "chain": specs_to_jsonable(chain),
            },
            indent=2,
        )
    )

    print(f"degraded audio -> {wav_out}")
    print(f"spec JSON     -> {json_out}")
    print("\nchain applied:")
    for spec in chain:
        params_short = ", ".join(
            f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in spec.params.items()
        )
        print(f"  {spec.name:22s} severity={spec.severity:.2f}  [{params_short}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
