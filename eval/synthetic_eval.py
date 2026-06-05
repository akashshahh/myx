"""Synthetic evaluation of the mastering agent on held-out tracks.

Two independent evaluations, both built on the same known-degradation supervision
that trains the perception head (`data.degradation`):

(a) PERCEPTION — for each clean chunk we apply a *known* random degradation chain
    and ask the perception head what it sees. Because we know exactly which of the
    8 issues were injected (and how severely), we can score the head directly:
      - per-dim ROC-AUC for the binary question "was this degradation applied?"
      - regression MSE between predicted severity and the true label vector
        (the same quantity train.py minimises — ties the eval back to val_mse)
      - among the chunks where a dim WAS applied, Spearman rank-correlation
        between true severity and predicted score (does the head rank severity?)
    This needs only the local model — no LLM, no network.

(b) RECOVERY — run the full agent (`agent.loop.master`, which calls the LLM) on a
    handful of degraded chunks and measure how much closer to the *clean* reference
    the output gets, versus the degraded input:
      - log-spectral distance to clean (reconstruction quality)
      - integrated-LUFS error against the loudness target
      - crest-factor delta against clean (dynamics restoration)
    Each metric is reported for the degraded input AND the mastered output so the
    improvement (or regression) is explicit. Needs OPENROUTER_API_KEY.

NOTE ON "held-out": the bundled MUSDB sample set is small and train.py selects the
best checkpoint by val-MSE on the `test` subset, so numbers on `test` are mildly
optimistic (model selection touched it). Treated as held-out here because it is the
only split the trainer never *trained* on; flagged in the JSON output. With full
MUSDB18-HQ, carve a third split and point `--subset` at it.

CLI:
    python eval/synthetic_eval.py                       # perception + recovery (if key)
    python eval/synthetic_eval.py --no-recovery         # perception only (no network)
    python eval/synthetic_eval.py --recovery-tracks 8   # more LLM recovery samples
    python eval/synthetic_eval.py --smoke               # tiny end-to-end self-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

# data.musdb_loader sets PATH/SSL env on import; keep it first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.musdb_loader import MUSDBLoader  # noqa: E402
from data.degradation import (  # noqa: E402
    ISSUE_DIMENSIONS,
    NUM_DIMENSIONS,
    apply_chain,
    chain_to_label_vec,
    random_degradation_chain,
)

NATIVE_SR = 44100
DEFAULT_CHUNK_SECONDS = 6.0
DEFAULT_TARGET_LUFS = -14.0


# --------------------------------------------------------------------------- #
# Audio metrics
# --------------------------------------------------------------------------- #
def _to_mono(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    return a.mean(axis=0) if a.ndim == 2 else a


def log_spectral_distance(
    a: np.ndarray, b: np.ndarray, n_fft: int = 2048, hop: int = 512,
    level_invariant: bool = True,
) -> float:
    """RMS difference of log-magnitude STFTs (dB). 0.0 for identical signals.

    Measures how far apart two signals are in spectral *shape* (tonal balance),
    in dB, averaged over time and frequency.

    `level_invariant` (default) removes the mean dB offset before taking the RMS,
    so a pure loudness change costs nothing — LSD(x, k*x) == 0. This matters here:
    mastering deliberately changes level (we normalise toward a LUFS target), and
    raw LSD would be dominated by that intended loudness move rather than the
    tonal-balance differences we actually want to score. Loudness is measured
    separately by the LUFS metric.
    """
    import librosa

    am, bm = _to_mono(a), _to_mono(b)
    n = min(am.shape[0], bm.shape[0])
    am, bm = am[:n], bm[:n]
    if n < n_fft:
        n_fft = max(256, 1 << (int(n).bit_length() - 1))
        hop = n_fft // 4
    A = np.abs(librosa.stft(am, n_fft=n_fft, hop_length=hop))
    B = np.abs(librosa.stft(bm, n_fft=n_fft, hop_length=hop))
    eps = 1e-8
    diff = 20.0 * np.log10(A + eps) - 20.0 * np.log10(B + eps)
    if level_invariant:
        diff = diff - diff.mean()
    return float(np.sqrt(np.mean(diff**2)))


def crest_factor_db(audio: np.ndarray) -> float:
    """Peak-to-RMS ratio in dB. High = dynamic, low = squashed."""
    m = _to_mono(audio)
    peak = float(np.max(np.abs(m))) + 1e-12
    rms = float(np.sqrt(np.mean(m**2))) + 1e-12
    return 20.0 * np.log10(peak / rms)


def integrated_lufs(audio: np.ndarray, sr: int) -> float:
    """ITU-R BS.1770 integrated loudness. Returns nan for silent/too-short input."""
    import pyloudnorm as pyln

    m = _to_mono(audio)
    try:
        val = float(pyln.Meter(sr).integrated_loudness(m))
    except Exception:
        return float("nan")
    return val if np.isfinite(val) else float("nan")


# --------------------------------------------------------------------------- #
# Example generation (deterministic)
# --------------------------------------------------------------------------- #
def _chunk(audio: np.ndarray, sr: int, chunk_seconds: float, rng: np.random.Generator) -> np.ndarray:
    """Pick a chunk of `(channels, samples)` audio; reflect-pad if too short."""
    chunk_samples = int(chunk_seconds * sr)
    total = audio.shape[-1]
    if total >= chunk_samples:
        start = int(rng.integers(0, total - chunk_samples + 1))
        return audio[..., start : start + chunk_samples].copy()
    return np.pad(audio, ((0, 0), (0, chunk_samples - total)), mode="reflect")


def _make_example(
    audio: np.ndarray, sr: int, chunk_seconds: float, rng: np.random.Generator
):
    """clean chunk -> (degraded, clean, label_vec, applied_names). Stereo, native sr."""
    clean = _chunk(audio, sr, chunk_seconds, rng)
    chain = random_degradation_chain(rng)
    degraded = apply_chain(clean, sr, chain)
    label = chain_to_label_vec(chain)
    applied = {s.name for s in chain}
    return degraded, clean, label, applied


class _Perceiver(Protocol):
    def predict(self, audio: np.ndarray, sr: int) -> np.ndarray: ...


# --------------------------------------------------------------------------- #
# (a) Perception evaluation
# --------------------------------------------------------------------------- #
def perception_eval(
    perception: _Perceiver,
    loader: MUSDBLoader,
    chunks_per_track: int = 6,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    seed: int = 0,
    max_tracks: Optional[int] = None,
) -> dict:
    """Score the perception head against known injected degradations.

    Returns per-dim ROC-AUC + present/absent mean scores, overall regression MSE,
    and per-dim severity rank-correlation. `perception` only needs `.predict`.
    """
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr

    n_tracks = len(loader)
    if max_tracks is not None:
        n_tracks = min(n_tracks, max_tracks)

    preds, bins, sevs = [], [], []  # each (N, 8)
    for t in range(n_tracks):
        audio, sr, _ = loader.get_mixture(t)
        for c in range(chunks_per_track):
            rng = np.random.default_rng(seed * 1_000_003 + t * 101 + c)
            degraded, _clean, label, applied = _make_example(audio, sr, chunk_seconds, rng)
            pred = np.asarray(perception.predict(degraded, sr), dtype=np.float64)
            preds.append(pred)
            bins.append([1 if name in applied else 0 for name in ISSUE_DIMENSIONS])
            sevs.append(label.astype(np.float64))

    preds = np.asarray(preds)   # (N, 8)
    bins = np.asarray(bins)     # (N, 8) in {0,1}
    sevs = np.asarray(sevs)     # (N, 8) in [0,1]
    n = len(preds)

    per_dim = {}
    aucs = []
    for j, name in enumerate(ISSUE_DIMENSIONS):
        y, p, s = bins[:, j], preds[:, j], sevs[:, j]
        n_pos = int(y.sum())
        # ROC-AUC needs both classes present.
        auc = float(roc_auc_score(y, p)) if 0 < n_pos < n else None
        if auc is not None:
            aucs.append(auc)
        # Severity ranking among applied chunks (needs >=2 distinct severities).
        applied_mask = y == 1
        if applied_mask.sum() >= 3 and np.unique(s[applied_mask]).size >= 2:
            rho = float(spearmanr(s[applied_mask], p[applied_mask]).statistic)
        else:
            rho = None
        per_dim[name] = {
            "roc_auc": auc,
            "n_present": n_pos,
            "n_absent": int(n - n_pos),
            "mean_score_present": float(p[applied_mask].mean()) if n_pos else None,
            "mean_score_absent": float(p[~applied_mask].mean()) if n_pos < n else None,
            "severity_spearman": rho,
        }

    return {
        "n_examples": int(n),
        "n_tracks": int(n_tracks),
        "chunks_per_track": int(chunks_per_track),
        "regression_mse": float(np.mean((preds - sevs) ** 2)),
        "macro_roc_auc": float(np.mean(aucs)) if aucs else None,
        "per_dim": per_dim,
    }


# --------------------------------------------------------------------------- #
# (b) Recovery evaluation (full agent, needs LLM)
# --------------------------------------------------------------------------- #
def recovery_eval(
    perception,
    reasoner,
    loader: MUSDBLoader,
    n_tracks: int = 5,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    target_lufs: float = DEFAULT_TARGET_LUFS,
    seed: int = 0,
) -> dict:
    """Run the full mastering agent on degraded chunks; compare to clean reference.

    Reports, for both the degraded input and the mastered output:
      - log-spectral distance to clean — level-invariant, so it scores TONAL
        BALANCE recovery (did the agent undo the injected EQ?), not loudness.
        Lower = closer to clean's spectral shape.
      - |LUFS - target| — did it hit the loudness target. Lower = better.
      - crest delta vs clean — directional dynamics change. The clean stem is
        unmastered, so a moderately more-compressed master (negative) is normal;
        this flags whether the agent is over-squashing.
    Per-track rows plus aggregate means.
    """
    from agent.loop import master

    n_tracks = min(n_tracks, len(loader))
    rows = []
    for t in range(n_tracks):
        audio, sr, name = loader.get_mixture(t)
        rng = np.random.default_rng(seed * 7919 + t)
        degraded, clean, _label, applied = _make_example(audio, sr, chunk_seconds, rng)

        result = master(degraded, sr, perception, reasoner, target_lufs=target_lufs)
        mastered = result.audio

        clean_crest = crest_factor_db(clean)
        row = {
            "track": name,
            "applied": sorted(applied),
            "iterations": len(result.iterations),
            "stopped_reason": result.stopped_reason,
            "lsd_degraded": log_spectral_distance(degraded, clean),
            "lsd_mastered": log_spectral_distance(mastered, clean),
            "lufs_err_degraded": abs(integrated_lufs(degraded, sr) - target_lufs),
            "lufs_err_mastered": abs(integrated_lufs(mastered, sr) - target_lufs),
            "crest_delta_degraded": crest_factor_db(degraded) - clean_crest,
            "crest_delta_mastered": crest_factor_db(mastered) - clean_crest,
        }
        rows.append(row)

    def _mean(key: str) -> float:
        vals = [r[key] for r in rows if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    keys = [
        "lsd_degraded", "lsd_mastered",
        "lufs_err_degraded", "lufs_err_mastered",
        "crest_delta_degraded", "crest_delta_mastered",
    ]
    aggregate = {k: _mean(k) for k in keys}
    aggregate["lsd_improvement"] = aggregate["lsd_degraded"] - aggregate["lsd_mastered"]
    aggregate["lufs_err_improvement"] = (
        aggregate["lufs_err_degraded"] - aggregate["lufs_err_mastered"]
    )
    return {"n_tracks": int(n_tracks), "target_lufs": target_lufs,
            "aggregate": aggregate, "per_track": rows}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _print_perception(rep: dict) -> None:
    print(f"\n=== PERCEPTION ({rep['n_examples']} examples / "
          f"{rep['n_tracks']} tracks) ===")
    print(f"regression MSE (pred vs label):  {rep['regression_mse']:.4f}"
          "   (compare to training val_mse)")
    macro = rep["macro_roc_auc"]
    print(f"macro ROC-AUC (present vs absent): "
          f"{macro:.3f}" if macro is not None else "macro ROC-AUC: n/a")
    print(f"\n  {'dimension':<20} {'AUC':>6} {'sev_rho':>8} "
          f"{'score+':>7} {'score-':>7}  present/absent")
    for name, d in rep["per_dim"].items():
        auc = f"{d['roc_auc']:.3f}" if d["roc_auc"] is not None else "  -  "
        rho = f"{d['severity_spearman']:.2f}" if d["severity_spearman"] is not None else "  - "
        sp = f"{d['mean_score_present']:.3f}" if d["mean_score_present"] is not None else "  -  "
        sa = f"{d['mean_score_absent']:.3f}" if d["mean_score_absent"] is not None else "  -  "
        print(f"  {name:<20} {auc:>6} {rho:>8} {sp:>7} {sa:>7}  "
              f"{d['n_present']}/{d['n_absent']}")


def _print_recovery(rep: dict) -> None:
    a = rep["aggregate"]
    print(f"\n=== RECOVERY ({rep['n_tracks']} tracks, target "
          f"{rep['target_lufs']} LUFS) ===")
    print(f"  {'metric':<22} {'degraded':>10} {'mastered':>10} {'improved':>10}")
    print(f"  {'log-spectral dist':<22} {a['lsd_degraded']:>10.3f} "
          f"{a['lsd_mastered']:>10.3f} {a['lsd_improvement']:>+10.3f}")
    print(f"  {'|LUFS - target|':<22} {a['lufs_err_degraded']:>10.3f} "
          f"{a['lufs_err_mastered']:>10.3f} {a['lufs_err_improvement']:>+10.3f}")
    print(f"  {'crest delta vs clean':<22} {a['crest_delta_degraded']:>10.3f} "
          f"{a['crest_delta_mastered']:>10.3f} {'(directional)':>13}")
    print("    note: clean MUSDB stems are UNMASTERED, so a more-compressed master "
          "(negative crest delta) is expected, not a failure.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_perception_inference():
    from perception.inference import PerceptionInference
    from perception.model import build_perception_model, load_finetuned_perception

    finetuned = "checkpoints/perception_best.pth"
    pretrained = "checkpoints/Cnn14_mAP=0.431.pth"
    if os.path.exists(finetuned):
        model, src = load_finetuned_perception(finetuned), f"finetuned ({finetuned})"
    elif os.path.exists(pretrained):
        model = build_perception_model(checkpoint_path=pretrained, freeze_backbone=True)
        src = f"pretrained ({pretrained}) — UNTRAINED HEAD, expect chance AUC"
    else:
        model = build_perception_model(checkpoint_path=None, freeze_backbone=True)
        src = "RANDOM INIT — expect chance AUC"
    return PerceptionInference(model), src


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", default="test",
                    help="MUSDB subset to evaluate on (default: test = held-out)")
    ap.add_argument("--chunks-per-track", type=int, default=6)
    ap.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS)
    ap.add_argument("--max-tracks", type=int, default=None,
                    help="cap tracks for the perception eval (default: all)")
    ap.add_argument("--recovery-tracks", type=int, default=5,
                    help="tracks for the LLM recovery eval (0 to skip)")
    ap.add_argument("--no-recovery", action="store_true",
                    help="skip the LLM recovery eval entirely (no network)")
    ap.add_argument("--target-lufs", type=float, default=DEFAULT_TARGET_LUFS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/eval/synthetic_eval.json")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny self-test: 1 track, stub perceiver, no network")
    args = ap.parse_args()

    if args.smoke:
        return _smoke()

    perception, src = _load_perception_inference()
    print(f"perception checkpoint: {src}")
    loader = MUSDBLoader(subsets=args.subset)
    print(f"eval subset: {args.subset!r}  ({len(loader)} tracks)")

    report = {
        "checkpoint": src,
        "subset": args.subset,
        "held_out_caveat": (
            "train.py selects best checkpoint by val-MSE on the 'test' subset; "
            "numbers here are mildly optimistic if --subset=test."
        ),
        "perception": perception_eval(
            perception, loader,
            chunks_per_track=args.chunks_per_track,
            chunk_seconds=args.chunk_seconds,
            seed=args.seed,
            max_tracks=args.max_tracks,
        ),
    }
    _print_perception(report["perception"])

    run_recovery = not args.no_recovery and args.recovery_tracks > 0
    if run_recovery:
        from dotenv import load_dotenv
        load_dotenv()
        if not os.environ.get("OPENROUTER_API_KEY"):
            print("\n[recovery skipped] OPENROUTER_API_KEY not set (.env). "
                  "Run with --no-recovery to silence this.")
        else:
            from agent.reasoner import Reasoner
            reasoner = Reasoner()
            print(f"\nrunning recovery eval on {args.recovery_tracks} tracks "
                  f"via {reasoner.model} ...")
            report["recovery"] = recovery_eval(
                perception, reasoner, loader,
                n_tracks=args.recovery_tracks,
                chunk_seconds=args.chunk_seconds,
                target_lufs=args.target_lufs,
                seed=args.seed,
            )
            _print_recovery(report["recovery"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


def _smoke() -> None:
    """End-to-end self-test with a deterministic stub perceiver and one track."""
    class StubPerceiver:
        def predict(self, audio, sr):
            rng = np.random.default_rng(int(abs(audio).sum() * 1e3) % (2**32))
            return rng.uniform(0, 1, NUM_DIMENSIONS).astype(np.float32)

    loader = MUSDBLoader(subsets="test")
    rep = perception_eval(StubPerceiver(), loader, chunks_per_track=2,
                          chunk_seconds=4.0, max_tracks=2)
    assert rep["n_examples"] == 4, rep
    assert 0.0 <= rep["regression_mse"] <= 2.0
    # metric sanity
    a = np.random.default_rng(0).standard_normal((2, 16000)).astype(np.float32)
    assert log_spectral_distance(a, a) < 1e-6
    assert crest_factor_db(a) > 0
    print("SMOKE OK — perception_eval ran end-to-end and metrics are sane.")


if __name__ == "__main__":
    main()
