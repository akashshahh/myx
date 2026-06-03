# Handoff — Hybrid Mastering Agent

Last touched: 2026-05-30. Project root: `/Users/akashshah/Desktop/cs153 final/mastering-agent/`.

Full plan: `/Users/akashshah/.claude/plans/project-hybrid-mastering-vivid-harp.md`.

## Status

**Phase 1 done** — degradation library.
**Phase 2 done** — musdb loader + PyTorch Dataset.
**Phase 3 step 5 done** — perception model (PANNs Cnn14, re-headed to 8 dims).
**Phase 3 path B done** — full agent loop (steps 7-11) wired end-to-end and
verified live against OpenRouter.
**75/75 tests passing.**

### Phase 3 path B notes (2026-05-30) — agent loop built & verified live
Built steps 7-11 and confirmed a real end-to-end run via OpenRouter
(`google/gemini-2.0-flash-001`):
- `perception/inference.py` — `PerceptionInference(model)`: downmix/resample to
  32k mono, window, batch-predict, mean-pool -> (8,) vector / `predict_dict`.
- `agent/analysis.py` — `analyze(audio, sr) -> dict`: loudness (LUFS/LRA/true
  peak), dynamics (crest + per-band crest + onset density), spectral (centroid,
  flatness, rolloff, octave + 5-band RMS), stereo (corr, mid/side), detectors
  (narrow_resonance_peaks via scipy find_peaks, sibilance, sub rumble, DC,
  noise floor). Fully JSON-serializable.
- `agent/executor.py` — Pydantic discriminated-union `MasteringChain` over 8
  general-purpose ops (low_shelf, high_shelf, peak_eq, highpass, lowpass,
  compressor, limiter, gain). Param ranges clamped by the schema. `render()`
  builds a Pedalboard and renders @ 44.1k stereo.
- `agent/reasoner.py` — `Reasoner.decide(...)` -> `MasteringDecision`
  (reasoning/chain/needs_another_pass). System prompt frames issue_vector as
  high-confidence learned + invites diagnosing OUTSIDE the 8 dims from features.
  JSON mode, one repair retry, no-op fallback. Patch `_raw_completion` in tests.
- `agent/loop.py` — `master(audio, sr, perception, reasoner, target_lufs)`:
  analyze + perceive -> decide -> render -> re-measure delta, cap 3 iters,
  returns `MasteringResult` with JSON `trace()`.
- `scripts/master_demo.py` — live demo (`--track N --degrade --seed S`); writes
  `outputs/master_demo.wav`. Verified: LLM correctly diagnosed injected
  loudness/mud/rumble AND picked up a 2207 Hz resonance from the analysis dict
  (a problem outside the 8 trained dims — the design goal).
- **Dep fix**: pinned `httpx==0.27.2` (openai 1.51 passes `proxies=` to httpx,
  removed in httpx>=0.28). Added to requirements.txt.
- NOTE: with the *untrained* perception head the issue_vector hovers ~0.5 and
  deltas are tiny/noisy, so the loop never converges and hits the 3-iter cap.
  This resolves once step 6 (training) lands a real checkpoint.

### Phase 3 step 5 notes (2026-05-30)
- `perception/model.py`: faithful PANNs `Cnn14` reproduction + `PerceptionModel`
  wrapper (`build_perception_model(checkpoint_path=..., freeze_backbone=...)`).
  forward(waveform (B, samples) mono@32k) -> (B, 8) in [0,1], aligned to
  `ISSUE_DIMENSIONS`. `freeze_backbone()` leaves only `fc1`+`fc_audioset`
  trainable; `unfreeze_backbone()` for the fine-tune epochs.
- Added `torchlibrosa==0.1.0` to requirements (PANNs spectrogram frontend; exact
  feature parity with the pretrained weights). Frontend hyperparams hard-coded
  (sr 32000 / win 1024 / hop 320 / 64 mel / fmin 50 / fmax 14000) — DO NOT change.
- Pretrained `Cnn14_mAP=0.431.pth` (312 MB) downloaded to `checkpoints/`
  (gitignored). Loads with `strict=True` exact key match, `weights_only=True`
  (safe restricted unpickler — NOT weights_only=False).
- `tests/test_model.py`: 7 tests (pretrained-load test auto-skips if checkpoint
  absent).
- **Security fix**: real OpenRouter key moved out of `.env.example` (now a blank
  template) into gitignored `.env`. Key was previously exposed — consider rotating.

## Quick start (every new shell)

```bash
cd "/Users/akashshah/Desktop/cs153 final/mastering-agent"
source .venv/bin/activate
export PATH="/opt/homebrew/bin:$PATH"           # so musdb can find ffmpeg
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
pytest                                          # should be 42 passed
```

`musdb_loader.py` and `dataset.py` set those PATH/SSL env vars themselves on
import, so scripts/notebooks that go through them don't strictly need the
export — but pytest discovery and ad-hoc `python -c '...import musdb...'`
do, since they import musdb before any of our code runs.

## What's been built

```
data/
  degradation.py    Synthetic supervision: 8 degradation types, random
                    chain sampler, label-vector generator. THIS IS THE CORE
                    METHODOLOGICAL CONTRIBUTION — protect its API.
  musdb_loader.py   Thin wrapper over musdb.DB. Bundled 7-sec samples
                    (~140 MB) auto-download to ~/MUSDB18/ on first use.
  dataset.py        torch.utils.data.Dataset yielding
                    (degraded_audio_32k_mono, label_vec_8). Fresh random
                    degradation per __getitem__; deterministic mode for val.
tests/
  test_degradation.py  33 tests
  test_dataset.py       9 tests
scripts/
  degrade_demo.py      One file -> one random chain. Useful for visualizing
                       a single training example.
  degrade_compare.py   Real music -> each of 8 degradations individually,
                       side-by-side. Use this when tuning severity ranges.
outputs/compare/       The 10-file A/B comparison set you tuned ranges with.
```

### Public contracts (don't break these)

- `ISSUE_DIMENSIONS` — 8 fixed names, in fixed order. The perception head's
  output indices and the label vector indices both follow this order.
- `DegradationSpec(name, params, severity in [0,1])` — frozen dataclass.
- `degrade(audio, sr, rng) -> (degraded, label_vec, chain)`.
- `MUSDBLoader.get_mixture(idx) -> (audio (channels, samples) float32, sr, name)`.
- `DegradedAudioDataset.__getitem__ -> (mono_chunk tensor, label_vec tensor)`.

### Tuning decisions already made

- **Degradation ranges widened once** based on listening test
  (everything except `dynamic_range_issue` got bigger upper bounds).
  See `data/degradation.py` per-sampler comments for current ranges.
- **Python 3.10.5 (arm64) instead of 3.11 via pyenv** — the python.org
  installer at `/Library/Frameworks/Python.framework/Versions/3.10/` was
  already there, brew's 3.12 is x86_64 (Rosetta) and torch 2.5 has no
  x86_64 mac wheels. 3.10 still meets the spec and has torch + audio-stack
  wheels.
- **`/opt/homebrew/bin` PATH and `SSL_CERT_FILE=certifi.where()`** are
  required for python.org Python to talk to musdb / ffmpeg.
- **`chunks_per_track=8`, `chunk_seconds=6.0`** defaults in the Dataset —
  appropriate for the 6.8-sec bundled samples. Bump to `chunk_seconds=10.0`
  and `chunks_per_track=20+` when full MUSDB18-HQ becomes available.

## Useful commands

```bash
# Run the full test suite
pytest -v

# Hear one random degraded example
python scripts/degrade_demo.py --seed 42

# A/B comparison on a real musdb mixture (regenerates outputs/compare/)
python scripts/degrade_compare.py --seed 1
python scripts/degrade_compare.py --severity 0.4   # the "typical" training example

# Peek inside what the DataLoader will feed the perception model
python -c "
from data.musdb_loader import MUSDBLoader
from data.dataset import DegradedAudioDataset
from torch.utils.data import DataLoader
ds = DegradedAudioDataset(MUSDBLoader(subsets='train'), chunks_per_track=4)
dl = DataLoader(ds, batch_size=8, num_workers=0)
x, y = next(iter(dl))
print('x', x.shape, 'y', y.shape, 'label mean', y.mean(0).numpy().round(2))
"
```

## Known issues / followups

1. **musdb cache miss in DataLoader.** Each `track.audio` re-decodes the
   .stem.mp4. With shuffled indices and 8 chunks/track the cache hit rate
   in practice is ~0%. Throughput is ~5 examples/s on the M3 with 2 workers.
   Add an LRU cache to `MUSDBLoader.get_mixture` if training is dataloader-bound.
   All 94 bundled tracks fit in ~100 MB of RAM uncompressed.
2. **`.env.example` real key — FIXED 2026-05-30.** Key moved to gitignored
   `.env`; `.env.example` is now a blank template. Still worth rotating the
   OpenRouter key since it was previously exposed.
3. **Rotate the OpenAI key** that got echoed in the initial env check.
4. **Loop never converges with untrained head** — issue_vector ~0.5, deltas
   noisy, always hits 3-iter cap. Resolves once step 6 (training) runs.

## End-to-end demo (live, needs OPENROUTER_API_KEY in .env)

```bash
python scripts/master_demo.py --track 0 --degrade --seed 3   # degrade then master
python scripts/master_demo.py --track 5                       # master a clean mixture
# writes outputs/master_demo.wav + prints the full per-iteration trace
```

### Step 6 built (2026-05-30) — training, runs on the 4050 laptop
DigitalOcean had no GPUs available; decided to train on a local **RTX 4050
laptop (6 GB CUDA)** instead — training is DataLoader-bound (musdb decode
~5 ex/s), not compute-bound, so 6 GB + AMP is plenty and cloud rental would
sit idle. Built:
- `perception/train.py` — AdamW two-group (head lr 1e-4 / backbone lr 1e-5),
  MSE loss, freeze epoch 0 → unfreeze epoch 1, val on MUSDB test (deterministic
  degradations), best-by-val-MSE → `checkpoints/perception_best.pth`,
  TensorBoard → `runs/`. Device auto-detect (cuda→mps→cpu), AMP on CUDA only.
  `--smoke` runs the whole loop in ~10 s on any device (verified on CPU here).
- `data/musdb_loader.py` — added LRU cache (`cache_size=128` default, holds the
  whole bundled set; verified 55 ms→0 ms on a hit). Per-DataLoader-worker;
  `persistent_workers=True` keeps it warm across epochs. LOWER cache_size for
  full MUSDB18-HQ.
- `perception/model.py` — `load_finetuned_perception(path)` loads our 8-dim
  checkpoints (distinct from `build_perception_model` which loads the 527-class
  pretrained). `scripts/master_demo.py` now auto-prefers `perception_best.pth`.
- **`TRAINING.md`** — full 4050-laptop setup (CUDA torch wheel, ffmpeg, run cmd,
  OOM fallbacks, bringing the checkpoint back).
- Dep note: `requirements.txt` torch lines are CPU wheels (Mac); the laptop must
  `pip install torch==2.5.1 torchaudio==2.5.1 --index-url .../cu121` instead.

**Next action: run training on the 4050** per `TRAINING.md`, then copy
`checkpoints/perception_best.pth` back to the Mac — the demo/agent pick it up
automatically. After that:

## What's next — remaining plan steps

Path B is done: perception model loads (step 5), training (6), inference (7),
analysis (8), executor (9), reasoner (10), loop (11) all built. Steps 7-11
live-verified against OpenRouter; step 6 smoke-verified (full train run pending
on the laptop). Remaining:

- **Step 12 — `eval/synthetic_eval.py`** — held-out tracks w/ known degradation
  params: per-dim ROC AUC for the perception head + recovery metrics
  (log-spectral distance to clean, LUFS error, crest delta) for the full agent.
- **Step 13 — `api/server.py` + `frontend/index.html`** — `POST /master`
  (multipart in, `{audio_url, trace}` out); wavesurfer.js A/B + collapsible
  reasoning log. The loop's `MasteringResult.trace()` is already the response shape.
- **Step 14 — deploy** — CPU droplet, uvicorn behind nginx, model at startup.

## Pointers

- The plan file (linked above) has detailed specs for steps 5-13 and the
  Phase 3+ outline. Refer to it before starting the next step.
- The spec for the agent's contracts (issue vector, chain JSON schema) is
  in the original user message — also reflected in this code.
- PANNs paper / repo: https://github.com/qiuqiangkong/audioset_tagging_cnn
- Zenodo checkpoint: https://zenodo.org/records/3987831
