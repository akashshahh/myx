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

### Step 6 RAN + step 12 built (2026-06-02) — training landed, perception evaluated
- **Training run done** on the 4050: `checkpoints/perception_best.pth`,
  val_mse=0.063, 12 epochs. Committed via **git LFS** (commit `cadc0a5`).
  ⚠️ The Mac needs **git-lfs** to materialize it: a plain `git pull` leaves a
  134-byte pointer, not the 323 MB weights. git-lfs binary is in `~/.local/bin`
  (brew was blocked on a `/usr/local/share/man/man8` perms issue — `sudo chown
  -R akashshah` it to unblock brew later). `git lfs pull` fetches the real file.
- **`eval/synthetic_eval.py` built (step 12).** Two evals on held-out tracks
  with known degradation params:
  (a) PERCEPTION (local, no LLM) — per-dim ROC-AUC (present vs absent),
      regression MSE vs label (ties back to val_mse), per-dim severity Spearman.
  (b) RECOVERY (needs OPENROUTER key) — runs full `master()`, reports
      log-spectral distance to clean / |LUFS-target| / crest delta vs clean,
      for degraded input AND mastered output so improvement is explicit.
  CLI: `python eval/synthetic_eval.py [--no-recovery] [--recovery-tracks N]
  [--subset test] [--smoke]`. Writes `outputs/eval/synthetic_eval.json`.
  `tests/test_eval.py` — 8 tests (stub perceiver, no network). **83/83 passing.**
- **Perception eval result (test subset, 300 ex):** regression MSE **0.0638**
  (matches val_mse — harness validated). macro ROC-AUC **0.685**. Strong on
  dynamics — over_compression **0.96**, loudness_deficit/dynamic_range ~0.74,
  low_mid_mud 0.70, harshness 0.69. Near chance on subtle EQ — low_excess 0.57,
  mid_balance 0.55, presence_lack 0.54. Honest writeup story: model sees
  dynamics/loudness reliably, struggles on gentle tonal shelves/peaks.
  CAVEAT: trainer selected best ckpt by val-MSE on `test`, so these are mildly
  optimistic; flagged in the JSON.
- **Recovery eval RAN (2026-06-02, 5 tracks via `google/gemini-2.5-flash`):**
  LUFS error 9.14 → **1.65** (agent reliably hits the −14 LUFS target). BUT
  level-invariant log-spectral-dist to clean 6.18 → **11.34** (tonal balance
  moves *away* from clean) and crest delta 3.39 → **−3.00** (master more
  compressed than the clean mix). Read: agent **overprocesses** — heavy limiting
  to hit −14 LUFS on short *unmastered* stems + weak EQ perception (AUC ~0.55)
  → misdirected/excessive tonal moves. Honest caveat for the writeup: clean
  MUSDB stems are unmastered, so LSD-to-clean penalizes legitimate mastering
  character; a cleaner future eval degrades a *mastered* reference instead.
- **Model comparison (2026-06-03, same 5 tracks/seed):** swapping flash →
  `google/gemini-2.5-pro` improves EVERY metric monotonically (good sign the eval
  discriminates agent quality): LUFS err 1.65 → **0.18**, LSD 11.34 → **9.93**,
  crest delta −3.00 → **−2.17**. So overprocessing is partly model quality, but
  the structural finding holds: even pro's LSD (9.93) > degraded (6.18). Pro JSON
  at `outputs/eval/synthetic_eval_pro.json`; flash at `synthetic_eval.json`.
- **Metric fix:** `log_spectral_distance` is now **level-invariant** (subtracts
  mean dB offset) so LSD scores tonal SHAPE, not loudness — raw LSD was dominated
  by the intended loudness change. `level_invariant=False` recovers the old raw
  metric. Test `test_lsd_gain_invariant` pins LSD(x, k·x) ≈ 0. **84/84 passing.**
- **MODEL SWAP (2026-06-02):** `google/gemini-2.0-flash-001` was RETIRED on
  OpenRouter (404 "no endpoints found"). Changed default to
  **`google/gemini-2.5-flash`** in `.env`, `.env.example`, and
  `agent/reasoner.py::DEFAULT_MODEL`. Swap to a stronger model (e.g.
  `anthropic/claude-3.5-haiku` or `google/gemini-2.5-pro`) for final eval —
  the overprocessing finding may partly reflect the cheap flash model.

## What's next — remaining plan steps

Path B + eval done: perception model loads (step 5), training (6, RAN),
inference (7), analysis (8), executor (9), reasoner (10), loop (11),
**synthetic eval (12)** all built. Steps 7-11 live-verified against OpenRouter;
step 12 perception half verified on the real checkpoint. Remaining:

- **Step 12 follow-up** — DONE. Recovery half ran live on 5 tracks with flash and
  pro; pro beats flash monotonically (LUFS err 1.65→0.18). Numbers in README §3.
- **Step 13 — `api/server.py` + `frontend/index.html`** — DONE (2026-06-04).
  FastAPI: `GET /`, `GET /tracks`, `POST /master` (upload OR demo-track index,
  optional `degrade`), `GET /audio/{id}/{which}`. Model loaded once via lifespan,
  jobs serialized behind a `threading.Lock`. Frontend is a single self-contained
  `index.html` (wavesurfer.js v7 from CDN): track picker + upload, target-LUFS
  slider, A/B players, collapsible reasoning trace. `tests/test_api.py` covers the
  wiring (patches `master`/loader/reasoner — no LLM/network). Live-verified
  end-to-end: degrade→master→stream in ~9s, 3 iterations, trace shape matches FE.
  Run: `uvicorn api.server:app --reload` → http://localhost:8000.
  NOTE: jobs write WAVs to `outputs/api/` (gitignored). For the demo recording set
  `OPENROUTER_MODEL=google/gemini-2.5-pro` for much better loudness targeting.
- **Submission docs — DONE** — README rewritten (problem/architecture/results/
  run/AI-disclosure), `LICENSE` (MIT) + `ATTRIBUTION.md` (PANNs/MUSDB/pedalboard/…)
  added. Suite at 89 passing.
- **Step 14 — deploy (OPTIONAL, future work)** — CPU droplet, uvicorn behind nginx,
  model at startup. Deliberately deferred; local run is enough for the demo video.

## Pointers

- The plan file (linked above) has detailed specs for steps 5-13 and the
  Phase 3+ outline. Refer to it before starting the next step.
- The spec for the agent's contracts (issue vector, chain JSON schema) is
  in the original user message — also reflected in this code.
- PANNs paper / repo: https://github.com/qiuqiangkong/audioset_tagging_cnn
- Zenodo checkpoint: https://zenodo.org/records/3987831
