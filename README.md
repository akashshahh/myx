# Hybrid Mastering Agent

**CS 153 final project.** An audio *mastering* agent that pairs a fine-tuned PANNs
perception model with an LLM reasoner in an iterative self-listening loop, executing all
DSP through Spotify's [`pedalboard`](https://github.com/spotify/pedalboard).

> Pick a track, the agent listens, diagnoses eight mastering problems, decides on a
> processing chain, renders it, then listens again — up to three times. A local web app
> lets you A/B the result and read exactly what it decided and why.

---

## 1. Problem & insight

**Mastering** is the final step of music production: balancing the spectrum, controlling
dynamics, and hitting a loudness target so a track translates across speakers and
streaming platforms. It is expensive (engineers charge per song), slow, and opaque —
mostly tacit expertise that's hard to learn from.

Existing "AI mastering" tools (LANDR, iZotope) are black boxes: audio in, louder audio
out, no explanation. The bottleneck I wanted to attack is **diagnosis you can inspect** —
a system that says *what* is wrong, *why* it chose each move, and lets you verify the
result.

The core methodological idea is **synthetic supervision via degradation**. Clean mastering
data with problem labels doesn't exist, so I manufacture it: take clean MUSDB18 mixtures,
apply randomized *bad* `pedalboard` chains with known type + severity, and train the
perception model to **invert** the degradation — i.e. to recognize the problems. The same
DSP primitives that create the training labels are what the agent uses to fix them at
inference time.

The second idea is **hybrid perception + reasoning**: a learned model is good at *hearing*
problems but bad at *planning* a fix; an LLM is good at planning but can't hear. So the LLM
receives both the 8-dim learned "issue vector" **and** a rich dict of raw DSP measurements,
and is explicitly invited to act on problems *outside* the eight trained dimensions
(sibilance, narrow resonances, sub rumble, stereo width…).

## 2. How it works

```
                ┌──────────── iterate up to 3× ────────────┐
   audio ──►  Perception (PANNs Cnn14, 8-dim)  ──┐
              + DSP analysis (loudness, spectrum, ├─► LLM reasoner ─► Pedalboard ─► audio'
                dynamics, stereo, detectors)  ────┘   (OpenRouter)      (render)      │
                                                                                      ▼
                                          re-listen; new issue vector feeds next pass ┘
```

- **Perception** (`perception/`) — PANNs `Cnn14` (AudioSet-pretrained) re-headed to regress
  8 severity scores in `[0,1]`. Fine-tuned on synthetic degradations. Ingests 32 kHz mono,
  windowed and mean-pooled to one vector per track.
- **Analysis** (`agent/analysis.py`) — deterministic DSP measurements (integrated LUFS, LRA,
  true peak, spectral balance, crest factor, stereo correlation, resonance/sibilance/rumble
  detectors).
- **Reasoner** (`agent/reasoner.py`) — an LLM via OpenRouter. System prompt frames the issue
  vector as high-confidence detections and the features dict as free-form evidence. Returns
  strict JSON (Pydantic-validated, one repair retry, safe no-op fallback) describing a chain
  of the 8 allowed ops (shelves, peak EQ, high/low-pass, compressor, limiter, gain).
- **Executor** (`agent/executor.py`) — renders the chain with `pedalboard`.
- **Loop** (`agent/loop.py`) — `master()` ties it together and returns the mastered audio +
  a full per-iteration trace.

**The 8 issue dimensions** (perception → reasoner contract, each `[0,1]`, higher = worse):
`low_excess`, `low_mid_mud`, `mid_balance`, `presence_lack`, `harshness`,
`over_compression`, `loudness_deficit`, `dynamic_range_issue`.

## 3. Evaluation & evidence

Full harness in `eval/synthetic_eval.py` (`python eval/synthetic_eval.py`). Two halves:

**Perception** (test subset, 300 synthetic examples). The model regresses the degradation
severity it was trained on; I report per-dimension ROC-AUC (problem present vs. absent):

| dimension | ROC-AUC | | dimension | ROC-AUC |
|---|---|---|---|---|
| over_compression | **0.96** | | low_mid_mud | 0.70 |
| dynamic_range_issue | 0.74 | | harshness | 0.69 |
| loudness_deficit | 0.74 | | low_excess | 0.57 |
| | | | mid_balance | 0.55 |
| | | | presence_lack | 0.54 |

Regression MSE **0.064** (matches training val MSE — the harness measures what the trainer
optimized), macro ROC-AUC **0.685**. **Honest read:** the model is strong on *dynamics/
loudness* problems and near-chance on *subtle EQ* problems. Detecting a 2 dB tilt in a
busy mix from 6 s of audio is genuinely hard, and that weakness propagates downstream.

**Recovery** (full agent on 5 tracks, degrade → master, vs. the clean reference):

| metric | degraded | mastered (flash) | mastered (pro) |
|---|---|---|---|
| \|LUFS − target\| (dB) | 9.14 | 1.65 | **0.18** |
| log-spectral dist. to clean | 6.18 | 11.34 | 9.93 |
| crest delta vs clean (dB) | +3.39 | −3.00 | −2.17 |

**What this shows:** the agent **reliably hits the loudness target** (error → 0.18 dB with
`gemini-2.5-pro`). But tonal balance moves *away* from the clean reference (LSD goes up) —
it **over-processes**. Two causes, both documented honestly: (1) weak EQ perception
(~0.55 AUC) misdirects tonal moves; (2) clean MUSDB stems are *unmastered*, so hitting
−14 LUFS needs heavy limiting that legitimately reduces crest factor — the "regression" is
partly the metric penalizing real mastering character. A stronger LLM closes most of the
gap monotonically, so it's partly model quality too. (LSD is made level-invariant so it
scores tonal *shape*, not the intended loudness change.)

Test suite: `pytest` (degradation, dataset, perception, agent, eval, and API wiring).

## 4. Run it

```bash
# 1. Environment (dev: arm64 Python 3.10; numpy pinned <2.0 for librosa/numba)
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. ffmpeg for MUSDB (Apple Silicon: /opt/homebrew/bin may not be on PATH)
brew install ffmpeg
export PATH="/opt/homebrew/bin:$PATH"
# python.org Python also needs a CA bundle for MUSDB's HTTPS download + OpenRouter:
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")

# 3. API key
cp .env.example .env        # then put your OPENROUTER_API_KEY in .env
```

**Web demo** (the main artifact):

```bash
uvicorn api.server:app --reload      # open http://localhost:8000
```

Pick a MUSDB demo track (optionally pre-degraded) or upload a WAV/FLAC, set a target
loudness, hit **Master it**, then A/B the players and expand the reasoning trace. A full
run is ~30–90 s (three LLM passes). For the best loudness targeting, set
`OPENROUTER_MODEL=google/gemini-2.5-pro` in `.env` (see the recovery table above).

**Other entry points:**

```bash
pytest                                       # full test suite
python eval/synthetic_eval.py                # reproduce the numbers above
python eval/synthetic_eval.py --smoke        # fast sanity pass
python scripts/master_demo.py --track 0 --degrade   # CLI run of the agent
```

Training the perception model from scratch is documented in **[TRAINING.md](TRAINING.md)**
(done on an RTX 4050 laptop; the checkpoint ships via git-LFS).

## 5. Layout

```
mastering-agent/
├── data/         degradation menu + labels, MUSDB loader, PyTorch Dataset
├── perception/   PANNs Cnn14 fine-tune (8-dim issue regressor) + inference
├── agent/        analysis, reasoner (OpenRouter), executor, iterative loop
├── eval/         synthetic perception + recovery evaluation
├── api/          FastAPI backend (POST /master, demo tracks, audio streaming)
├── frontend/     single-page A/B player + reasoning trace
└── scripts/      CLI demos
```

## 6. Limitations & future work

- **Weak subtle-EQ perception** (~0.55 AUC) → the agent over-processes tonal balance. The
  clearest next step is more/harder EQ degradations and full-length MUSDB18-HQ training.
- **No psychoacoustic/reference target** — "clean unmastered stem" is an imperfect ground
  truth for a *mastered* result. A perceptual or reference-track objective would be fairer.
- **Local only** — single-process, jobs serialized behind a lock. Public deployment
  (CPU droplet, model loaded at startup) is scoped but not done.

## 7. AI-usage disclosure

This project was built by me with **Claude Code** (Anthropic) as a pair-programming agent.
I designed the approach (synthetic supervision, the hybrid perception/reasoning loop, the
8-dim contract), trained the perception model, ran and interpreted all evaluations, and
made the engineering decisions. Claude Code assisted with scaffolding modules, writing the
evaluation harness and the FastAPI/frontend demo, drafting docs, and debugging. All results
reported here were produced by running the code and are stated honestly, including the
limitations above. The LLM *inside* the agent is accessed at runtime via OpenRouter.

Third-party code, models, and data are credited in **[ATTRIBUTION.md](ATTRIBUTION.md)**.
Licensed under [MIT](LICENSE) (my code; third-party components keep their own licenses —
note MUSDB18 is research/non-commercial).
