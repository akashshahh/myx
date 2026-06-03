# Hybrid Mastering Agent

CS 153 final project. An audio mastering agent that pairs a fine-tuned PANNs
CNN14 perception model with LLM reasoning (OpenRouter) in an iterative
self-listening loop. DSP is executed by Spotify's `pedalboard`.

The methodological contribution is **synthetic supervision via degradation**:
we take clean MUSDB18 mixtures, apply randomized "bad" pedalboard chains with
known type+severity labels, and train PANNs to recognize the degradations.
The same primitive ops are used at inference time.

## Setup

Python 3.10–3.12 recommended (dev uses arm64 Python 3.10.5 — torch/pedalboard/
librosa all have native wheels and numpy is pinned <2.0 for librosa/numba).

```bash
# Native arm64 Python 3.10 from python.org installer
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch, pedalboard, librosa, pyloudnorm, soundfile; print('OK')"
```

`musdb` (Phase 2) needs `ffmpeg` + `ffprobe`. On Apple Silicon they live in
`/opt/homebrew/bin`, which may not be on the default PATH:

```bash
brew install ffmpeg       # /opt/homebrew/bin/brew
export PATH="/opt/homebrew/bin:$PATH"   # add to ~/.zshrc for persistence
python -c "import musdb; print('musdb OK')"
```

Set up your env:

```bash
cp .env.example .env
# fill in OPENROUTER_API_KEY
```

## Phase 1 (current)

- `data/degradation.py` — degradation menu + label generator (the core)
- `tests/test_degradation.py` — pytest suite
- `scripts/degrade_demo.py` — sanity script

Run them:

```bash
pytest -v
python scripts/degrade_demo.py            # generates + degrades a sine sweep
python scripts/degrade_demo.py path/to/your.wav  # degrades any local file
```

## Layout

```
mastering-agent/
├── data/         degradation, MUSDB loading, PyTorch Dataset
├── perception/   PANNs CNN14 fine-tune (8-dim issue regressor)
├── agent/        analysis, reasoner (OpenRouter), executor, iterative loop
├── api/          FastAPI POST /master
├── frontend/     drag-drop UI with A/B player + reasoning log
├── eval/         synthetic recovery evaluation
└── scripts/      demo + utility scripts
```

## Issue vector (the perception → reasoner contract)

8-dim, each in `[0, 1]`, higher = more severe:

`low_excess`, `low_mid_mud`, `mid_balance`, `presence_lack`, `harshness`,
`over_compression`, `loudness_deficit`, `dynamic_range_issue`

## Status

Phase 1 in progress (requirements + degradation module). See
`/Users/akashshah/.claude/plans/project-hybrid-mastering-vivid-harp.md` for the
full plan.
