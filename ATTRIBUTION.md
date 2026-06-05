# Attribution

This project builds on the following third-party models, datasets, and libraries. The
original code in this repository is MIT-licensed (see `LICENSE`); each component below
retains its own license.

## Models & pretrained weights

- **PANNs — Pretrained Audio Neural Networks** (the perception backbone).
  Kong, Cao, Iqbal, Wang, Wang, Plumbley, *"PANNs: Large-Scale Pretrained Audio Neural
  Networks for Audio Pattern Recognition,"* IEEE/ACM TASLP, 2020.
  Code: <https://github.com/qiuqiangkong/audioset_tagging_cnn> (MIT).
  Pretrained `Cnn14_mAP=0.431.pth` checkpoint: <https://zenodo.org/records/3987831>.
  I re-headed `Cnn14` to regress 8 mastering-issue severities and fine-tuned it on
  synthetic degradations.

## Datasets

- **MUSDB18** — Rafii, Liutkus, Stöter, Mimilakis, Bittner, *"The MUSDB18 corpus for music
  separation,"* 2017. <https://sigsep.github.io/datasets/musdb.html>.
  **License: non-commercial research use only.** Used here purely as a source of clean
  music mixtures to degrade and master. The repo uses the small bundled 7-second excerpts
  auto-downloaded by the `musdb` package.

## Core libraries

- **pedalboard** — Spotify's audio-effects library; all DSP (degradation + mastering
  chains) runs through it. <https://github.com/spotify/pedalboard> (GPL-3.0).
- **pyloudnorm** — Steinmetz & Reiss, ITU-R BS.1770 loudness metering (LUFS/LRA/true peak).
  <https://github.com/csteinmetz1/pyloudnorm> (MIT).
- **torchlibrosa** — PANNs spectrogram frontend, for feature parity with the pretrained
  weights. <https://github.com/qiuqiangkong/torchlibrosa> (MIT).
- **PyTorch / torchaudio**, **librosa**, **soundfile**, **numpy**, **scipy**,
  **scikit-learn** — numerics, audio I/O, and the ROC-AUC/Spearman metrics in the eval.
- **musdb** — MUSDB18 loader/parser. <https://github.com/sigsep/sigsep-mus-db> (MIT).

## Agent / serving

- **OpenRouter** — LLM inference gateway (OpenAI-compatible) used by the reasoner at
  runtime. <https://openrouter.ai>. Accessed via the **openai** Python SDK. Default model
  `google/gemini-2.5-flash` (or `google/gemini-2.5-pro`).
- **FastAPI** + **uvicorn** — the web API and server (MIT / BSD).
- **pydantic** — strict JSON validation of LLM output (MIT).
- **wavesurfer.js** (v7, via CDN) — waveform rendering / A/B playback in the frontend (BSD-3).

## Development assistance

Built with **Claude Code** (Anthropic) as a pair-programming agent. Scope of that
assistance is disclosed in the README's "AI-usage disclosure" section.
