# Demo video script — Hybrid Mastering Agent

Target length **~4.5 min** (10 min is the cap; you don't need it). Covers the four
required questions (Why → How → Use cases → What's next). Narration is written to be read
aloud; *italics* are on-screen actions / what to show.

Before recording: set `OPENROUTER_MODEL=google/gemini-2.5-pro` in `.env`, start the server
(`uvicorn api.server:app`), and have http://localhost:8000 open. Do one practice master so
the MUSDB tracks are cached and the first real take is fast. See the recording checklist at
the bottom.

---

## 0:00 — Cold open (15s)

*Screen: the web app, already loaded. Hit "Master it" on a degraded track so a render is
running in the background while you talk — you'll cut back to it.*

> "This is a mastering agent. I give it a song, it listens, decides what's wrong, fixes it,
> and — unlike every commercial tool — it tells me exactly why. Let me show you how it
> works and how I built it."

---

## 0:15 — Q1: Why I built this (45s)

*Screen: slide or just talk over the app. Optionally show a LANDR/iZotope screenshot.*

> "Mastering is the last step in making a record — balancing the tone, controlling the
> dynamics, and getting the loudness right so it translates everywhere. It's expensive,
> engineers charge per song, and it's mostly tacit expertise that's hard to learn.
>
> There are AI mastering tools — LANDR, iZotope — but they're black boxes. Audio in, louder
> audio out, no explanation. The bottleneck I wanted to attack isn't *making* the change,
> it's **diagnosis you can inspect**: a system that says what's wrong, why it chose each
> move, and lets you check the result. So I built a hybrid: a model that *hears* problems,
> and an LLM that *reasons* about how to fix them."

---

## 1:00 — Q2: How it works (1) The research / model (60s)

*Screen: a simple diagram — clean audio → degrade → train. Or show `data/degradation.py`
and the 8 dimensions briefly.*

> "The hard part is data. There's no dataset of 'here's a song and here's what's wrong with
> its master.' So I manufactured one — **synthetic supervision via degradation**.
>
> I take clean music from MUSDB18, and apply randomized *bad* processing with a known label
> — too much bass, harshness, over-compression, and so on — eight problem dimensions, each
> with a severity. Then I train a perception model — PANNs, a CNN pretrained on AudioSet —
> to **invert** that: to listen and score how much of each problem it hears.
>
> I fine-tuned it on my laptop's GPU. On held-out data it's strong on dynamics problems —
> over-compression detection is 0.96 ROC-AUC — and, honestly, weaker on subtle EQ, around
> chance. I'll come back to that, because measuring your own limitations is the point."

---

## 2:00 — Q2: How it works (2) The agent loop + live demo (90s)

*Screen: cut back to the web app and the finished result. Play Original, then Mastered.
Then expand the reasoning trace and scroll through the iterations.*

> "Here's the agent itself. The perception model gives those eight scores, and alongside it
> I run classic DSP analysis — loudness, spectral balance, dynamics, stereo width. All of
> that goes to an LLM reasoner. It picks a chain of mastering moves — EQ, compression,
> limiting, gain — Spotify's Pedalboard renders it, and then the agent **listens again**.
> Up to three passes.
>
> *[Play Original]* That's the degraded input. *[Play Mastered]* And that's the agent's
> master — louder, tighter, sitting right at the target loudness.
>
> *[Expand trace]* And this is the part the black boxes don't give you. Every iteration: the
> problems it perceived, in plain English the reasoning for its decisions, and the exact
> chain it applied. You can audit the whole thing.
>
> It's a local web app — FastAPI backend, the model loaded once, a simple front end — so you
> can drop in a demo track or upload your own and hear it in under a minute."

---

## 3:30 — Evaluation & honesty (35s)

*Screen: the results table from the README (perception AUC + recovery LUFS/LSD).*

> "I evaluated it properly. The agent reliably nails the loudness target — error under a
> quarter of a decibel. But measured against the clean reference, its tonal balance actually
> drifts — it over-processes. Two honest reasons: the weak EQ perception I mentioned, and
> the fact that my 'clean' references are unmastered stems, so hitting streaming loudness
> legitimately needs heavy limiting. A stronger LLM closes most of the gap. That's a real
> finding, not a failure I'm hiding — and it points straight at what to fix next."

---

## 4:05 — Q3: Use cases & Q4: What's next (40s)

*Screen: back to the app, or a closing slide.*

> "Who's this for? Bedroom and indie producers who can't afford a mastering engineer but
> want a result they can *understand*. It's also a teaching tool — it shows you what each
> problem is and how it's fixed — and a fast second opinion for engineers.
>
> What would I add? First, fix the EQ perception — harder EQ degradations and full-length
> training data. Second, a better target than 'unmastered stem' — reference-track matching,
> so you can say 'make it sound like this song.' And then deploy it publicly so anyone can
> try it.
>
> The whole thing — training, the agent, the evaluation, and this demo — is on GitHub, with
> an honest write-up of what works and what doesn't. Thanks for watching."

---

## Recording checklist

- [ ] `.env` → `OPENROUTER_MODEL=google/gemini-2.5-pro`, valid `OPENROUTER_API_KEY`.
- [ ] `cd mastering-agent && source .venv/bin/activate && uvicorn api.server:app`
- [ ] Browser at http://localhost:8000, window sized so the trace is readable.
- [ ] Do one warm-up master (caches MUSDB, primes the model) before the real take.
- [ ] Pick a track where the master lands near the target — e.g. try a couple and keep the
      best-sounding A/B. Leave "Degrade first" checked so the agent has obvious work.
- [ ] Audio: record system audio so the Original vs Mastered A/B is audible in the video.
- [ ] Have the README results table on a second tab for the evaluation beat.
- [ ] Show the repo / commit history briefly at the end (integrity + effort-over-time).
- [ ] Optional B-roll: `data/degradation.py`, the reasoning trace JSON, the eval script.

## The four required questions → where each is answered

| Question | Section |
|---|---|
| Q1 Why did you build it? | 0:15 |
| Q2 How does it work? (research + product + agent) | 1:00, 2:00, 3:30 |
| Q3 Use cases / impact | 4:05 |
| Q4 What more would you add? | 4:05 |
