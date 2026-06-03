"""End-to-end demo: run the full mastering agent on one track via OpenRouter.

    python scripts/master_demo.py [--track 0] [--degrade] [--target-lufs -14] [--seed 0]

Loads .env for OPENROUTER_API_KEY/OPENROUTER_MODEL, builds the (pretrained but
not-yet-fine-tuned) perception model, runs the iterative loop, prints the trace,
and writes the mastered audio to outputs/master_demo.wav.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

# Allow running from project root with `python scripts/master_demo.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# data.musdb_loader sets PATH/SSL env on import; do it before musdb work.
from data.musdb_loader import MUSDBLoader
from data.degradation import random_degradation_chain, apply_chain, specs_to_jsonable
from agent.loop import master
from agent.reasoner import Reasoner
from perception.inference import PerceptionInference
from perception.model import build_perception_model, load_finetuned_perception

PRETRAINED = "checkpoints/Cnn14_mAP=0.431.pth"
FINETUNED = "checkpoints/perception_best.pth"


def load_perception():
    """Prefer our fine-tuned 8-dim checkpoint; fall back to pretrained, then random."""
    if os.path.exists(FINETUNED):
        return load_finetuned_perception(FINETUNED), f"finetuned ({FINETUNED})"
    ckpt = PRETRAINED if os.path.exists(PRETRAINED) else None
    return build_perception_model(checkpoint_path=ckpt, freeze_backbone=True), \
        f"pretrained ({PRETRAINED})" if ckpt else "RANDOM INIT"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", type=int, default=0)
    ap.add_argument("--degrade", action="store_true", help="degrade the mixture first to give the agent work")
    ap.add_argument("--target-lufs", type=float, default=-14.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    load_dotenv()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY not set (check .env)")

    loader = MUSDBLoader(subsets="train")
    audio, sr, name = loader.get_mixture(args.track)
    print(f"track: {name}  shape={audio.shape}  sr={sr}")

    if args.degrade:
        rng = np.random.default_rng(args.seed)
        chain = random_degradation_chain(rng)
        audio = apply_chain(audio, sr, chain)
        print("pre-degraded with:", json.dumps(specs_to_jsonable(chain), indent=2))

    model, ckpt_desc = load_perception()
    perception = PerceptionInference(model, window_seconds=6.0)
    reasoner = Reasoner()
    print(f"model: {reasoner.model}   perception: {ckpt_desc}")

    result = master(audio, sr, perception, reasoner, target_lufs=args.target_lufs)

    print("\n===== TRACE =====")
    for r in result.iterations:
        print(f"\n--- iteration {r.iteration} ---")
        print("reasoning:", r.reasoning)
        print("chain:", json.dumps(r.chain))
        print("needs_another_pass:", r.needs_another_pass)
        if r.perception_delta:
            moved = {k: v for k, v in r.perception_delta.items() if abs(v) > 0.005}
            print("perception delta (|>0.005|):", moved)
    print("\nstopped:", result.stopped_reason)

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/master_demo.wav"
    sf.write(out_path, result.audio.T, sr)  # soundfile wants (samples, channels)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
