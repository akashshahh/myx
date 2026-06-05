"""FastAPI backend for the hybrid mastering agent — local web demo.

One process, model loaded once at startup, jobs serialized behind a lock (the
torch model isn't thread-safe and `master()` is long-running ~30-90s).

    uvicorn api.server:app --reload    # then open http://localhost:8000

Endpoints:
  GET  /                      -> the single-page frontend
  GET  /tracks                -> built-in MUSDB demo tracks [{id, name}, ...]
  POST /master                -> run the agent on an uploaded file OR a demo track
  GET  /audio/{id}/{which}    -> stream the original/mastered WAV produced by /master

`/master` returns the full per-iteration reasoning trace so the frontend can show
what the agent decided and why.
"""
from __future__ import annotations

import io
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

# data.musdb_loader sets PATH/SSL env on import; keep these imports up top.
from data.musdb_loader import MUSDBLoader
from agent.loop import master
from agent.reasoner import Reasoner
from perception.inference import PerceptionInference
from perception.model import build_perception_model, load_finetuned_perception

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend" / "index.html"
PRETRAINED = ROOT / "checkpoints" / "Cnn14_mAP=0.431.pth"
FINETUNED = ROOT / "checkpoints" / "perception_best.pth"
OUT_DIR = ROOT / "outputs" / "api"

PIPELINE_SR = 44100          # everything downstream assumes 44.1 kHz stereo float32
MAX_DEMO_TRACKS = 10
MAX_UPLOAD_SECONDS = 60.0    # keep a stray huge upload from hanging the demo

load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------------------
# State loaded once at startup. Built lazily so importing the module (e.g. in
# tests that patch `master`) is cheap and needs no checkpoint/key.
# ---------------------------------------------------------------------------
_perception: Optional[PerceptionInference] = None
_reasoner: Optional[Reasoner] = None
_loader: Optional[MUSDBLoader] = None
_ckpt_desc: str = "uninitialised"
_job_lock = threading.Lock()


def _load_perception() -> tuple[PerceptionInference, str]:
    """Fine-tuned 8-dim checkpoint if present, else pretrained, else random."""
    if FINETUNED.exists():
        model = load_finetuned_perception(str(FINETUNED))
        desc = f"finetuned ({FINETUNED.name})"
    elif PRETRAINED.exists():
        model = build_perception_model(checkpoint_path=str(PRETRAINED), freeze_backbone=True)
        desc = f"pretrained ({PRETRAINED.name})"
    else:
        model = build_perception_model(checkpoint_path=None, freeze_backbone=True)
        desc = "RANDOM INIT (no checkpoint found)"
    return PerceptionInference(model, window_seconds=6.0), desc


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _perception, _reasoner, _loader, _ckpt_desc
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _perception, _ckpt_desc = _load_perception()
    _reasoner = Reasoner()
    _loader = MUSDBLoader(subsets="test")
    print(f"[startup] perception: {_ckpt_desc}  model: {_reasoner.model}  "
          f"tracks: {len(_loader)}")
    yield


app = FastAPI(title="Hybrid Mastering Agent", lifespan=_lifespan)


# ---------------------------------------------------------------------------
# Audio decoding for uploads
# ---------------------------------------------------------------------------
def _decode_upload(raw: bytes) -> tuple[np.ndarray, int]:
    """Bytes of an audio file -> ((channels, samples) float32, sr=44100).

    Tries soundfile first (wav/flac/ogg), falls back to librosa+audioread for
    anything else (mp3/m4a via ffmpeg). Forces stereo at 44.1 kHz and trims to
    MAX_UPLOAD_SECONDS.
    """
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        audio = data.T  # soundfile gives (samples, channels)
    except Exception:
        import librosa
        audio, sr = librosa.load(io.BytesIO(raw), sr=None, mono=False)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[None, :]

    if audio.shape[0] == 1:                       # mono -> fake stereo
        audio = np.repeat(audio, 2, axis=0)
    elif audio.shape[0] > 2:                       # downmix >2ch to stereo
        audio = audio[:2]

    if sr != PIPELINE_SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=PIPELINE_SR, res_type="soxr_hq")
        sr = PIPELINE_SR

    max_samples = int(MAX_UPLOAD_SECONDS * sr)
    if audio.shape[1] > max_samples:
        audio = audio[:, :max_samples]

    return np.ascontiguousarray(audio.astype(np.float32)), sr


def _lufs(audio: np.ndarray, sr: int) -> Optional[float]:
    """Integrated LUFS, reusing the eval harness; None if unmeasurable."""
    from eval.synthetic_eval import integrated_lufs
    val = integrated_lufs(audio, sr)
    return None if val is None or np.isnan(val) else round(float(val), 2)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def index() -> FileResponse:
    if not FRONTEND.exists():
        raise HTTPException(500, "frontend/index.html not found")
    return FileResponse(str(FRONTEND))


@app.get("/tracks")
def tracks() -> list[dict]:
    if _loader is None:
        raise HTTPException(503, "loader not initialised")
    names = _loader.names[:MAX_DEMO_TRACKS]
    return [{"id": i, "name": name} for i, name in enumerate(names)]


@app.post("/master")
def run_master(
    track: Optional[int] = Form(None),
    target_lufs: float = Form(-14.0),
    degrade: bool = Form(False),
    file: Optional[UploadFile] = File(None),
) -> dict:
    """Master an uploaded file (if provided) or a built-in demo `track` index."""
    if _perception is None or _reasoner is None or _loader is None:
        raise HTTPException(503, "server still initialising")

    # --- resolve input audio ---
    if file is not None:
        raw = file.file.read()
        if not raw:
            raise HTTPException(400, "uploaded file is empty")
        audio, sr = _decode_upload(raw)
        name = file.filename or "upload"
    elif track is not None:
        if track < 0 or track >= len(_loader):
            raise HTTPException(400, f"track {track} out of range")
        audio, sr, name = _loader.get_mixture(track)
        audio = np.array(audio, dtype=np.float32, copy=True)  # cache buffer is read-only
    else:
        raise HTTPException(400, "provide either a file upload or a track index")

    # Optionally pre-degrade a clean demo track so the agent has real work to do.
    if degrade and file is None:
        from data.degradation import random_degradation_chain, apply_chain
        rng = np.random.default_rng(0)
        audio = apply_chain(audio, sr, random_degradation_chain(rng))

    # --- run the agent (serialized: model isn't thread-safe) ---
    with _job_lock:
        result = master(audio, sr, _perception, _reasoner, target_lufs=target_lufs)

    # --- persist original + mastered for A/B playback ---
    job_id = uuid.uuid4().hex
    sf.write(str(OUT_DIR / f"{job_id}_original.wav"), audio.T, sr)
    sf.write(str(OUT_DIR / f"{job_id}_mastered.wav"), result.audio.T, sr)

    return {
        "id": job_id,
        "name": name,
        "target_lufs": target_lufs,
        "original_url": f"/audio/{job_id}/original",
        "mastered_url": f"/audio/{job_id}/mastered",
        "lufs": {"original": _lufs(audio, sr), "mastered": _lufs(result.audio, sr)},
        "trace": result.trace(),
    }


@app.get("/audio/{job_id}/{which}")
def audio(job_id: str, which: str) -> FileResponse:
    if which not in ("original", "mastered"):
        raise HTTPException(404, "which must be 'original' or 'mastered'")
    if not job_id.isalnum():                       # job ids are uuid4 hex
        raise HTTPException(400, "bad job id")
    path = OUT_DIR / f"{job_id}_{which}.wav"
    if not path.exists():
        raise HTTPException(404, "audio not found (expired or never created)")
    return FileResponse(str(path), media_type="audio/wav")
