"""API wiring tests — no real model, no LLM, no network.

We patch `_load_perception`, `Reasoner`, `MUSDBLoader`, and `master` on the
server module so startup is cheap and `/master` returns a canned result. The
goal is to verify the HTTP plumbing (routes, multipart handling, audio
streaming), not the agent itself (covered elsewhere).
"""
from __future__ import annotations

import io

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

import api.server as server
from agent.loop import IterationRecord, MasteringResult


class _FakeLoader:
    """Stand-in for MUSDBLoader with two short stereo 'tracks'."""

    def __init__(self, *args, **kwargs):
        self._names = ["Demo - Alpha", "Demo - Beta"]

    def __len__(self):
        return len(self._names)

    @property
    def names(self):
        return list(self._names)

    def get_mixture(self, idx):
        rng = np.random.default_rng(idx)
        audio = (0.1 * rng.standard_normal((2, 44100))).astype(np.float32)
        return audio, 44100, self._names[idx]


class _FakeReasoner:
    model = "fake/model"


def _fake_master(audio, sr, perception, reasoner, target_lufs=-14.0, **kwargs):
    """Return a tiny but real MasteringResult so trace() is exercised."""
    out = (audio * 0.9).astype(np.float32)
    rec = IterationRecord(
        iteration=0,
        issue_vector={"loudness_deficit": 0.8},
        features={"loudness": {"integrated_lufs": -20.0}},
        reasoning="canned: applied a gain trim toward target.",
        chain=[{"type": "gain", "gain_db": 3.0}],
        needs_another_pass=False,
        perception_delta={"loudness_deficit": -0.3},
    )
    return MasteringResult(
        audio=out, sr=sr, target_lufs=target_lufs,
        iterations=[rec], stopped_reason="LLM signalled completion after iteration 0",
    )


def _client(monkeypatch):
    monkeypatch.setattr(server, "_load_perception", lambda: (object(), "fake"))
    monkeypatch.setattr(server, "Reasoner", _FakeReasoner)
    monkeypatch.setattr(server, "MUSDBLoader", _FakeLoader)
    monkeypatch.setattr(server, "master", _fake_master)
    return TestClient(server.app)  # context-manager entry triggers startup


def test_tracks_lists_demo_tracks(monkeypatch):
    with _client(monkeypatch) as client:
        resp = client.get("/tracks")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0] == {"id": 0, "name": "Demo - Alpha"}


def test_master_on_track_returns_trace_and_audio(monkeypatch):
    with _client(monkeypatch) as client:
        resp = client.post("/master", data={"track": 0, "target_lufs": -14.0})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["mastered_url"].startswith("/audio/")
        assert body["trace"]["iterations"][0]["reasoning"].startswith("canned")
        assert "original" in body["lufs"]

        # the advertised audio URLs actually stream non-empty WAV bytes
        for which in ("original", "mastered"):
            a = client.get(body[f"{which}_url"])
            assert a.status_code == 200
            assert a.headers["content-type"] == "audio/wav"
            assert len(a.content) > 1000


def test_master_on_upload(monkeypatch):
    buf = io.BytesIO()
    sf.write(buf, np.zeros((22050, 2), dtype=np.float32), 44100, format="WAV")
    buf.seek(0)
    with _client(monkeypatch) as client:
        resp = client.post(
            "/master",
            data={"target_lufs": -14.0},
            files={"file": ("clip.wav", buf, "audio/wav")},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["name"] == "clip.wav"


def test_master_requires_input(monkeypatch):
    with _client(monkeypatch) as client:
        resp = client.post("/master", data={"target_lufs": -14.0})
        assert resp.status_code == 400


def test_audio_bad_which_404(monkeypatch):
    with _client(monkeypatch) as client:
        resp = client.get("/audio/deadbeef/sideways")
        assert resp.status_code == 404
