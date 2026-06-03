"""Rich DSP diagnostic surface for the mastering agent.

`analyze(audio, sr) -> dict` returns a fully JSON-serializable dictionary of
mix/master measurements. The 8 trained issue dimensions tell the LLM what the
*learned* model is confident about; this dict lets the LLM diagnose everything
*outside* those 8 (sibilance, narrow resonances, sub rumble, stereo problems,
dullness, pumping, DC offset, noise floor, ...) and choose corrective ops
accordingly. Every value here is heuristic DSP, not learned.

Audio convention: `(channels, samples)` or `(samples,)` float32. Internally we
keep a mono view for spectral/dynamics work and use both channels for stereo
metrics.
"""
from __future__ import annotations

import math

import librosa
import numpy as np
import pyloudnorm as pyln
from scipy import signal

_EPS = 1e-10


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _db(x: float) -> float:
    return float(20.0 * math.log10(max(abs(x), _EPS)))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + _EPS))


def _to_stereo_mono(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (mono, left, right) float32 1-D arrays. Mono is the channel mean."""
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        return a, a, a
    if a.shape[0] == 1:
        m = a[0]
        return m, m, m
    left, right = a[0], a[1]
    mono = a.mean(axis=0)
    return mono.astype(np.float32), left, right


def _safe(x: float, default: float = -120.0) -> float:
    return float(x) if np.isfinite(x) else default


# --------------------------------------------------------------------------
# loudness
# --------------------------------------------------------------------------
def _loudness(mono: np.ndarray, left: np.ndarray, right: np.ndarray, sr: int) -> dict:
    meter = pyln.Meter(sr)
    # pyloudnorm wants (samples,) or (samples, channels)
    data = np.stack([left, right], axis=1) if not np.shares_memory(left, right) else mono
    try:
        integrated = _safe(meter.integrated_loudness(data))
    except Exception:
        integrated = -120.0

    # short-term loudness: 3 s windows, 1 s hop; max + range (simplified LRA)
    win = int(3.0 * sr)
    hop = int(1.0 * sr)
    st_vals = []
    if mono.shape[0] >= win:
        for start in range(0, mono.shape[0] - win + 1, hop):
            seg = data[start : start + win] if data.ndim == 2 else mono[start : start + win]
            try:
                lv = meter.integrated_loudness(seg)
                if np.isfinite(lv):
                    st_vals.append(float(lv))
            except Exception:
                pass
    if st_vals:
        st_arr = np.array(st_vals)
        short_term_max = float(st_arr.max())
        # EBU-style LRA ~ 95th - 10th percentile of gated short-term loudness
        lra = float(np.percentile(st_arr, 95) - np.percentile(st_arr, 10))
    else:
        short_term_max = integrated
        lra = 0.0

    sample_peak_db = _db(np.max(np.abs(np.stack([left, right]))))
    # true peak via 4x oversampling per channel
    tp = 0.0
    for ch in (left, right):
        up = signal.resample_poly(ch, 4, 1)
        tp = max(tp, float(np.max(np.abs(up))))
    true_peak_db = _db(tp)

    return {
        "integrated_lufs": round(integrated, 2),
        "short_term_lufs_max": round(short_term_max, 2),
        "lra": round(lra, 2),
        "sample_peak_dbfs": round(sample_peak_db, 2),
        "true_peak_dbtp": round(true_peak_db, 2),
    }


# --------------------------------------------------------------------------
# dynamics
# --------------------------------------------------------------------------
_CREST_BANDS = [(20, 120), (120, 500), (500, 2000), (2000, 6000), (6000, 20000)]


def _crest_db(x: np.ndarray) -> float:
    peak = float(np.max(np.abs(x)) + _EPS)
    return round(_db(peak) - _db(_rms(x)), 2)


def _dynamics(mono: np.ndarray, sr: int) -> dict:
    crest = _crest_db(mono)
    nyq = sr / 2.0
    per_band = {}
    for lo, hi in _CREST_BANDS:
        hi_c = min(hi, nyq * 0.999)
        if lo >= hi_c:
            per_band[f"{lo}-{hi}Hz"] = 0.0
            continue
        sos = signal.butter(4, [lo / nyq, hi_c / nyq], btype="band", output="sos")
        band = signal.sosfilt(sos, mono)
        per_band[f"{lo}-{hi}Hz"] = _crest_db(band)

    # onset density: onsets per second
    try:
        onsets = librosa.onset.onset_detect(y=mono, sr=sr, units="time")
        duration = mono.shape[0] / sr
        onset_density = round(len(onsets) / duration, 3) if duration > 0 else 0.0
    except Exception:
        onset_density = 0.0

    return {
        "crest_factor_db": crest,
        "per_band_crest_db": per_band,
        "onset_density_per_s": onset_density,
    }


# --------------------------------------------------------------------------
# spectral balance
# --------------------------------------------------------------------------
_OCTAVE_CENTERS = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
_FIVE_BAND_EDGES = [0, 60, 250, 1000, 4000, 10000, 24000]
_FIVE_BAND_NAMES = ["sub", "low", "low_mid", "mid", "presence", "air"]


def _power_spectrum(mono: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Long-term power spectrum: mean of |STFT|^2 over time. Returns (freqs, power)."""
    n_fft = 4096 if mono.shape[0] >= 4096 else 1024
    S = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=n_fft // 4)) ** 2
    power = S.mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return freqs, power


def _band_energy_db(freqs: np.ndarray, power: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return -120.0
    return round(10.0 * math.log10(float(power[mask].sum()) + _EPS), 2)


def _spectral(mono: np.ndarray, sr: int, freqs: np.ndarray, power: np.ndarray) -> dict:
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=mono, sr=sr)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=mono)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=mono, sr=sr, roll_percent=0.95)))

    octave = {
        f"{int(c)}Hz": _band_energy_db(freqs, power, c / math.sqrt(2), c * math.sqrt(2))
        for c in _OCTAVE_CENTERS
    }
    five = {
        _FIVE_BAND_NAMES[i]: _band_energy_db(freqs, power, _FIVE_BAND_EDGES[i], _FIVE_BAND_EDGES[i + 1])
        for i in range(len(_FIVE_BAND_NAMES))
    }
    return {
        "spectral_centroid_hz": round(centroid, 1),
        "spectral_flatness": round(flatness, 4),
        "hf_rolloff_hz": round(rolloff, 1),
        "octave_band_db": octave,
        "five_band_db": five,
    }


# --------------------------------------------------------------------------
# stereo
# --------------------------------------------------------------------------
def _stereo(left: np.ndarray, right: np.ndarray) -> dict:
    if np.shares_memory(left, right):
        return {"lr_correlation": 1.0, "mid_side_ratio_db": None, "is_mono": True}
    corr = float(np.corrcoef(left, right)[0, 1]) if np.std(left) > 0 and np.std(right) > 0 else 1.0
    mid = (left + right) / 2.0
    side = (left - right) / 2.0
    ms_ratio = round(_db(_rms(mid)) - _db(_rms(side)), 2)
    return {
        "lr_correlation": round(_safe(corr, 1.0), 3),
        "mid_side_ratio_db": ms_ratio,
        "is_mono": False,
    }


# --------------------------------------------------------------------------
# targeted problem detectors
# --------------------------------------------------------------------------
def _resonance_peaks(freqs: np.ndarray, power: np.ndarray, max_peaks: int = 6) -> list:
    """Narrow resonances from a smoothed long-term log spectrum (room modes,
    ringing). Returns [{hz, prominence_db}] sorted by prominence."""
    with np.errstate(divide="ignore"):
        logp = 10.0 * np.log10(power + _EPS)
    # smooth, then look for peaks standing proud of the local trend
    smooth = signal.savgol_filter(logp, window_length=min(31, len(logp) // 2 * 2 + 1), polyorder=3) \
        if len(logp) > 33 else logp
    detrended = logp - smooth
    peaks, props = signal.find_peaks(detrended, prominence=3.0, distance=3)
    out = []
    for i, p in enumerate(peaks):
        f = float(freqs[p])
        if f < 80 or f > 16000:
            continue
        out.append({"hz": round(f, 1), "prominence_db": round(float(props["prominences"][i]), 2)})
    out.sort(key=lambda d: d["prominence_db"], reverse=True)
    return out[:max_peaks]


def _detectors(mono: np.ndarray, left: np.ndarray, right: np.ndarray, sr: int,
               freqs: np.ndarray, power: np.ndarray) -> dict:
    sib_hi = _band_energy_db(freqs, power, 5000, 8000)
    sib_lo = _band_energy_db(freqs, power, 1000, 5000)
    sibilance_index = round(sib_hi - sib_lo, 2)

    sub_rumble_db = _band_energy_db(freqs, power, 0, 30)

    if np.shares_memory(left, right):
        dc = [round(float(np.mean(mono)), 6)]
    else:
        dc = [round(float(np.mean(left)), 6), round(float(np.mean(right)), 6)]

    # noise floor: quietest 100 ms RMS
    win = max(1, int(0.1 * sr))
    if mono.shape[0] >= win:
        n = mono.shape[0] // win
        seg_rms = [_rms(mono[i * win : (i + 1) * win]) for i in range(n)]
        noise_floor_db = round(_db(min(seg_rms)), 2)
    else:
        noise_floor_db = round(_db(_rms(mono)), 2)

    return {
        "narrow_resonance_peaks": _resonance_peaks(freqs, power),
        "sibilance_index_db": sibilance_index,
        "sub_rumble_db": sub_rumble_db,
        "dc_offset": dc,
        "noise_floor_db": noise_floor_db,
    }


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def analyze(audio: np.ndarray, sr: int) -> dict:
    """Full diagnostic dictionary (JSON-serializable) for one audio buffer."""
    mono, left, right = _to_stereo_mono(audio)
    if mono.shape[0] == 0:
        raise ValueError("empty audio")

    freqs, power = _power_spectrum(mono, sr)
    return {
        "duration_s": round(mono.shape[0] / sr, 2),
        "sample_rate": int(sr),
        "loudness": _loudness(mono, left, right, sr),
        "dynamics": _dynamics(mono, sr),
        "spectral": _spectral(mono, sr, freqs, power),
        "stereo": _stereo(left, right),
        "detectors": _detectors(mono, left, right, sr, freqs, power),
    }
