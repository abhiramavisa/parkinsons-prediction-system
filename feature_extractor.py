import numpy as np
import soundfile as sf
import parselmouth
from parselmouth.praat import call


def extract_features(audio_path):
    """
    Extract vocal features using ONLY parselmouth (NO librosa).
    Compatible with Parkinson's UCI dataset features.
    """

    # ── Load audio safely ────────────────────────────────
    data, sr = sf.read(audio_path)

    if len(data) == 0:
        raise ValueError("Audio file is empty.")

    # Convert to mono if needed
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    print(f"Audio loaded: {len(data)/sr:.2f}s at {sr}Hz")

    # Convert to parselmouth Sound
    sound = parselmouth.Sound(data, sampling_frequency=sr)

    # ── Pitch ───────────────────────────────────────────
    pitch = call(sound, "To Pitch", 0.0, 75, 500)

    fo_mean = _safe(call(pitch, "Get mean", 0, 0, "Hertz"))
    fo_max  = _safe(call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic"))
    fo_min  = _safe(call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic"))

    # ── PointProcess ────────────────────────────────────
    pp = call(sound, "To PointProcess (periodic, cc)", 75, 500)

    # ── Jitter ──────────────────────────────────────────
    j_local = _praat(call, pp, "Get jitter (local)",           0, 0, 0.0001, 0.02, 1.3)
    j_abs   = _praat(call, pp, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    j_rap   = _praat(call, pp, "Get jitter (rap)",             0, 0, 0.0001, 0.02, 1.3)
    j_ppq   = _praat(call, pp, "Get jitter (ppq5)",            0, 0, 0.0001, 0.02, 1.3)
    j_ddp   = j_rap * 3.0

    # ── Shimmer ─────────────────────────────────────────
    s_local = _praat(call, [sound, pp], "Get shimmer (local)",  0, 0, 0.0001, 0.02, 1.3, 1.6)
    s_apq3  = _praat(call, [sound, pp], "Get shimmer (apq3)",   0, 0, 0.0001, 0.02, 1.3, 1.6)
    s_apq5  = _praat(call, [sound, pp], "Get shimmer (apq5)",   0, 0, 0.0001, 0.02, 1.3, 1.6)
    s_apq11 = _praat(call, [sound, pp], "Get shimmer (apq11)",  0, 0, 0.0001, 0.02, 1.3, 1.6)
    s_dda   = s_apq3 * 3.0

    if 0 < s_local < 1:
        s_db = float(-20.0 * np.log10(1.0 - s_local))
    else:
        s_db = 0.0

    # ── HNR / NHR ───────────────────────────────────────
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)

    hnr = _safe(call(harmonicity, "Get mean", 0, 0))
    nhr = 1.0 / (10 ** (hnr / 10.0)) if hnr > 0 else 1.0

    # ── Feature Vector ──────────────────────────────────
    features = [
        fo_mean,
        fo_max,
        fo_min,
        j_local,
        j_abs,
        j_rap,
        j_ppq,
        j_ddp,
        s_local,
        s_db,
        s_apq3,
        s_apq5,
        s_apq11,
        s_dda,
        nhr,
        hnr,
    ]

    # ── Sanitize ────────────────────────────────────────
    features = [0.0 if (np.isnan(v) or np.isinf(v)) else float(v) for v in features]

    print(f"Features extracted: {len(features)} values")

    return np.array(features).reshape(1, -1)


# ── Helpers ─────────────────────────────────────────────
def _safe(val, default=0.0):
    if val is None:
        return default
    try:
        if np.isnan(val):
            return default
    except Exception:
        pass
    return float(val)


def _praat(fn, *args, default=0.0):
    try:
        return _safe(fn(*args), default)
    except Exception as e:
        print(f"Praat warning: {e}")
        return default