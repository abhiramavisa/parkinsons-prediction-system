import numpy as np
import librosa


def extract_features(audio_path):
    """
    Stable feature extraction (Render-safe version)

    Uses librosa instead of parselmouth to avoid server crashes.
    Keeps feature size = 16 to match trained model input.
    """

    # Load audio
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    if len(y) == 0:
        raise ValueError("Audio file is empty.")

    print(f"Audio loaded: {len(y)/sr:.2f}s at {sr}Hz")

    # ── MFCC (13 features) ─────────────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # ── Additional spectral features ───────────────────
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))

    # ── Combine features (16 total) ────────────────────
    features = list(mfcc_mean) + [
        spectral_centroid,
        spectral_bandwidth,
        zero_crossing
    ]

    # Ensure exactly 16 features
    features = features[:16]

    # ── Sanitize values ────────────────────────────────
    features = [
        0.0 if (np.isnan(v) or np.isinf(v)) else float(v)
        for v in features
    ]

    print(f"Features extracted: {len(features)} values")

    return np.array(features).reshape(1, -1)