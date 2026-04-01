import os, numpy as np, pandas as pd
import parselmouth
from parselmouth.praat import call
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import joblib

# ── PATHS ───────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PD_FOLDER = os.path.join(BASE_DIR, 'PD_AH')
HC_FOLDER = os.path.join(BASE_DIR, 'HC_AH')
CSV_PATH  = os.path.join(BASE_DIR, 'parkinsons.data')

# ── FEATURE EXTRACTION ─────────────────────────
def extract(audio_path):
    try:
        sound = parselmouth.Sound(audio_path)

        pitch = call(sound, "To Pitch", 0.0, 75, 500)

        fo   = call(pitch, "Get mean", 0, 0, "Hertz") or 0
        fohi = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic") or 0
        folo = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic") or 0

        pp     = call(sound, "To PointProcess (periodic, cc)", 75, 500)
        jit    = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) or 0
        jitabs = call(pp, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3) or 0
        rap    = call(pp, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3) or 0
        ppq    = call(pp, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3) or 0
        ddp    = rap * 3.0

        shi   = call([sound, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) or 0
        apq3  = call([sound, pp], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6) or 0
        apq5  = call([sound, pp], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6) or 0
        apq11 = call([sound, pp], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6) or 0
        dda   = apq3 * 3.0
        shidb = float(-20.0 * np.log10(1.0 - shi)) if 0 < shi < 1 else 0.0

        harm = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr  = call(harm, "Get mean", 0, 0) or 0
        nhr  = 1.0 / (10 ** (hnr / 10.0)) if hnr > 0 else 1.0

        return [fo, fohi, folo, jit, jitabs, rap, ppq, ddp,
                shi, shidb, apq3, apq5, apq11, dda, nhr, hnr]

    except:
        return None


# ── LOAD NUMERICAL DATA ───────────────────────
df = pd.read_csv(CSV_PATH)

FEATURE_COLUMNS = [
    'MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)',
    'MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP',
    'MDVP:PPQ','Jitter:DDP','MDVP:Shimmer',
    'MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5',
    'MDVP:APQ','Shimmer:DDA','NHR','HNR'
]

X_num = df[FEATURE_COLUMNS].values
y_num = df['status'].values


# ── LOAD AUDIO DATA ───────────────────────────
rows, labels = [], []

for folder, label in [(PD_FOLDER, 1), (HC_FOLDER, 0)]:
    for fname in os.listdir(folder):
        if fname.endswith('.wav'):
            feats = extract(os.path.join(folder, fname))
            if feats:
                rows.append(feats)
                labels.append(label)

X_audio = np.array(rows)
y_audio = np.array(labels)


# ── COMBINE ──────────────────────────────────
X = np.vstack((X_num, X_audio))
y = np.hstack((y_num, y_audio))


# 🔥 BALANCE DATASET (VERY IMPORTANT)
healthy = X[y == 0]
parkinsons = X[y == 1]

healthy_up = resample(healthy,
                      replace=True,
                      n_samples=len(parkinsons),
                      random_state=42)

X = np.vstack((healthy_up, parkinsons))
y = np.array([0]*len(healthy_up) + [1]*len(parkinsons))


print(f"Balanced: Healthy={len(healthy_up)} Parkinson={len(parkinsons)}")


# ── SCALE ────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ── MODEL ────────────────────────────────────
model = GradientBoostingClassifier(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=3,
    subsample=0.8,
    random_state=42
)

model.fit(X_scaled, y)


# ── SMART THRESHOLD (REDUCE FALSE POSITIVES) ─
probs = model.predict_proba(X_scaled)[:, 1]

best_thr = 0.5
best_score = 0

for thr in np.arange(0.4, 0.9, 0.01):
    preds = (probs >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()

    if (tn + fp) == 0 or (tp + fn) == 0:
        continue

    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    score = 0.7 * specificity + 0.3 * sensitivity

    if score > best_score:
        best_score = score
        best_thr = thr


print(f"\n🔥 Best Threshold: {best_thr:.2f}")
print(classification_report(y, (probs >= best_thr).astype(int)))


# ── SAVE ─────────────────────────────────────
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(float(best_thr), 'threshold.pkl')

print("\n✅ FINAL MODEL READY")