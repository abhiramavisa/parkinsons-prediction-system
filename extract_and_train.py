# extract_and_train.py
import os, numpy as np, pandas as pd
import parselmouth, librosa, soundfile as sf
from parselmouth.praat import call
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, f1_score
import joblib

# ── UPDATE THESE PATHS ───────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PD_FOLDER = os.path.join(BASE_DIR, 'PD_AH')
HC_FOLDER = os.path.join(BASE_DIR, 'HC_AH')
CSV_PATH  = os.path.join(BASE_DIR, 'parkinsons.data')
# ────────────────────────────────────────────────────────────

# ── Feature extraction from audio ────────────────────────────
def extract(audio_path):
    try:
        # Directly load into parselmouth (NO temp file needed)
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

    except Exception as e:
        print(f"Error: {e}")
        return None

# ── Load numerical dataset ───────────────────────────────────
print("\nLoading numerical dataset...")
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

print(f"Numerical data: {len(X_num)} samples")

# ── Extract audio features ───────────────────────────────────
rows, labels = [], []

for folder, label in [(PD_FOLDER, 1), (HC_FOLDER, 0)]:
    name = "Parkinson's" if label == 1 else "Healthy"
    files = [f for f in os.listdir(folder) if f.lower().endswith('.wav')]

    print(f"\nProcessing {name} audio ({len(files)} files)...")

    for fname in files:
        path = os.path.join(folder, fname)
        print(f"  {fname}", end=' ')

        feats = extract(path)

        if feats and not any(np.isnan(v) or np.isinf(v) for v in feats):
            rows.append(feats)
            labels.append(label)
            print("✓")
        else:
            print("✗ skipped")

X_audio = np.array(rows)
y_audio = np.array(labels)

print(f"\nAudio data: {len(X_audio)} samples")

# ── 🔥 COMBINE DATASETS ─────────────────────────────────────
X = np.vstack((X_num, X_audio))
y = np.hstack((y_num, y_audio))

print(f"\nCombined dataset: {len(X)} samples")
print(f"Healthy={(y==0).sum()} | Parkinson's={(y==1).sum()}")

# ── Train ────────────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=3,
    subsample=0.8,
    random_state=42
)

cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1')
print(f"\nCross-val F1: {scores.mean():.3f} ± {scores.std():.3f}")

model.fit(X_scaled, y)

# ── Optimal threshold ────────────────────────────────────────
probs = model.predict_proba(X_scaled)[:, 1]

best_thr, best_f1 = 0.5, 0.0
for thr in np.arange(0.2, 0.8, 0.01):
    preds = (probs >= thr).astype(int)
    f1    = f1_score(y, preds)
    if f1 > best_f1:
        best_f1, best_thr = f1, thr

print(f"\nOptimal threshold: {best_thr:.2f} (F1={best_f1:.3f})")
print(classification_report(y, model.predict(X_scaled),
      target_names=['Healthy', "Parkinson's"]))

# ── Save ─────────────────────────────────────────────────────
FEATURES = ['Fo','Fhi','Flo','Jitter','JitterAbs','RAP','PPQ','DDP',
            'Shimmer','ShimmerDB','APQ3','APQ5','APQ11','DDA','NHR','HNR']

joblib.dump(model,           'model.pkl')
joblib.dump(scaler,          'scaler.pkl')
joblib.dump(FEATURES,        'features.pkl')
joblib.dump(float(best_thr), 'threshold.pkl')

print("\n✅ DONE: Model trained on BOTH numerical + audio data")
print("Run: python app.py")