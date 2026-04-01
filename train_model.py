import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import joblib

# ── LOAD DATA ─────────────────────────────────
df = pd.read_csv('parkinsons.data')

FEATURES = [
    'MDVP:Fo(Hz)',
    'MDVP:Fhi(Hz)',
    'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)',
    'MDVP:Jitter(Abs)',
    'MDVP:RAP',
    'MDVP:PPQ',
    'Jitter:DDP',
    'MDVP:Shimmer',
    'MDVP:Shimmer(dB)',
    'Shimmer:APQ3',
    'Shimmer:APQ5',
    'MDVP:APQ',
    'Shimmer:DDA',
    'NHR',
    'HNR',
]

X = df[FEATURES].values
y = df['status'].values

print(f"Dataset: {len(df)} samples")
print(f"Healthy: {(y==0).sum()} | Parkinson's: {(y==1).sum()}")

# 🔥 BALANCE DATASET (CRITICAL FIX)
healthy = X[y == 0]
parkinsons = X[y == 1]

healthy_up = resample(
    healthy,
    replace=True,
    n_samples=len(parkinsons),
    random_state=42
)

X = np.vstack((healthy_up, parkinsons))
y = np.array([0]*len(healthy_up) + [1]*len(parkinsons))

print(f"\nBalanced Dataset:")
print(f"Healthy: {len(healthy_up)} | Parkinson's: {len(parkinsons)}")

# ── SCALE ─────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── MODEL ─────────────────────────────────────
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

    specificity = tn / (tn + fp)   # healthy correct
    sensitivity = tp / (tp + fn)   # parkinson correct

    score = 0.7 * specificity + 0.3 * sensitivity

    if score > best_score:
        best_score = score
        best_thr = thr

print(f"\n🔥 Best Threshold: {best_thr:.2f}")

# ── FINAL EVALUATION ──────────────────────────
final_preds = (probs >= best_thr).astype(int)
print("\nClassification Report:")
print(classification_report(y, final_preds, target_names=['Healthy', "Parkinson's"]))

# ── SAVE ──────────────────────────────────────
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(float(best_thr), 'threshold.pkl')

print("\n✅ MODEL TRAINED & SAVED SUCCESSFULLY")