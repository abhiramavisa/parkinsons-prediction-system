import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score
import joblib

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
print(f"Healthy (0): {(y==0).sum()} | Parkinson's (1): {(y==1).sum()}")

# Scale on full data so scaler covers entire distribution
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Gradient Boosting handles imbalance better than Random Forest
# and generalizes better to out-of-distribution real-world audio
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=42,
)

# Cross-validation
cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1')
print(f"\nCross-val F1: {scores.mean():.3f} ± {scores.std():.3f}")

# Fit on all data
model.fit(X_scaled, y)
y_pred = model.predict(X_scaled)
print(f"Training accuracy: {accuracy_score(y, y_pred):.2%}")
print(classification_report(y, y_pred, target_names=['Healthy', "Parkinson's"]))

# ── Find optimal threshold favoring fewer false positives ────
# We want to reduce false Parkinson's detections on healthy people
probs    = model.predict_proba(X_scaled)[:, 1]
best_thr = 0.5
best_score = 0.0

for thr in np.arange(0.25, 0.85, 0.01):
    preds = (probs >= thr).astype(int)
    # Penalize false positives (healthy predicted as Parkinson's) more
    tn = np.sum((preds == 0) & (y == 0))
    tp = np.sum((preds == 1) & (y == 1))
    fp = np.sum((preds == 1) & (y == 0))
    fn = np.sum((preds == 0) & (y == 1))
    # Weighted score: prioritize specificity (healthy correctly identified)
    if (tp + fn) > 0 and (tn + fp) > 0:
        sensitivity  = tp / (tp + fn)
        specificity  = tn / (tn + fp)
        score = 0.4 * sensitivity + 0.6 * specificity  # weight specificity higher
        if score > best_score:
            best_score = score
            best_thr   = thr

print(f"\nOptimal threshold: {best_thr:.2f} (score={best_score:.3f})")

# Show what this threshold does
preds = (probs >= best_thr).astype(int)
tn = np.sum((preds == 0) & (y == 0))
tp = np.sum((preds == 1) & (y == 1))
fp = np.sum((preds == 1) & (y == 0))
fn = np.sum((preds == 0) & (y == 1))
print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
print(f"Sensitivity: {tp/(tp+fn):.2%} | Specificity: {tn/(tn+fp):.2%}")

# Save
joblib.dump(model,           'model.pkl')
joblib.dump(scaler,          'scaler.pkl')
joblib.dump(FEATURES,        'features.pkl')
joblib.dump(float(best_thr), 'threshold.pkl')

print("\nSaved: model.pkl, scaler.pkl, features.pkl, threshold.pkl")

# ── Simulate your two audio files ───────────────────────────
print("\n=== Simulating your audio files ===")
test_cases = {
    "AH_545616 (Parkinson's patient)": [126.35, 137.03, 120.28, 0.006658, 0.0000441, 0.003779, 0.003779, 0.011337, 0.064818, 0.577, 0.034205, 0.038, 0.050, 0.102615, 0.045, 13.6451],
    "AH_064F   (Healthy patient)":     [132.01, 157.68, 128.10, 0.004138, 0.0000313, 0.002028, 0.002028, 0.006084, 0.038810, 0.330, 0.021972, 0.025, 0.032, 0.065916, 0.025, 20.3022],
}

for label, feats in test_cases.items():
    f_scaled = scaler.transform([feats])
    prob     = model.predict_proba(f_scaled)[0][1]
    pred     = 1 if prob >= best_thr else 0
    result   = "Parkinson's" if pred == 1 else "Healthy"
    print(f"{label}: prob={prob:.3f} threshold={best_thr:.2f} → {result}")