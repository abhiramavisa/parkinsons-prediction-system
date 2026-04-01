from flask import Flask, request, jsonify, render_template
import joblib
import os
import numpy as np
from pydub import AudioSegment
from feature_extractor import extract_features
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ── Load model files ───────────────────────────────────────
model  = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# 🔥 FORCE BETTER THRESHOLD (fix false positives)
threshold = 0.82

print(f"✅ Model loaded. Using threshold: {threshold}")

# ── Upload folder ──────────────────────────────────────────
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Routes ─────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file received'}), 400

        file = request.files['audio']

        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        filename = secure_filename(file.filename)
        print(f"\n📁 Received file: {filename}")

        # ── Save original file ─────────────────────────────
        ext = os.path.splitext(filename)[-1] or '.webm'
        original_path = os.path.join(UPLOAD_FOLDER, f'input_raw{ext}')
        file.save(original_path)

        if not os.path.exists(original_path) or os.path.getsize(original_path) == 0:
            return jsonify({'error': 'File saving failed or empty file'}), 400

        print(f"✅ File saved: {original_path}")

        # ── Convert to WAV ─────────────────────────────────
        wav_path = os.path.join(UPLOAD_FOLDER, 'input.wav')

        try:
            audio = AudioSegment.from_file(original_path)

            # Standardize audio
            audio = audio.set_channels(1).set_frame_rate(22050)

            audio.export(wav_path, format="wav")

            print(f"🎧 Converted to WAV")

        except Exception as e:
            print(f"❌ Audio conversion error: {str(e)}")
            return jsonify({'error': f'Audio decoding failed: {str(e)}'}), 400

        # ── Feature extraction ─────────────────────────────
        try:
            features = extract_features(wav_path)

            features = np.array(features).reshape(1, -1)

            print(f"📊 Features: {features}")

            # Validate feature size
            if features.shape[1] != scaler.n_features_in_:
                return jsonify({
                    'error': f'Feature mismatch: expected {scaler.n_features_in_}, got {features.shape[1]}'
                }), 500

            features_scaled = scaler.transform(features)

        except Exception as e:
            print(f"❌ Feature error: {str(e)}")
            return jsonify({'error': f'Feature extraction failed: {str(e)}'}), 500

        # ── Prediction ─────────────────────────────────────
        # ── Prediction ─────────────────────────────────────
        # ── Prediction ─────────────────────────────────────
        try:
            probability = model.predict_proba(features_scaled)[0][1]

            print(f"🧠 Raw Probability: {probability:.4f}")

    # 🔥 EXTRA SAFETY: check HNR (important feature)
            hnr = features[0][-1]   # last feature = HNR
            jitter = features[0][3]

            print(f"HNR: {hnr}, Jitter: {jitter}")

    # 🚀 FINAL DECISION
            if probability >= 0.85:
        # 🔥 override if voice is actually stable
                if hnr > 15 and jitter < 0.005:
                    label = "Healthy (Override)"
                    prediction = 0
                    confidence = 0.7
                else:
                    label = "Parkinson's Detected"
                    prediction = 1
                    confidence = probability

            elif probability <= 0.45:
                label = "Healthy"
                prediction = 0
                confidence = 1 - probability

            else:
                label = "Uncertain (Better audio required)"
                prediction = -1
                confidence = 0.0

            print(f"✅ Final Result: {label}")

        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")

        # ── Response ───────────────────────────────────────
        return jsonify({
            "prediction": int(prediction),
            "label": label,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        print(f"🔥 Unexpected error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ── Run ────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)