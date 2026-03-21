from flask import Flask, request, jsonify, render_template
import joblib
import os
import numpy as np
import soundfile as sf
from feature_extractor import extract_features
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ── Load model files ───────────────────────────────────────
model     = joblib.load('model.pkl')
scaler    = joblib.load('scaler.pkl')
threshold = joblib.load('threshold.pkl')

print(f"✅ Model loaded. Threshold: {threshold}")

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
        # ── Check file ──────────────────────────────────────
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file received'}), 400

        file = request.files['audio']

        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        filename = secure_filename(file.filename)
        print(f"📁 Received file: {filename}")

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
            data, samplerate = sf.read(original_path)

            if len(data) == 0:
                return jsonify({'error': 'Audio is empty'}), 400

            # Stereo → mono
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            sf.write(wav_path, data, samplerate)

            print(f"🎧 Audio processed: {len(data)} samples @ {samplerate} Hz")

        except Exception as e:
            print(f"❌ Audio decode error: {str(e)}")
            return jsonify({'error': f'Audio decoding failed: {str(e)}'}), 400

        # ── Feature extraction ─────────────────────────────
        try:
            features = extract_features(wav_path)

            print(f"📊 Raw features shape: {np.array(features).shape}")

            # Ensure 2D shape
            features = np.array(features).reshape(1, -1)

            # Check feature size matches scaler
            expected_features = scaler.n_features_in_
            if features.shape[1] != expected_features:
                return jsonify({
                    'error': f'Feature mismatch: expected {expected_features}, got {features.shape[1]}'
                }), 500

            features_scaled = scaler.transform(features)

        except Exception as e:
            print(f"❌ Feature error: {str(e)}")
            return jsonify({'error': f'Feature extraction failed: {str(e)}'}), 500

        # ── Prediction ─────────────────────────────────────
        try:
            probability = model.predict_proba(features_scaled)[0][1]
            prediction  = 1 if probability >= threshold else 0

            label = "Parkinson's Detected" if prediction == 1 else "Healthy"
            confidence = probability if prediction == 1 else (1 - probability)

            print(f"🧠 Prob: {probability:.4f} | Result: {label}")

        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            return jsonify({'error': f'Model prediction failed: {str(e)}'}), 500

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