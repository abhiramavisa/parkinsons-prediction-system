from flask import Flask, request, jsonify, render_template
import joblib
import os
import numpy as np
import soundfile as sf
from feature_extractor import extract_features

app = Flask(__name__)

# ── Load model files ───────────────────────────────────────
model     = joblib.load('model.pkl')
scaler    = joblib.load('scaler.pkl')
threshold = joblib.load('threshold.pkl')

print(f"Model loaded. Decision threshold: {threshold:.2f}")

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

        print(f"Received: {file.filename}")

        # ── Save original file ─────────────────────────────
        ext = os.path.splitext(file.filename)[-1] or '.webm'
        original_path = os.path.join(UPLOAD_FOLDER, f'input_raw{ext}')
        file.save(original_path)

        if os.path.getsize(original_path) == 0:
            return jsonify({'error': 'Empty audio file'}), 400

        # ── Convert to WAV (NO LIBROSA) ────────────────────
        wav_path = os.path.join(UPLOAD_FOLDER, 'input.wav')

        try:
            data, samplerate = sf.read(original_path)

            # Convert stereo → mono
            if len(data.shape) > 1:
                data = data.mean(axis=1)

            if len(data) == 0:
                return jsonify({'error': 'Audio is empty'}), 400

            sf.write(wav_path, data, samplerate)

            print(f"Audio processed: {len(data)} samples @ {samplerate} Hz")

        except Exception as e:
            print(f"Audio decode error: {str(e)}")
            return jsonify({'error': f'Audio decoding failed: {str(e)}'}), 400

        # ── Feature extraction ─────────────────────────────
        try:
            features = extract_features(wav_path)
            features_scaled = scaler.transform(features)

        except Exception as e:
            print(f"Feature error: {str(e)}")
            return jsonify({'error': f'Feature extraction failed: {str(e)}'}), 500

        # ── Prediction ─────────────────────────────────────
        probability = model.predict_proba(features_scaled)[0][1]
        prediction  = 1 if probability >= threshold else 0

        label = "Parkinson's Detected" if prediction == 1 else "Healthy"
        confidence = probability if prediction == 1 else (1 - probability)

        print(f"Prob: {probability:.3f} | Result: {label}")

        # ── FINAL RESPONSE (JSON ONLY) ─────────────────────
        return jsonify({
            "prediction": int(prediction),
            "label": label,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


# ── Run ────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)