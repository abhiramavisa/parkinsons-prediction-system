from flask import Flask, request, jsonify, render_template
import joblib
import os
import numpy as np
import librosa
import soundfile as sf
from feature_extractor import extract_features

app = Flask(__name__)

model     = joblib.load('model.pkl')
scaler    = joblib.load('scaler.pkl')
threshold = joblib.load('threshold.pkl')

print(f"Model loaded. Decision threshold: {threshold:.2f}")

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file received'}), 400

    file = request.files['audio']
    print(f"Received: filename={file.filename}, mimetype={file.mimetype}")

    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Save with original extension
    ext           = os.path.splitext(file.filename)[-1] or '.webm'
    original_path = os.path.join(UPLOAD_FOLDER, f'input_raw{ext}')
    file.save(original_path)

    file_size = os.path.getsize(original_path)
    print(f"Saved file size: {file_size} bytes")

    if file_size == 0:
        return jsonify({'error': 'Received empty audio file'}), 400

    # Convert to proper WAV
    wav_path = os.path.join(UPLOAD_FOLDER, 'input.wav')
    try:
        y, sr = librosa.load(original_path, sr=None, mono=True)
        if len(y) == 0:
            return jsonify({'error': 'Audio appears silent or unreadable'}), 400
        sf.write(wav_path, y, sr)
        print(f"Converted to WAV: {len(y)} samples at {sr}Hz, duration={len(y)/sr:.2f}s")
    except Exception as e:
        return jsonify({'error': f'Could not decode audio: {str(e)}'}), 400

    # Extract features
    try:
        features        = extract_features(wav_path)
        features_scaled = scaler.transform(features)

        # Use optimal threshold instead of default 0.5
        probability  = model.predict_proba(features_scaled)[0]
        prob_parkinsons = float(probability[1])
        prediction   = 1 if prob_parkinsons >= threshold else 0

        print(f"Prob Parkinson's: {prob_parkinsons:.3f} | Threshold: {threshold:.2f} | Result: {prediction}")

        return jsonify({
            'prediction': prediction,
            'label':      "Parkinson's Detected" if prediction == 1 else "Healthy",
            'confidence': round(prob_parkinsons * 100 if prediction == 1 else (1 - prob_parkinsons) * 100, 1)
        })

    except Exception as e:
        print(f"Feature extraction error: {str(e)}")
        return jsonify({'error': f'Feature extraction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)