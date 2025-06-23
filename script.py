# predict.py
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Load model and encoder classes
model = load_model('best_model_improved.h5')
encoder_classes = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Constants
N_MELS = 128
N_MFCC = 40
MAX_FRAMES = 175
TOTAL_FEATURES = N_MELS + N_MFCC + 12 + 1 + 1

def extract_features(data, sr, n_mels=N_MELS, n_mfcc=N_MFCC, fmax=8000):
    stft = np.abs(librosa.stft(data, n_fft=2048, hop_length=512))
    mel_spec = librosa.feature.melspectrogram(S=stft**2, sr=sr, n_mels=n_mels, fmax=fmax)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=data)
    rmse = librosa.feature.rms(y=data)

    min_frames = min(log_mel_spec.shape[1], mfccs.shape[1], chroma.shape[1], zcr.shape[1], rmse.shape[1])
    combined = np.vstack([
        log_mel_spec[:, :min_frames],
        mfccs[:, :min_frames],
        chroma[:, :min_frames],
        zcr[:, :min_frames],
        rmse[:, :min_frames]
    ])
    return combined

def pad_features(f, max_frames=MAX_FRAMES):
    if f.shape[1] < max_frames:
        pad_width = max_frames - f.shape[1]
        return np.pad(f, ((0, 0), (0, pad_width)), mode='constant')
    else:
        return f[:, :max_frames]

def preprocess(file_path):
    data, sr = librosa.load(file_path, duration=3, offset=0.5, sr=22050)
    features = extract_features(data, sr)
    padded = pad_features(features)
    return np.expand_dims(padded[..., np.newaxis], axis=0)

def predict_emotion(file_path):
    features = preprocess(file_path)
    probs = model.predict(features)[0]
    predicted_label = encoder_classes[np.argmax(probs)]
    prob_dict = {label: float(f"{p:.4f}") for label, p in zip(encoder_classes, probs)}
    return predicted_label, prob_dict

# --- User Prompt ---
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = 'audio test\excited.wav'  # fallback or default

    file_path = path.strip()
    print(f"\n🔍 Predicting emotion for: {file_path}")
    predicted_label, probabilities = predict_emotion(file_path)

    print(f"\n🎯 Predicted Emotion: {predicted_label}")
    print("\n📊 Class Probabilities:")
    for cls, prob in probabilities.items():
        print(f"  {cls}: {prob:.4f}")
