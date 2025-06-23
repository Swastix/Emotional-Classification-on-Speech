import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Constants
N_MELS = 128
N_MFCC = 40
MAX_FRAMES = 175
TOTAL_FEATURES = N_MELS + N_MFCC + 12 + 1 + 1
EMOTION_CLASSES = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Load trained model
@st.cache_resource
def load_trained_model():
    return load_model("best_model_improved.h5")

model = load_trained_model()

# Feature extraction function
def extract_features(data, sr):
    stft = np.abs(librosa.stft(data, n_fft=2048, hop_length=512))
    mel_spec = librosa.feature.melspectrogram(S=stft**2, sr=sr, n_mels=N_MELS, fmax=8000)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=N_MFCC)
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

def pad_features(f):
    if f.shape[1] < MAX_FRAMES:
        pad_width = MAX_FRAMES - f.shape[1]
        return np.pad(f, ((0, 0), (0, pad_width)), mode='constant')
    else:
        return f[:, :MAX_FRAMES]

def preprocess(file):
    data, sr = librosa.load(file, duration=3, offset=0.5, sr=22050)
    features = extract_features(data, sr)
    padded = pad_features(features)
    return np.expand_dims(padded[..., np.newaxis], axis=0)

def predict_emotion(file):
    input_features = preprocess(file)
    probabilities = model.predict(input_features)[0]
    prediction = EMOTION_CLASSES[np.argmax(probabilities)]
    return prediction, probabilities

# --- Streamlit UI ---
st.set_page_config(page_title="Speech Emotion Classifier 🎙️", layout="centered")
st.title("🎧 Speech Emotion Recognition")
st.markdown("Upload a `.wav` file to detect the **emotion** using your trained CNN-RNN model.")

uploaded_file = st.file_uploader("Upload Audio File (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    with st.spinner("Analyzing..."):
        predicted_emotion, probs = predict_emotion(uploaded_file)
        
    st.success(f"🎯 **Predicted Emotion**: `{predicted_emotion}`")
    
    st.subheader("📊 Class Probabilities")
    prob_df = {
        "Emotion": EMOTION_CLASSES,
        "Probability": [round(float(p), 4) for p in probs]
    }
    st.bar_chart(prob_df, x="Emotion", y="Probability")

    # Optional: show raw probability values
    with st.expander("Show raw probability values"):
        for emotion, prob in zip(EMOTION_CLASSES, probs):
            st.write(f"**{emotion}**: {prob:.4f}")
