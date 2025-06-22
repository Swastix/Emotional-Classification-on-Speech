
The model is trained on the **RAVDESS** dataset (Ryerson Audio-Visual Database of Emotional Speech and Song).

---

## 🎤 Dataset

- **RAVDESS**: 24 professional actors (12 male, 12 female), vocalizing two lexically-matched statements in a neutral North American accent.
- Emotions: calm, happy, sad, angry, fearful, surprise, disgust, and neutral.
- Audio files: `.wav` format, standardized sample rate `22050 Hz`.

We also included **data augmentation** for robustness.

---

## 🧪 Preprocessing

1. **Loading**: Audio clips are loaded using `librosa.load()` (3s clips, offset 0.5s).
2. **Augmentation**:
   - Noise Injection
   - Time Stretching
   - Pitch Shifting
3. **Label Encoding**: One-hot encoding for emotion labels.

---

## 🎛 Feature Extraction

For each audio clip, the following features are extracted and concatenated:
- **Mel-Spectrogram** (128 bins)
- **MFCCs** (40 coefficients)
- **Chroma STFT** (12 features)
- **Zero Crossing Rate** (1)
- **Root Mean Square Energy (RMSE)** (1)

Each feature matrix is padded/truncated to `MAX_FRAMES = 175`, and reshaped as:
Where `TOTAL_FEATURES = 128 + 40 + 12 + 1 + 1 = 182`

---

## 🧠 Model Architecture (CNN + RNN)

```text
Input Shape: (182, 175, 1)

Conv2D → BN → Conv2D → BN → MaxPool → Dropout
Conv2D → BN → Conv2D → BN → MaxPool → Dropout
Conv2D → BN → Conv2D → BN → MaxPool → Dropout

Reshape to (time_steps, features)
→ Bidirectional LSTM (256, return_sequences)
→ Dropout → Bidirectional LSTM (128)
→ BN → Dropout → Dense → BN → Dropout → Softmax
