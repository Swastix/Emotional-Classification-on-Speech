# Speech Emotion Recognition using CNN-RNN with Audio Augmentation

## Project Description

This project implements a robust **Speech Emotion Recognition (SER)** system using a hybrid deep learning approach that combines Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs, specifically BiLSTM layers). The system processes raw `.wav` audio clips, applies data augmentation for better generalization, extracts rich acoustic features, and classifies the emotional content of speech into one of eight categories.

**Goal:**  
To accurately recognize emotions in speech audio (e.g., anger, joy, sadness, etc.) by leveraging advanced signal processing and deep learning.

---

## Data Pre-processing Methodology

### 1. Data Sources
- Audio files (`.wav`) are collected from multiple directories.
- Each file is labeled with an emotion using a mapping provided in `all_labels.csv`.

### 2. Audio Augmentation
To increase dataset diversity and robustness, each sample undergoes:
- **Original** (unaltered)
- **Noise Injection:** Adds Gaussian noise with a random factor.
- **Time Stretching:** Speeds up or slows down the clip slightly.
- **Pitch Shifting:** Raises or lowers the pitch randomly.

### 3. Feature Extraction
For each (augmented) audio sample:
- **Mel Spectrogram:** 128 bands, representing the audio in a perceptually-relevant frequency scale.
- **MFCCs:** 40 coefficients, capturing timbral and envelope content.
- **Chroma Features:** 12 bins, encoding harmonic content.
- **Zero-Crossing Rate:** Measures noisiness.
- **Root Mean Square Energy (RMSE):** Quantifies loudness.
- Features are concatenated into a unified matrix per file.

### 4. Padding & Formatting
- All feature matrices are padded or trimmed to a uniform frame length (`MAX_FRAMES`) for batch processing.

### 5. Label Encoding
- Emotion labels are one-hot encoded for multi-class classification.

### 6. Dataset Splitting
- **Stratified split:** 80% for training/validation, 20% for testing.
- Of train/val: 90% train, 10% val.
- Ensures balance among emotion classes.

---

## Model Pipeline

### 1. Input Layer
- Shape: `(TOTAL_FEATURES, MAX_FRAMES, 1)` representing (features × time × 1 channel).

### 2. CNN Front-End
- 3 stacked blocks, each with:
  - Two Conv2D layers with ReLU activation and batch normalization
  - Max Pooling and Dropout for regularization
- Purpose: Capture spatial (frequency-time) patterns in input features.

### 3. Reshape Layer
- Converts CNN output for RNN ingestion.

### 4. RNN Back-End
- Two Bidirectional LSTM layers:
  - First: 256 units, returns sequences
  - Second: 128 units, returns final state only
- Purpose: Model temporal dynamics and sequential dependencies in speech.

### 5. Classifier Head
- Dense layer with 256 units, BatchNorm, and Dropout
- Output layer: Softmax activation for 8 emotional classes

### 6. Compilation and Training
- **Loss:** Categorical cross-entropy
- **Optimizer:** Adam (LR: 5e-5)
- **Callbacks:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Batch size:** 64, up to 200 epochs (early-stopping enabled)

---

## Evaluation Metrics

After training, model performance is measured on the held-out test set using:

- **Confusion Matrix:** Visualizes correct vs. incorrect predictions per class
- **Classification Report:** Precision, Recall, and F1-score for each emotion
- **Overall Accuracy:** Percentage of total correct predictions
- **Macro F1 Score:** Harmonic mean of per-class F1-scores
- **Per-Class Accuracy:** Accuracy for each emotion class

### Sample Results
*## 📊 Evaluation Metrics

The final model was evaluated on a held-out test set. Here are the results:

| Emotion    | Precision | Recall | F1-score | Support |
|------------|:---------:|:------:|:--------:|:-------:|
| angry      |   0.96    |  0.86  |   0.90   |   287   |
| calm       |   0.84    |  1.00  |   0.91   |   279   |
| disgust    |   0.90    |  0.93  |   0.91   |   139   |
| fearful    |   0.84    |  0.90  |   0.87   |   240   |
| happy      |   0.94    |  0.84  |   0.89   |   254   |
| neutral    |   0.94    |  0.90  |   0.92   |   115   |
| sad        |   0.90    |  0.84  |   0.86   |   267   |
| surprised  |   0.90    |  0.97  |   0.93   |   79    |

**Overall accuracy:** 0.89 (1660 samples)  
**Macro avg:**    0.90 (precision), 0.90 (recall), 0.90 (f1-score)  
**Weighted avg:** 0.90 (precision), 0.89 (recall), 0.89 (f1-score)
*


---

## How to Run

1. **Install dependencies**
2. 
2. **Place audio files and `all_labels.csv` in the specified locations.**

3. **Run the notebook or Python script.**

---

## Model Saving

- The best model weights are saved as `best_model2_weights.keras`
- The final model is saved as `best_model_improved.h5`

---

## Acknowledgements

- Audio processing: [Librosa](https://librosa.org/)
- Deep learning: [Tensorflow/Keras](https://www.tensorflow.org/)
- Inspired by RAVDESS and similar emotion datasets.

---

## Contact

For questions or collaboration, please reach out to the project maintainer.

---

**Happy Learning!**

