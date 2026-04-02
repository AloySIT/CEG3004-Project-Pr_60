# CEG3004 DSP Mini-Project — Environmental Sound Classification
**Group ID:** Pr_60

## Overview

This project implements a robust audio classification pipeline for Environmental Sound Classification (ESC-50). The pipeline classifies audio clips into 50 sound categories and is designed to perform well under clean, noisy, and band-limited conditions.

---

## Project Structure

```
CEG3004_Project/
├── CEG3004_Project_Colab_Improved.ipynb   # Main notebook (run this)
├── Pr_60_model.joblib                     # Trained SVM model (auto-generated)
├── Pr_60_predictions.csv                  # Submission predictions (auto-generated)
└── README.md                              # This file
```

---

## How to Run

### Step 1 — Open the notebook
Upload `CEG3004_Project_Colab_Improved.ipynb` to [Google Colab](https://colab.research.google.com) via **File → Upload notebook**.

### Step 2 — Set Group ID
In Cell 1, ensure the Group ID is set correctly:
```python
GROUP_ID = "Pr_60"
```

### Step 3 — Run all cells
Click **Runtime → Run all** and wait for everything to complete (approximately 15–20 minutes).

### Step 4 — Collect output files
Two files will automatically download to your computer:
- `Pr_60_model.joblib` — the trained model
- `Pr_60_predictions.csv` — predictions on the submission set

---

## Dataset

- Source: [ESC-50](https://github.com/karolpiczak/ESC-50) collection
- 2,000 audio clips across 50 sound classes (40 clips per class)
- Each clip is 5 seconds, mono, 16kHz
- Submission set contains 3 versions of each clip: **clean**, **noisy**, **band-limited**

---

## Pipeline

### 1. Preprocessing
Applied to every audio clip before feature extraction:

| Step | Description |
|------|-------------|
| Pre-emphasis filter | `y[n] = x[n] - 0.97 * x[n-1]` — boosts high-frequency content |
| Silence trimming | Removes leading/trailing silence using `top_db=25` |
| Fixed-length | Pads or truncates every clip to exactly 5 seconds |
| Peak normalisation | Scales waveform to `[-1, 1]` to remove gain differences |

### 2. Feature Extraction
Five DSP feature blocks are concatenated into a single feature vector (~500+ dimensions):

| Block | Description | Why it helps |
|-------|-------------|--------------|
| MFCC (40 coefficients) | Cepstral coefficients with CMVN + delta + delta-delta | Core timbral identity of each sound |
| Log-mel spectrogram (64 bands) | Log-scaled mel filterbank energies | Complementary texture information |
| Spectral descriptors | Centroid, bandwidth, rolloff, flux | Captures frequency shape — rolloff drops in band-limited clips |
| Zero-crossing rate | Rate of sign changes in the waveform | Stable under noise; measures noisiness |
| Chroma | Pitch-class distribution | Helps separate harmonic sounds (bells, instruments) |

All blocks use **median, 10th and 90th percentile pooling** in addition to mean and std — median is robust to impulsive noise spikes in individual frames.

**CMVN (Cepstral Mean and Variance Normalisation)** is applied to MFCCs before delta computation, making features robust to channel and recording-condition differences.

### 3. Training-time Augmentation
Each training clip is processed twice — once clean and once augmented — doubling the effective training set to 4,000 samples. Three independent augmentations:

| Augmentation | Parameters | Purpose |
|-------------|------------|---------|
| Additive Gaussian noise | SNR randomly 15–30 dB | Robustness to noisy submission clips |
| Random bandpass filter | 100–800 Hz low, 3000–7000 Hz high | Robustness to band-limited submission clips |
| Random gain scaling | Factor uniform in [0.6, 1.4] | Robustness to volume differences |

### 4. Model
SVM with RBF kernel via scikit-learn `Pipeline`:

```python
Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf', C=10, gamma='scale',
                class_weight='balanced', probability=True, random_state=42))
])
```

| Parameter | Value | Reason |
|-----------|-------|--------|
| `kernel` | `rbf` | Handles non-linear boundaries between 50 classes |
| `C` | `10` | Strong regularisation without overfitting |
| `gamma` | `scale` | Auto-scaled based on number of features |
| `class_weight` | `balanced` | Compensates for any class imbalance |

---

## Results

| Metric | Score |
|--------|-------|
| Validation Macro-F1 | **0.80** |

Performance score is weighted as: **50% clean + 25% noisy + 25% band-limited**.

---

## Dependencies

All dependencies are installed automatically in the notebook:

```
numpy
scipy
pandas
scikit-learn
librosa
soundfile
tqdm
gdown
```

---

## Submission Files

| File | Description |
|------|-------------|
| `Pr_60_model.joblib` | Trained SVM pipeline (scaler + classifier) |
| `Pr_60_predictions.csv` | Predicted labels for all submission clips |

The CSV has two columns: `clip_id` and `predicted_label`.
