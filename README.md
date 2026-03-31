# Medical Image Contrast Detection using LeNet-Inspired CNN

A Convolutional Neural Network that classifies whether a medical image was taken **with or without contrast enhancement**, built on a LeNet-5-inspired architecture using TensorFlow/Keras.

---

## Overview

Contrast-enhanced medical imaging (e.g., CT scans with contrast agents) requires detection for downstream processing pipelines. This project frames it as a binary classification task: **contrast (1) vs. no contrast (0)**.

**Pipeline:**
1. Load and preprocess image data from `.npz` and `.csv` files
2. Build a LeNet-style CNN adapted for 256×256 grayscale inputs
3. Train with Adam optimizer and early stopping
4. Evaluate with classification report and confusion matrix
5. Run single-image inference

---

## Model Architecture

| Block | Layer | Output Shape |
|-------|-------|--------------|
| Input | — | 256×256×1 |
| Conv1 | Conv2D(20, 5×5) + ReLU + MaxPool(2×2) | 128×128×20 |
| Conv2 | Conv2D(50, 5×5) + ReLU + MaxPool(2×2) | 64×64×50 |
| FC1 | Flatten → Dense(100) + ReLU + Dropout(0.5) | 100 |
| Output | Dense(2) + Softmax | 2 |

---

## Requirements

```
tensorflow >= 2.x
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

---

## Data Format

Place the following files in `../input/`:

| File | Description |
|------|-------------|
| `full_archive.npz` | NumPy archive with keys `idx` (image IDs) and `image` (raw arrays) |
| `overview.csv` | Metadata CSV with columns `idx` and `Contrast` (boolean/int) |

Images are automatically:
- Min-max normalised to [0, 255]
- Downsampled 2× (every other row/column) to reach 256×256

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image size | 256×256 |
| Conv1 filters | 20 |
| Conv2 filters | 50 |
| Kernel size | 5×5 |
| FC units | 100 |
| Dropout rate | 0.5 |
| Batch size | 8 |
| Max epochs | 10 |
| Learning rate | 1e-3 (Adam) |
| Train samples | 400 |
| Test samples | 75 |

---

## Usage

### Training

Open and run `contrast_detection_lenet_v2.ipynb` end-to-end. The notebook will:

- Load data from `../input/`
- Train the model with early stopping (patience=3) and LR reduction on plateau
- Plot training/validation loss and accuracy curves
- Print a full classification report and confusion matrix

### Inference on a Single Image

```python
from tensorflow import keras
import numpy as np

model = keras.models.load_model('contrast_detector_lenet.keras')

# image_array: np.ndarray of shape (256, 256), pixel values in [0, 1]
result = predict_single(image_array, model)
print(result)
# {'class': 'Contrast', 'confidence': 0.97, 'probabilities': {'No Contrast': 0.03, 'Contrast': 0.97}}
```

### Saving / Loading

The trained model is saved automatically:

```python
# Save
model.save('contrast_detector_lenet.keras')

# Load
model = keras.models.load_model('contrast_detector_lenet.keras')
```

---

## Project Structure

```
├── contrast_detection_lenet_v2.ipynb   # Main notebook
├── contrast_detector_lenet.keras       # Saved model (generated after training)
└── ../input/
    ├── full_archive.npz                # Image data
    └── overview.csv                    # Labels and metadata
```

---

## Reproducibility

A fixed random seed (`SEED = 42`) is set for both NumPy and TensorFlow to ensure reproducible results across runs.
