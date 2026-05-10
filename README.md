# 🍎 Fruit Quality Detector

A deep learning model that automatically classifies fruits as **fresh** or **rotten** using Convolutional Neural Networks (CNN) built with TensorFlow and Keras.

---

## 📌 Overview

This project uses image classification to detect the quality of 6 common fruits. Given an input image, the model predicts whether the fruit is fresh or rotten — making it useful for automated quality control in food supply chains, grocery stores, or smart refrigerators.

---

## 🍇 Supported Fruit Classes

The model classifies images into **12 categories** (fresh + rotten variants):

| Fruit        | Fresh Label         | Rotten Label         |
|--------------|---------------------|----------------------|
| Apple        | `freshapples`       | `rottenapples`       |
| Banana       | `freshbanana`       | `rottenbanana`       |
| Mango        | `freshmango`        | `rottenmango`        |
| Orange       | `freshoranges`      | `rottenoranges`      |
| Pomegranate  | `freshpomegranate`  | `rottenpomegranate`  |
| Watermelon   | `freshwatermelon`   | `rottenwatermelon`   |

---

## 🧠 Model Architecture

A custom CNN built with Keras `Sequential` API:

```
Input (224×224×3)
    │
    ├── Conv2D(32, 3×3, relu) → MaxPool2D(2×2)      # Block 1
    ├── Conv2D(64, 3×3, relu) → MaxPool2D(2×2)      # Block 2
    ├── Conv2D(128, 3×3, relu) → MaxPool2D(2×2)     # Block 3
    │
    ├── Flatten
    ├── Dense(256, relu)
    ├── Dropout(0.4)
    ├── Dense(128, relu)
    └── Dense(12, softmax)                           # Output
```

- **Optimizer:** Adam (lr = 0.001)
- **Loss:** Sparse Categorical Crossentropy
- **Metric:** Accuracy
- **Early Stopping:** Monitors `val_accuracy` with patience = 2

---

## 📁 Dataset

- **Total Images:** ~13,252
- **Train Split:** 10,600 images
- **Test Split:** 2,652 images
- **Image Size:** 224 × 224 pixels (RGB)
- **Normalization:** Pixel values scaled to [0, 1]

Dataset folder structure expected:
```
Fruits_New/
├── freshapples/
├── freshbanana/
├── freshmango/
├── freshoranges/
├── freshpomegranate/
├── freshwatermelon/
├── rottenapples/
├── rottenbanana/
├── rottenmango/
├── rottenoranges/
├── rottenpomegranate/
└── rottenwatermelon/
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow keras opencv-python matplotlib seaborn scikit-learn numpy
```

### Training the Model

1. Update the `path` variable in the notebook to point to your dataset directory.
2. Run all cells in `Fruit_Quality_Detector.ipynb`.
3. The trained model will be saved as:
   - `food_freshness_or_rotten_model.keras`
   - `food_freshness_or_rotten_model.h5`

### Running a Single Prediction

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

cate = ['freshapples', 'freshbanana', 'freshpomegranate', 'freshwatermelon',
        'freshmango', 'freshoranges', 'rottenapples', 'rottenmango',
        'rottenpomegranate', 'rottenbanana', 'rottenwatermelon', 'rottenoranges']

model = load_model('food_freshness_or_rotten_model.keras')

img = cv2.imread('your_fruit_image.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (224, 224)) / 255.0
img_input = np.expand_dims(img_resized, axis=0)

pred = model.predict(img_input)[0]
print(f"Prediction: {cate[pred.argmax()]} ({pred.max()*100:.1f}% confidence)")
```

---

## 📊 Evaluation

After training, the notebook generates:
- **Confusion Matrix** (heatmap with Seaborn)
- **Classification Report** (precision, recall, F1-score per class)
- **Training/Validation Accuracy & Loss curves**
- **Sample Predictions grid** (12 random test images with true vs. predicted labels)

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| TensorFlow / Keras | Model building & training |
| OpenCV | Image loading & preprocessing |
| NumPy | Array operations |
| Matplotlib | Visualization |
| Seaborn | Confusion matrix heatmap |
| Scikit-learn | Metrics & evaluation |

---

## 📂 Project Structure

```
├── Fruit_Quality_Detector.ipynb       # Main notebook
├── food_freshness_or_rotten_model.keras  # Saved model (Keras format)
├── food_freshness_or_rotten_model.h5    # Saved model (HDF5 format)
└── README.md
```

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
