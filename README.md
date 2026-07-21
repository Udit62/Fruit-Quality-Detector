# 🍎 Fruit Quality Detector

🔗 **Live Demo:** [http://107.22.221.241](http://107.22.221.241)

Note: hosted on AWS EC2 free tier — first request may take a few seconds while the model loads.

A deep learning model that automatically classifies fruits as **fresh** or **rotten** using Convolutional Neural Networks (CNN) built with TensorFlow and Keras.

---

## 📌 Overview

This project uses image classification to detect the quality of 6 common fruits. Given an input image, the model predicts whether the fruit is fresh or rotten — making it useful for automated quality control in food supply chains, grocery stores, or smart refrigerators.

---

## 🍇 Supported Fruit Classes

The model classifies images into **13 categories** (6 fruits × fresh/rotten, plus an Other/Unknown class):

| Fruit        | Fresh Label         | Rotten Label         |
|--------------|---------------------|----------------------|
| Apple        | `freshapples`       | `rottenapples`       |
| Banana       | `freshbanana`       | `rottenbanana`       |
| Mango        | `freshmango`        | `rottenmango`        |
| Orange       | `freshoranges`      | `rottenoranges`      |
| Pomegranate  | `freshpomegranate`  | `rottenpomegranate`  |
| Watermelon   | `freshwatermelon`   | `rottenwatermelon`   |

Plus a 13th class, **`Other_Unknown`**, trained on non-fruit and unsupported-fruit images so the model can explicitly reject inputs outside its scope instead of forcing a wrong prediction (see [Known Limitations](#-known-limitations--resolved) below).

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
    └── Dense(13, softmax)                           # Output (12 fruit classes + Other/Unknown)
```

- **Optimizer:** Adam (lr = 0.001)
- **Loss:** Sparse Categorical Crossentropy
- **Metric:** Accuracy
- **Early Stopping:** Monitors `val_accuracy` with patience = 2

---

## 📁 Dataset

- **Total Images:** ~14,329 (across 13 classes)
- **Train Split:** 11,463 images
- **Test Split:** 2,866 images
- **Image Size:** 224 × 224 pixels (RGB)
- **Normalization:** Pixel values scaled to [0, 1]
- **Other_Unknown class:** ~1,078 images sourced from non-fruit scenes (landscapes, buildings, objects) and unsupported fruit types, used to teach the model to reject out-of-distribution inputs

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
├── rottenwatermelon/
└── Other_Unknown/
```

---

## 💰 Cost & Deployment Efficiency

One goal of this project was to keep it deployable at **zero recurring cost** without sacrificing usability.

- **Model compression:** The saved model initially included the optimizer state (needed only for resuming training, not for inference). Stripping the optimizer state reduced the model file size from **295 MB to ~98.5 MB**, which brought it under the free-tier deployment limits.
- **Hosting cost:** Currently deployed on **AWS EC2 free tier at $0/month**. No GPU or paid inference endpoint is used.
- **Cost per identification:** At current traffic, cost per prediction is effectively **$0**. For reference, if this were scaled onto a pay-per-invocation service like AWS Lambda, cost would scale at roughly **$0.012 per 1,000 predictions** (based on Lambda's per-request + compute-time pricing, assuming 2 GB memory and ~0.36s average inference time), making the current architecture significantly cheaper for low-to-moderate traffic use cases.
- **Trade-off:** This is a size/cost optimization, not an accuracy optimization — the underlying trained weights and prediction accuracy are unaffected by removing optimizer state.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```
Key dependencies: `tensorflow-cpu`, `flask`, `opencv-python-headless`, `numpy`, `h5py`, `gunicorn`

### Training the Model

1. Update the `path` variable in the notebook to point to your dataset directory (including the `Other_Unknown` folder).
2. Run all cells in `Fruit_Quality_Detector.ipynb`.
3. The trained model will be saved as `food_freshness_deploy.h5`.

### Running a Single Prediction

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

cate = ['freshapples', 'freshbanana', 'freshpomegranate', 'freshwatermelon',
        'freshmango', 'freshoranges', 'rottenapples', 'rottenmango',
        'rottenpomegranate', 'rottenbanana', 'rottenwatermelon', 'rottenoranges',
        'Other_Unknown']

model = load_model('food_freshness_deploy.h5')

img = cv2.imread('your_fruit_image.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (224, 224)) / 255.0
img_input = np.expand_dims(img_resized, axis=0)

pred = model.predict(img_input)[0]
print(f"Prediction: {cate[pred.argmax()]} ({pred.max()*100:.1f}% confidence)")
```

### Running the Web App Locally

```bash
python app.py
```
Flask serves the app at `http://localhost:5000` (or the configured port). The app applies an 85% confidence threshold on top of the model's own `Other_Unknown` class as a second safety net against low-confidence predictions.

---

## 📊 Evaluation

After training, the notebook generates:
- **Confusion Matrix** (heatmap with Seaborn)
- **Classification Report** (precision, recall, F1-score per class)
- **Training/Validation Accuracy & Loss curves**
- **Sample Predictions grid** (12 random test images with true vs. predicted labels)

---

## ✅ Known Limitations & Resolved

**Resolved: Open-set classification problem**

The original 12-class model used a plain softmax output, meaning it was mathematically forced to assign a prediction across only its known fruit classes — even when shown a non-fruit image or an unsupported fruit. This occasionally caused high-confidence wrong predictions on out-of-distribution inputs (e.g., classifying a photo of a person or landscape as a fruit).

**Fix implemented:**
- Retrained the model with an explicit **13th class (`Other_Unknown`)**, using ~1,078 negative examples spanning non-fruit scenes (landscapes, buildings, objects) and unsupported fruit types
- Added an **85% confidence threshold** in the Flask app as a second safety net — any prediction (even a fruit class) below this threshold is rejected as "not confident enough," rather than shown as a definite result
- Added user-facing messaging that names the 6 supported fruits, both as a persistent hint on the upload page and in the rejection message, so users understand what the app does and doesn't support

**Results after retraining** (13-class model, evaluated on held-out test set):
- Overall accuracy: **95%** (unchanged from the original 12-class model — the new class didn't hurt existing fruit classification)
- `Other_Unknown` class performance: **Precision: 0.97, Recall: 0.95, F1-score: 0.96** (on 204 held-out test images)

**Remaining minor limitation:** Some visually similar round/yellow fruits can occasionally be misclassified due to the model leaning on color as a shortcut feature rather than shape. Partial mitigation is in place via hard-negative examples and the confidence threshold; further fix (shape-focused data augmentation) is a possible future improvement.

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
| Flask | Web app framework |
| Gunicorn | WSGI server for production deployment |
| AWS EC2 | Hosting (free tier) |
| systemd | Process management & auto-restart on the server |

---

## 📂 Project Structure

```
├── Fruit_Quality_Detector.ipynb   # Training notebook (13-class model)
├── food_freshness_deploy.h5       # Deployed model (optimizer-stripped, ~98.5 MB)
├── app.py                         # Flask application (inference + serving)
├── requirements.txt                # Python dependencies
├── templates/
│   ├── index.html                 # Upload page
│   └── result.html                # Prediction results page
├── static/
│   └── uploads/                   # User-uploaded images (runtime)
├── Fruits_New/                    # Training dataset (13 class folders)
└── README.md
```

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
