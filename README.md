🍎 **Fruit Quality Detector**

**🔗 **Live Demo:**** http://107.22.221.241

Note: hosted on AWS EC2 free tier — first request may take a few seconds while the model loads.

A deep learning model that automatically classifies fruits as fresh or rotten using Convolutional Neural Networks (CNN) built with TensorFlow and Keras.


****📌 Overview
****
This project uses image classification to detect the quality of 6 common fruits. Given an input image, the model predicts whether the fruit is fresh or rotten — making it useful for automated quality control in food supply chains, grocery stores, or smart refrigerators.


**🍇 Supported Fruit Classes
**
The model classifies images into 12 categories (fresh + rotten variants):

FruitFresh LabelRotten LabelApplefreshapplesrottenapplesBananafreshbananarottenbananaMangofreshmangorottenmangoOrangefreshorangesrottenorangesPomegranatefreshpomegranaterottenpomegranateWatermelonfreshwatermelonrottenwatermelon


🧠 Model Architecture

A custom CNN built with Keras Sequential API:

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


Optimizer: Adam (lr = 0.001)
Loss: Sparse Categorical Crossentropy
Metric: Accuracy
Early Stopping: Monitors val_accuracy with patience = 2



📁 Dataset


Total Images: ~13,252
Train Split: 10,600 images
Test Split: 2,652 images
Image Size: 224 × 224 pixels (RGB)
Normalization: Pixel values scaled to [0, 1]


Dataset folder structure expected:

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


💰 Cost & Deployment Efficiency

One goal of this project was to keep it deployable at zero recurring cost without sacrificing usability.


Model compression: The saved model initially included the optimizer state (needed only for resuming training, not for inference). Stripping the optimizer state reduced the model file size from  300 MB to 97 MB, which brought it under the free-tier deployment limits.
Hosting cost: Currently deployed on AWS EC2 free tier at $0/month. No GPU or paid inference endpoint is used.
Cost per identification: At current traffic, cost per prediction is effectively $0. For reference, if this were scaled onto a pay-per-invocation service like AWS Lambda, cost would scale at roughly $0.012 per 1,000 predictions (based on Lambda's per-request + compute-time pricing), making the current architecture significantly cheaper for low-to-moderate traffic use cases.
Trade-off: This is a size/cost optimization, not an accuracy optimization — the underlying trained weights and prediction accuracy are unaffected by removing optimizer state.
