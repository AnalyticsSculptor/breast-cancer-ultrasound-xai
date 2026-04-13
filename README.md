# breast-cancer-ultrasound-xai
End-to-end clinical image classification pipeline featuring uncertainty quantification (MC Dropout) and explainable AI (Grad-CAM)
# 🩺 Breast Cancer Ultrasound Classification with XAI

[![Live Demo](https://xazette-breast-cancer-ultrasound-xai.hf.space)](#) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)

An end-to-end, production-ready machine learning pipeline for classifying breast ultrasound images (Benign vs. Malignant). This project goes beyond basic accuracy by implementing **Uncertainty Quantification** and **Explainable AI (XAI)**, bridging the gap between raw deep learning and clinical trust.

### 🚀 Live Demo
You can try the live web application here: **[https://xazette-breast-cancer-ultrasound-xai.hf.space]**

## 🧠 Key Features
* **Medical-Grade Data Pipeline:** Handles severe class imbalances using Weighted Random Samplers and applies clinical augmentations (CLAHE, Elastic Transforms) via Albumentations.
* **EfficientNet-B4 Backbone:** Fine-tuned using Mixed Precision (AMP) and a custom Focal Loss implementation to heavily penalize false negatives.
* **Uncertainty Quantification (MC Dropout):** Runs 30 stochastic forward passes at inference to output a mathematical confidence range (± Standard Deviation) alongside the prediction.
* **Explainable AI (Grad-CAM):** Generates real-time heatmap overlays to visualize exactly which tissue structures the model analyzed to make its decision.
* **Strict Clinical Evaluation:** Evaluated strictly on Sensitivity (Recall) to ensure actual malignant cases are not missed, achieving a **1.0 (100%) Sensitivity** on the holdout test set.

## 🏗️ Project Architecture
```text
📦 project_root
 ┣ 📂 app                  # Gradio deployment application
 ┣ 📂 src                  
 ┃ ┣ 📂 data               # Dataloaders, splits, and Albumentations pipeline
 ┃ ┣ 📂 models             # EfficientNet backbone and custom classification head
 ┃ ┣ 📂 training           # Staged training loop, Focal Loss, AMP, MLflow
 ┃ ┣ 📂 eval               # Clinical metrics (Sensitivity, Specificity, AUC)
 ┃ ┗ 📂 xai                # Grad-CAM implementation
 ┣ 📜 config.yaml          # Centralized hyperparameters
 ┣ 📜 run_training.py      # Main execution script
 ┗ 📜 requirements.txt
