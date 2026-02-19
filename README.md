# Label Noise Detection and Mitigation using Machine Learning

## 📌 Project Overview

This project focuses on detecting and reducing the impact of label noise in a supervised classification dataset. 

The dataset contains 15% intentionally corrupted labels. The goal is to:
- Identify suspicious (potentially mislabeled) samples
- Reduce their impact using machine learning techniques
- Compare model performance before and after noise handling

---

## 🎯 Problem Statement

In real-world datasets, labels are often incorrect due to:
- Human annotation errors
- Data entry mistakes
- Automated labeling inaccuracies

Label noise negatively affects model performance and generalization.

This project implements an ML-based approach to:
1. Detect noisy labels using cross-validation and confidence analysis
2. Reduce their influence using sample re-weighting
3. Evaluate model improvement

---

## 🏗 Project Structure

label_noise_project/
│
├── data/
│ └── noisy_classification_dataset.csv
│
├── src/
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── model.py
│ ├── noise_detection.py
│ ├── evaluation.py
│ └── train_pipeline.py
│
├── models/
│ ├── trained_model.pkl
label_noise_project/
│
├── data/
│ └── noisy_classification_dataset.csv
│
├── src/
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── model.py
│ ├── noise_detection.py
│ ├── evaluation.py
│ └── train_pipeline.py
│
├── models/
│ ├── trained_model.pkl
