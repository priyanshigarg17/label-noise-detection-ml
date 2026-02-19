import os
import joblib
import numpy as np

from data_loader import load_data
from preprocessing import (
    split_data,
    create_preprocessing_pipeline,
    apply_preprocessing
)
from model import get_model, train_model
from noise_detection import detect_suspicious_samples
from evaluation import evaluate
from visualize import compare_models



DATA_PATH = "../data/noisy_classification_dataset_15_percent_noise.csv"
MODEL_PATH = "../models/trained_model.pkl"
PIPELINE_PATH = "../models/preprocessing_pipeline.pkl"


def main():

    # ---------------------------
    # 1. Load Data
    # ---------------------------
    X, y = load_data(DATA_PATH)

    # ---------------------------
    # 2. Split Data
    # ---------------------------
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ---------------------------
    # 3. Preprocessing
    # ---------------------------
    preprocessing_pipeline = create_preprocessing_pipeline()

    X_train_processed, X_test_processed, preprocessing_pipeline = apply_preprocessing(
        preprocessing_pipeline,
        X_train,
        X_test
    )

    # ---------------------------
    # 4. Baseline Model
    # ---------------------------
    model = get_model()
    model = train_model(model, X_train_processed, y_train)

    baseline_metrics = evaluate(model, X_test_processed, y_test)
    print("Baseline Metrics:", baseline_metrics)

    baseline_model = model


    # ---------------------------
    # 5. Detect Noisy Samples
    # ---------------------------
    suspicious = detect_suspicious_samples(
        X_train,
        y_train
    )

    print("Suspicious Samples:", len(suspicious))

    # ---------------------------
    # 6. Sample Re-weighting
    # ---------------------------
     