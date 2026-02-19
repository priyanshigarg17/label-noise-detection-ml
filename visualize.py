import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


# ---------------------------------
# 1️⃣ Accuracy Comparison Plot
# ---------------------------------
def plot_accuracy_comparison(baseline_metrics, clean_metrics):

    labels = ["Baseline", "After Noise Handling"]
    accuracies = [
        baseline_metrics["accuracy"],
        clean_metrics["accuracy"]
    ]

    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, accuracies)

    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison")

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                 f"{height:.3f}", ha='center', va='bottom')

    plt.ylim(0, 1)
    plt.show()


# ---------------------------------
# 2️⃣ Confusion Matrix Plot
# ---------------------------------
def plot_confusion_matrix(model, X_test, y_test, title="Confusion Matrix"):

    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# ---------------------------------
# 3️⃣ Combined Comparison
# ---------------------------------
def compare_models(
    baseline_model,
    clean_model,
    X_test,
    y_test,
    baseline_metrics,
    clean_metrics
):

    # Accuracy Comparison
    plot_accuracy_comparison(baseline_metrics, clean_metrics)

    # Confusion Matrices
    plot_confusion_matrix(
        baseline_model,
        X_test,
        y_test,
        title="Baseline Model Confusion Matrix"
    )

    plot_confusion_matrix(
        clean_model,
        X_test,
        y_test,
        title="After Noise Handling Confusion Matrix"
    )
