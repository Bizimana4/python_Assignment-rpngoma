"""Step 4 - Evaluation: accuracy, precision, recall, F1, confusion matrix."""
import matplotlib

matplotlib.use("Agg")  # works without a display (safe in labs and servers)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_model(model, X, y, class_names):
    probs = model.predict(X, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    p, r, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
    }
    report = classification_report(
        y, y_pred, labels=range(len(class_names)),
        target_names=class_names, zero_division=0,
    )
    cm = confusion_matrix(y, y_pred, labels=range(len(class_names)))
    return metrics, report, cm, y_pred


def plot_confusion_matrix(cm, class_names, path):
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax,
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    ax.set_xlabel("Predicted crop")
    ax.set_ylabel("Actual crop")
    ax.set_title("Confusion matrix (test set)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_history(history, path, title="Training curves"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
