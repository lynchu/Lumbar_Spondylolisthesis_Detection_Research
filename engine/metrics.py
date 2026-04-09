import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    # handle degenerate confusion matrix safely
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        tn = fp = fn = tp = 0
    else:
        tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp + 1e-12)
    sensitivity = tp / (tp + fn + 1e-12)
    balanced_acc = 0.5 * (specificity + sensitivity)

    if len(np.unique(y_true)) > 1:
        auroc = float(roc_auc_score(y_true, y_prob))
        ap = float(average_precision_score(y_true, y_prob))
    else:
        auroc = 0.0
        ap = 0.0

    return {
        "threshold": float(threshold),
        "auroc": auroc,
        "ap": ap,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_acc": float(balanced_acc),
        "specificity": float(specificity),
        "sensitivity": float(sensitivity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_best_threshold(y_true, y_prob, metric="balanced_acc", thresholds=None):
    """
    Select threshold from validation/monitor probabilities.

    metric options:
      - "balanced_acc"
      - "f1"
      - "accuracy"
      - "precision"
      - "recall"
      - "youden"
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 181)

    best_threshold = 0.5
    best_score = -1.0
    best_metrics = None

    for th in thresholds:
        m = compute_binary_metrics(y_true, y_prob, threshold=th)

        if metric == "youden":
            score = m["sensitivity"] + m["specificity"] - 1.0
        else:
            if metric not in m:
                raise ValueError(f"Unsupported threshold metric: {metric}")
            score = m[metric]

        if score > best_score:
            best_score = float(score)
            best_threshold = float(th)
            best_metrics = m

    return {
        "best_threshold": best_threshold,
        "best_score": best_score,
        "metric_name": metric,
        "metrics_at_best_threshold": best_metrics,
    }