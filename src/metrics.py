# src/metrics.py
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import numpy as np

def compute_metrics(y_true, y_prob, threshold=0.5):
    # transform probabilities to binary predictions
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    # compute metrics
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
