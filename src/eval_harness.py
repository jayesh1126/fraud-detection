from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
import numpy as np

def evaluate(y_true, proba, threshold=0.5, name=""):
    """Standard evaluation: threshold-free ranking metrics + thresholded confusion."""
    tn, fp, fn, tp = confusion_matrix(y_true, proba >= threshold).ravel()
    return {
        "name":      name,
        "pr_auc":    round(average_precision_score(y_true, proba), 4),  # primary
        "roc_auc":   round(roc_auc_score(y_true, proba), 4),
        "threshold": threshold,
        "recall":    round(tp / (tp + fn), 4),
        "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "flagged":   int(tp + fp),      # investigator workload
        "missed_fraud": int(fn),
    }

def pick_threshold_for_budget(proba, budget):
    """Highest threshold that flags at most `budget` claims — set ON VALIDATION ONLY."""
    return float(np.quantile(proba, 1 - budget / len(proba)))
