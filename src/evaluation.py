"""Shared binary classification metrics: Response=1 is the positive class."""
import numpy as np
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


def positive_scores(estimator, X):
    """Return class-1 ranking scores, using probabilities or decision margin.

    SVM margins are not probabilities; no internal non-grouped probability
    calibration is used. This interface can also evaluate Tuananh's Logistic.
    """
    classes = np.asarray(estimator.classes_)
    if set(classes) != {0, 1}:
        raise ValueError("The estimator must have binary classes 0 and 1.")
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(X))[:, int(np.flatnonzero(classes == 1)[0])]
    scores = np.asarray(estimator.decision_function(X))
    return scores if classes[1] == 1 else -scores


def classification_metrics(y_true, y_pred, scores):
    """PR-AUC uses trapezoidal integration; AP is a separate metric.

    Precision/recall/F1 use hard class predictions and pos_label=1.
    ROC-AUC, PR-AUC and AP use continuous class-1 scores, not hard labels.
    """
    y_true, y_pred, scores = map(np.asarray, (y_true, y_pred, scores))
    if any(a.ndim != 1 for a in (y_true, y_pred, scores)) or not (len(y_true) == len(y_pred) == len(scores)):
        raise ValueError("Expected aligned one-dimensional labels and scores.")
    if len(y_true) == 0 or not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("Labels must be nonempty and binary 0/1.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite.")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result = dict(rows=len(y_true), positive_rate=float(y_true.mean()),
                  accuracy=float(accuracy_score(y_true, y_pred)),
                  precision=float(precision_score(y_true, y_pred, zero_division=0)),
                  recall=float(recall_score(y_true, y_pred, zero_division=0)),
                  f1=float(f1_score(y_true, y_pred, zero_division=0)),
                  tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                  pr_auc=np.nan, average_precision=np.nan, roc_auc=np.nan)
    if len(np.unique(y_true)) == 2:
        precision, recall, _ = precision_recall_curve(y_true, scores)
        result.update(pr_auc=float(auc(recall, precision)),
                      average_precision=float(average_precision_score(y_true, scores)),
                      roc_auc=float(roc_auc_score(y_true, scores)))
    return result
