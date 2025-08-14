import numpy as np
from typing import Dict
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, average_precision_score

def classification_metrics_from_logits(logits: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """
    logits: (N, C), y_true: (N,)
    Возвращает: accuracy, f1_macro, f1_micro, balanced_acc, map_macro
    """
    probs = softmax_np(logits)
    y_pred = probs.argmax(axis=1)
    acc  = float(accuracy_score(y_true, y_pred))
    f1ma = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1mi = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    # mAP по one-vs-rest вероятностям
    C = probs.shape[1]
    y_ovr = np.eye(C, dtype=int)[y_true]     # (N,C)
    map_macro = float(average_precision_score(y_ovr, probs, average="macro"))
    return {"accuracy": acc, "f1_macro": f1ma, "f1_micro": f1mi, "balanced_acc": bacc, "map_macro": map_macro}

def softmax_np(x: np.ndarray):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)