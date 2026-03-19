from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    average_precision_score, # <-- 1. Nouvel import pour l'AUPRC
)


# 2. L'ancienne fonction reste INTACTE (Ne casse aucun ancien notebook)
def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(
            y_true,
            y_pred,
            zero_division=0,
            digits=4
        ),
    }

# 3. Nouvelle fonction dédiée pour le Challenge Pancréas
def compute_auprc(y_true, y_pred_proba) -> float:
    """
    Calcule l'AUPRC (Average Precision).
    Attention : y_pred_proba doit contenir les probabilités de la classe positive (1 / Tumeur),
    et non les prédictions binaires brutes (0 ou 1).
    """
    return average_precision_score(y_true, y_pred_proba)