import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def apply_threshold(probs, threshold=0.5):
    """
    Convertit des probabilités en prédictions binaires.
    probs : liste ou array de probabilités de la classe 1
    """
    probs = np.asarray(probs)
    return (probs >= threshold).astype(int)


def evaluate_threshold(y_true, probs, threshold):
    """
    Évalue un seuil donné.
    """
    y_pred = apply_threshold(probs, threshold)

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def scan_thresholds(y_true, probs, thresholds=None):
    """
    Teste plusieurs seuils et retourne les résultats.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    results = []
    for thr in thresholds:
        results.append(evaluate_threshold(y_true, probs, thr))
    return results


def select_threshold_by_best_f1(results):
    """
    Sélectionne le seuil qui maximise le F1.
    """
    return max(results, key=lambda x: x["f1"])


def select_threshold_by_recall_constraint(results, min_recall=0.90, min_precision=None):
    """
    Sélectionne le seuil qui respecte un recall minimal.
    Optionnellement, impose aussi une précision minimale.
    Parmi les seuils admissibles, on choisit le meilleur F1.
    """
    filtered = [r for r in results if r["recall"] >= min_recall]

    if min_precision is not None:
        filtered = [r for r in filtered if r["precision"] >= min_precision]

    if not filtered:
        return None

    return max(filtered, key=lambda x: x["f1"])