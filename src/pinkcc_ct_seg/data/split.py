from collections import Counter
from typing import Sequence

import numpy as np
from sklearn.model_selection import train_test_split


def make_train_val_split(
    labels: Sequence[int],
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retourne deux tableaux d'indices : train_idx, val_idx
    avec split stratifié sur les labels.
    """
    idx_all = np.arange(len(labels))

    train_idx, val_idx = train_test_split(
        idx_all,
        test_size=val_size,
        random_state=random_state,
        stratify=labels,
    )
    return train_idx, val_idx


def describe_split(labels: Sequence[int], name: str = "split") -> None:
    counts = Counter(labels)
    total = len(labels)
    print(f"{name} size = {total}")
    print(f"{name} distribution = {counts}")