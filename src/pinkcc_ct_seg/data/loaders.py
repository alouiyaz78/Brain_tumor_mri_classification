from collections import Counter

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler


def make_weighted_sampler(subset: Subset) -> WeightedRandomSampler:
    """
    Crée un sampler pondéré pour compenser le déséquilibre des classes.
    """
    labels = [subset[i][1].item() for i in range(len(subset))]
    counts = Counter(labels)

    class_weights = {cls: 1.0 / count for cls, count in counts.items()}
    sample_weights = [class_weights[label] for label in labels]

    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def make_loaders(
    train_set,
    val_set,
    batch_size: int = 32,
    num_workers: int = 2,
    weighted: bool = False,
    pin_memory: bool = False,
):
    if weighted:
        sampler = make_weighted_sampler(train_set)
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader