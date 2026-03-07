from collections import Counter
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

# N'oubliez pas l'import de vos transformations !
from pinkcc_ct_seg.data.transforms import get_train_transforms, get_eval_transforms


def make_weighted_sampler(dataset) -> WeightedRandomSampler:
    """
    Crée un sampler pondéré pour compenser le déséquilibre des classes.
    """
    # Extraction des labels (suppose que dataset[i] renvoie (image, label))
    labels = [dataset[i][1] for i in range(len(dataset))]
    
    # Si les labels sont des tenseurs, on les convertit avec .item()
    if isinstance(labels[0], torch.Tensor):
        labels = [label.item() for label in labels]
        
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
    augmentation_level: str = 'standard'  # <-- Notre ajout pour le raffinage !
):
    """
    Construit les DataLoaders PyTorch avec gestion optionnelle du déséquilibre (weighted)
    et choix du niveau d'augmentation de données.
    """
    # 1. Application dynamique des transformations
    train_set.transform = get_train_transforms(augmentation_level=augmentation_level)
    val_set.transform = get_eval_transforms()

    # 2. Création des chargeurs
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