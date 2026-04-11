from __future__ import annotations

import torch
from torch import nn
from tqdm import tqdm


def _extract_positive_probs(outputs: torch.Tensor) -> torch.Tensor:
    """
    Cas classification binaire avec 2 logits :
    retourne la probabilité de la classe positive.
    """
    return torch.softmax(outputs, dim=1)[:, 1]


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    model.train()

    running_loss = 0.0
    n_samples = 0
    correct = 0

    all_preds = []
    all_targets = []
    all_probs = []

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        probs = _extract_positive_probs(outputs)
        preds = torch.argmax(outputs, dim=1)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size
        correct += (preds == labels).sum().item()

        all_probs.extend(probs.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(labels.detach().cpu().tolist())

    epoch_loss = running_loss / n_samples
    epoch_acc = correct / n_samples

    return {
        "loss": epoch_loss,
        "acc": epoch_acc,
        "probs": all_probs,
        "preds": all_preds,
        "targets": all_targets,
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()

    running_loss = 0.0
    n_samples = 0
    correct = 0

    all_preds = []
    all_targets = []
    all_probs = []

    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        probs = _extract_positive_probs(outputs)
        preds = torch.argmax(outputs, dim=1)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size
        correct += (preds == labels).sum().item()

        all_probs.extend(probs.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(labels.detach().cpu().tolist())

    epoch_loss = running_loss / n_samples
    epoch_acc = correct / n_samples

    return {
        "loss": epoch_loss,
        "acc": epoch_acc,
        "probs": all_probs,
        "preds": all_preds,
        "targets": all_targets,
    }


@torch.no_grad()
def predict_probabilities(model, loader, device):
    model.eval()

    all_probs = []
    all_targets = []

    for images, labels in tqdm(loader, desc="Predicting", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = _extract_positive_probs(outputs)

        all_probs.extend(probs.detach().cpu().tolist())
        all_targets.extend(labels.detach().cpu().tolist())

    return all_targets, all_probs