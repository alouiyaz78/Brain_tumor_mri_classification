from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from tqdm import tqdm


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

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        n_samples += images.size(0)

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()

        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(labels.detach().cpu().tolist())

    epoch_loss = running_loss / n_samples
    epoch_acc = correct / n_samples

    return {
        "loss": epoch_loss,
        "acc": epoch_acc,
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

    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        n_samples += images.size(0)

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()

        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(labels.detach().cpu().tolist())

    epoch_loss = running_loss / n_samples
    epoch_acc = correct / n_samples

    return {
        "loss": epoch_loss,
        "acc": epoch_acc,
        "preds": all_preds,
        "targets": all_targets,
    }