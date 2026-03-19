from pathlib import Path

import torch
from torch import nn, optim

from brain_tumor_mri.data.dataset import BrainMRIDataset
from brain_tumor_mri.data.transforms import get_train_transforms, get_eval_transforms
from brain_tumor_mri.data.loaders import make_loaders

from brain_tumor_mri.models.builder import build_model
from brain_tumor_mri.training.engine import train_one_epoch, validate_one_epoch

from brain_tumor_mri.evaluation.metrics import compute_auprc
from brain_tumor_mri.utils import set_seed, get_device, save_checkpoint


def main():
    # =========================
    # Configuration simple
    # =========================
    seed = 42
    model_name = "resnet18"       # ex: "resnet18", "smallcnn"
    pretrained = True
    num_classes = 2
    img_size = 224
    augmentation_level = "standard"
    batch_size = 32
    num_workers = 2
    weighted = True
    pin_memory = False
    learning_rate = 1e-4
    num_epochs = 10
    patience = 3

    # =========================
    # Initialisation
    # =========================
    set_seed(seed)

    device = get_device()
    print("Device:", device)

    train_dir = Path("data/raw/brain_mri/Training")
    val_dir = Path("data/raw/brain_mri/Testing")

    train_tfms = get_train_transforms(
        img_size=img_size,
        augmentation_level=augmentation_level,
    )
    eval_tfms = get_eval_transforms(img_size=img_size)

    train_set = BrainMRIDataset(train_dir, transform=train_tfms)
    val_set = BrainMRIDataset(val_dir, transform=eval_tfms)

    print(f"Train size: {len(train_set)}")
    print(f"Val size: {len(val_set)}")

    train_loader, val_loader = make_loaders(
        train_set,
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        weighted=weighted,
        pin_memory=pin_memory,
    )

    # =========================
    # Modèle
    # =========================
    model = build_model(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=img_size,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint_path = Path(f"artifacts/checkpoints/best_{model_name}.pt")

    # =========================
    # Early stopping
    # =========================
    best_prauc = 0.0
    epochs_no_improve = 0

    # =========================
    # Entraînement
    # =========================
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_stats = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        val_prauc = compute_auprc(
            y_true=val_stats["targets"],
            y_pred_proba=val_stats["probs"],
        )

        print(f"Train loss: {train_stats['loss']:.4f} | Train acc: {train_stats['acc']:.4f}")
        print(f"Val loss:   {val_stats['loss']:.4f} | Val acc:   {val_stats['acc']:.4f}")
        print(f"Val PRAUC:  {val_prauc:.4f}")

        if val_prauc > best_prauc:
            best_prauc = val_prauc
            epochs_no_improve = 0

            save_checkpoint(model, checkpoint_path)
            print(f" Best model saved to: {checkpoint_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")

            if epochs_no_improve >= patience:
                print("\n Early stopping triggered")
                break

    print(f"\nBest validation PRAUC: {best_prauc:.4f}")


if __name__ == "__main__":
    main()