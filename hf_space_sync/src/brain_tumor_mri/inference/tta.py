from __future__ import annotations

import torch
from PIL import Image


def _extract_positive_prob(outputs: torch.Tensor) -> float:
    return torch.softmax(outputs, dim=1)[:, 1].item()


@torch.no_grad()
def predict_probabilities_tta(model, dataset, device, tta_transforms):
    """
    Retourne :
    - y_true : labels réels
    - probs_1 : probabilité moyenne de la classe positive avec TTA

    dataset : BrainMRIDataset avec transform=None
    tta_transforms : liste de transforms (Compose)
    """
    model.eval()

    all_probs = []
    all_targets = []

    for img_path, label in dataset.samples:
        image = Image.open(img_path).convert("RGB")

        probs = []

        for tta_tfms in tta_transforms:
            x = tta_tfms(image).unsqueeze(0).to(device)
            outputs = model(x)
            prob_1 = _extract_positive_prob(outputs)
            probs.append(prob_1)

        mean_prob = sum(probs) / len(probs)

        all_probs.append(mean_prob)
        all_targets.append(label)

    return all_targets, all_probs