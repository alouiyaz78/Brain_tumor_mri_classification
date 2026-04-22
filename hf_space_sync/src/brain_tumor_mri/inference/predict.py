from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brain_tumor_mri.models.builder import build_model
from brain_tumor_mri.data.transforms import get_eval_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = {0: "no_tumor", 1: "tumor"}


def load_model(
    checkpoint_path: str | Path,
    model_name: str = "efficientnet_b0",
    num_classes: int = 2,
    pretrained: bool = False,
    img_size: int = 224,
) -> torch.nn.Module:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        img_size=img_size,
    ).to(DEVICE)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


def prepare_pil_image(image: Image.Image, img_size: int = 224):
    image_rgb = image.convert("RGB")
    image_np = np.array(image_rgb)

    transform = get_eval_transforms(img_size=img_size)
    tensor = transform(image_rgb).unsqueeze(0).to(DEVICE)

    return tensor, image_np


@torch.no_grad()
def predict_pil_image(
    model: torch.nn.Module,
    image: Image.Image,
    threshold: float = 0.5,
    img_size: int = 224,
) -> dict[str, Any]:
    tensor, image_np = prepare_pil_image(image, img_size=img_size)
    logits = model(tensor)
    probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    tumor_prob = float(probs[1])
    pred_idx = 1 if tumor_prob >= threshold else 0

    return {
        "pred_index": pred_idx,
        "pred_label": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {
            "no_tumor": float(probs[0]),
            "tumor": tumor_prob,
        },
        "raw_image": image_np,
    }
    
prepare_image = prepare_pil_image    