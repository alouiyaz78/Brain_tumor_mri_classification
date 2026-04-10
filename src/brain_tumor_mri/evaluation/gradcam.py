from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


def disable_inplace_relu(module: nn.Module) -> None:
    for child_name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, child_name, nn.ReLU(inplace=False))
        else:
            disable_inplace_relu(child)


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, inputs, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())

        score = output[:, class_idx]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture gradients/activations.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        probs = F.softmax(output, dim=1).detach().cpu().numpy()[0]
        pred_class = int(np.argmax(probs))
        pred_prob = float(np.max(probs))
        return cam, pred_class, pred_prob, probs


def load_original_rgb(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def make_input_tensor(path: str | Path, transform: Callable, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


def overlay_heatmap(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize((image_rgb.shape[1], image_rgb.shape[0]))
    cam_resized = np.array(cam_resized) / 255.0
    cmap = plt.get_cmap("jet")
    heatmap = cmap(cam_resized)[..., :3]
    overlay = np.clip((1 - alpha) * (image_rgb / 255.0) + alpha * heatmap, 0, 1)
    return overlay
