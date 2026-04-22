from __future__ import annotations

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from brain_tumor_mri.inference.predict import prepare_pil_image


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def _jet_colormap(gray_uint8: np.ndarray) -> np.ndarray:
    x = gray_uint8.astype(np.float32) / 255.0
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    heatmap = np.stack([r, g, b], axis=-1)
    return (heatmap * 255).astype(np.uint8)


def _resize_array(arr: np.ndarray, size=(300, 300)) -> np.ndarray:
    return np.array(Image.fromarray(arr).resize(size))


def _make_overlay(img_rgb_uint8: np.ndarray, heatmap_rgb_uint8: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    base = img_rgb_uint8.astype(np.float32)
    heat = heatmap_rgb_uint8.astype(np.float32)
    overlay = (1 - alpha) * base + alpha * heat
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _get_target_layer(model: torch.nn.Module):
    if hasattr(model, "features"):
        return model.features[-1]
    raise ValueError("Impossible de trouver la couche cible pour Grad-CAM sur ce modèle.")


def generate_gradcam_outputs(
    model,
    image: Image.Image,
    cam_threshold: float = 0.45,
    img_size: int = 224,
):
    input_tensor, _ = prepare_pil_image(image, img_size=img_size)
    target_layer = _get_target_layer(model)

    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    try:
        try:
            handle_bwd = target_layer.register_full_backward_hook(backward_hook)
        except Exception:
            handle_bwd = target_layer.register_backward_hook(backward_hook)

        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        pred_class = logits.argmax(dim=1)
        score = logits[:, pred_class.item()].sum()
        score.backward()

    finally:
        handle_fwd.remove()
        handle_bwd.remove()

    acts = activations[0]           # [1, C, H, W]
    grads = gradients[0]            # [1, C, H, W]

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=False)  # [1, H, W]
    cam = F.relu(cam)

    cam = cam[0].cpu().numpy()
    cam = _normalize_cam(cam)

    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((img_size, img_size))
    grayscale_cam = np.array(cam_img).astype(np.float32) / 255.0

    image_resized = image.resize((img_size, img_size)).convert("RGB")
    img_uint8 = np.array(image_resized).astype(np.uint8)

    heatmap_uint8 = np.uint8(255 * grayscale_cam)
    heatmap_rgb = _jet_colormap(heatmap_uint8)

    overlay = _make_overlay(img_uint8, heatmap_rgb, alpha=0.4)

    binary_mask = (grayscale_cam >= cam_threshold).astype(np.uint8)

    mask_rgb = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    mask_rgb[:, :, 0] = binary_mask * 255

    intersection = img_uint8.copy()
    intersection[binary_mask == 0] = 0

    FINAL_SIZE = (300, 300)

    overlay = _resize_array(overlay, FINAL_SIZE)
    heatmap_rgb = _resize_array(heatmap_rgb, FINAL_SIZE)
    mask_rgb = _resize_array(mask_rgb, FINAL_SIZE)
    intersection = _resize_array(intersection, FINAL_SIZE)

    return {
        "grayscale_cam": grayscale_cam,
        "overlay": overlay,
        "heatmap_rgb": heatmap_rgb,
        "mask_rgb": mask_rgb,
        "intersection": intersection,
    }
