from __future__ import annotations

import numpy as np
import cv2
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from brain_tumor_mri.inference.predict import prepare_pil_image


def generate_gradcam_outputs(
    model,
    image: Image.Image,
    cam_threshold: float = 0.45,
    img_size: int = 224,
):
    input_tensor, _ = prepare_pil_image(image, img_size=img_size)

    target_layer = model.features[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    image_resized = image.resize((img_size, img_size)).convert("RGB")
    img_np = np.array(image_resized).astype(np.float32) / 255.0

    overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    heatmap_uint8 = np.uint8(255 * grayscale_cam)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    binary_mask = (grayscale_cam >= cam_threshold).astype(np.uint8)

    mask_rgb = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    mask_rgb[:, :, 0] = binary_mask * 255

    img_uint8 = np.uint8(img_np * 255)
    intersection = img_uint8.copy()
    intersection[binary_mask == 0] = 0

    return {
        "grayscale_cam": grayscale_cam,
        "overlay": overlay,
        "heatmap_rgb": heatmap_rgb,
        "mask_rgb": mask_rgb,
        "intersection": intersection,
    }