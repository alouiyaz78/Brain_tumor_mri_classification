from pathlib import Path
import sys
import numpy as np
from PIL import Image
import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brain_tumor_mri.inference.predict import load_model, predict_pil_image
from brain_tumor_mri.evaluation.gradcam import generate_gradcam_outputs

CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "checkpoints" / "best_efficientnet_b0.pt"

model = load_model(str(CHECKPOINT_PATH))


def run_inference(image_np, cam_threshold):
    if image_np is None:
        return None, None, None, None, None, None

    image = Image.fromarray(image_np).convert("RGB")

    result = predict_pil_image(
        model=model,
        image=image,
        threshold=0.5,
        img_size=224,
    )

    gradcam_result = generate_gradcam_outputs(
        model=model,
        image=image,
        cam_threshold=cam_threshold,
        img_size=224,
    )

    prediction_text = (
        f"Predicted class: {result['pred_label']}\n"
        f"Confidence: {result['confidence']:.4f}\n"
        f"Tumor probability: {result['probabilities']['tumor']:.4f}"
    )

    return (
        np.array(image),
        prediction_text,
        gradcam_result["overlay"],
        gradcam_result["mask_rgb"],
        gradcam_result["intersection"],
        gradcam_result["heatmap_rgb"],
    )


demo = gr.Interface(
    fn=run_inference,
    inputs=[
        gr.Image(type="numpy", label="Upload MRI image"),
        gr.Slider(0.10, 0.90, value=0.45, step=0.05, label="Pseudo-segmentation threshold"),
    ],
    outputs=[
        gr.Image(label="Original image"),
        gr.Textbox(label="Prediction"),
        gr.Image(label="Grad-CAM overlay"),
        gr.Image(label="Pseudo-segmentation"),
        gr.Image(label="Intersection"),
        gr.Image(label="Heatmap"),
    ],
    title="Brain Tumor MRI Classification",
    description="Binary classification with Grad-CAM and pseudo-segmentation.",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()