from pathlib import Path
import sys

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brain_tumor_mri.inference.predict import load_model, predict_pil_image
from brain_tumor_mri.evaluation.gradcam import generate_gradcam_outputs

st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="wide",
)

checkpoint_path = PROJECT_ROOT / "artifacts" / "checkpoints" / "best_efficientnet_b0.pt"

st.title("Brain Tumor MRI Classification")
st.caption("Classification binaire avec visualisation Grad-CAM. Démo éducative uniquement.")

with st.sidebar:
    st.header("Options")
    cam_threshold = st.slider(
        "Seuil pseudo-segmentation",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
    )
    show_heatmap = st.checkbox("Afficher la heatmap brute", value=False)

uploaded_files = st.file_uploader(
    "Upload une ou plusieurs images IRM",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

@st.cache_resource
def get_model():
    return load_model(str(checkpoint_path))

if uploaded_files:
    model = get_model()

    st.success(f"{len(uploaded_files)} image(s) chargée(s).")

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        image = Image.open(uploaded_file).convert("RGB")

        result = predict_pil_image(model, image)
        gradcam_result = generate_gradcam_outputs(
            model=model,
            image=image,
            cam_threshold=cam_threshold,
        )

        st.markdown(f"---")
        st.subheader(f"Image {idx} — {uploaded_file.name}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Classe prédite", result["pred_label"])
        m2.metric("Confiance", f'{result["confidence"]:.4f}')
        m3.metric("Probabilité tumeur", f'{result["probabilities"]["tumor"]:.4f}')

        c1, c2 = st.columns(2)
        with c1:
            st.image(
                image,
                caption="Image originale",
                use_container_width=True,
            )
        with c2:
            st.image(
                gradcam_result["overlay"],
                caption="Grad-CAM overlay",
                use_container_width=True,
            )

        c3, c4 = st.columns(2)
        with c3:
            st.image(
                gradcam_result["mask_rgb"],
                caption="Pseudo-segmentation",
                use_container_width=True,
            )
        with c4:
            st.image(
                gradcam_result["intersection"],
                caption="Intersection image × zone prédite",
                use_container_width=True,
            )

        if show_heatmap:
            st.image(
                gradcam_result["heatmap_rgb"],
                caption="Heatmap Grad-CAM",
                use_container_width=True,
            )
else:
    st.info("Charge une ou plusieurs images pour lancer la prédiction.")