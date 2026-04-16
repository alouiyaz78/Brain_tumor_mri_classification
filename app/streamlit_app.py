from pathlib import Path
import sys

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brain_tumor_mri.inference.predict import load_model, predict_pil_image

st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="wide",
)

CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "checkpoints" / "best_efficientnet_b0.pt"


@st.cache_resource
def get_model():
    return load_model(str(CHECKPOINT_PATH))


st.title("Brain Tumor MRI Classification")
st.markdown("Classification binaire **tumor / no_tumor** à partir d’images IRM.")

uploaded_files = st.file_uploader(
    "Charger une ou plusieurs images IRM",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if uploaded_files:
    model = get_model()
    st.success(f"{len(uploaded_files)} image(s) chargée(s).")

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        image = Image.open(uploaded_file).convert("RGB")

        result = predict_pil_image(
            model=model,
            image=image,
            threshold=0.5,
            img_size=224,
        )

        st.markdown("---")
        st.subheader(f"Image {idx} — {uploaded_file.name}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Classe prédite", result["pred_label"])
        m2.metric("Confiance", f"{result['confidence']:.4f}")
        m3.metric("Probabilité tumeur", f"{result['probabilities']['tumor']:.4f}")

        st.image(
            image,
            caption="Image originale",
            width="stretch",
        )

else:
    st.info("Charge une ou plusieurs images pour lancer la prédiction.")
