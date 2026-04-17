from pathlib import Path
import sys
import numpy as np
from PIL import Image
import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brain_tumor_mri.inference.predict import load_model, predict_pil_image
from brain_tumor_mri.evaluation.gradcam import generate_gradcam_outputs

# =========================
# Chargement du modèle
# =========================
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "checkpoints" / "best_efficientnet_b0.pt"
model = load_model(str(CHECKPOINT_PATH))


# =========================
# Fonction d'inférence
# =========================
def run_inference(image_np, cam_threshold):
    if image_np is None:
        return "Aucune image fournie.", None, None, None, None

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
        f"### Résultat de la prédiction\n"
        f"- **Classe prédite :** {result['pred_label']}\n"
        f"- **Confiance :** {result['confidence']:.4f}\n"
        f"- **Probabilité de tumeur :** {result['probabilities']['tumor']:.4f}"
    )

    return (
        prediction_text,
        gradcam_result["overlay"],
        gradcam_result["mask_rgb"],
        gradcam_result["intersection"],
        gradcam_result["heatmap_rgb"],
    )


with gr.Blocks(
    title="Classification Binaire de Tumeurs Cérébrales par IRM",
    fill_width=True,
) as demo:
    # =========================
    # En-tête
    # =========================
    with gr.Row():
        with gr.Column(scale=1, min_width=180):
            gr.Image(
                value="app/logo_lacite.png",
                show_label=False,
                container=False,
                height=120,
            )

        with gr.Column(scale=4):
            gr.Markdown(
                """
### Collège La Cité  
**Programme : Sciences des données appliquées**

## Classification Binaire de Tumeurs Cérébrales par IRM

**Réalisé par :**
- Redouane Hamecha
- Malick Diatara
- Yazid Aloui
- Omar El Medjoubi
- Mohamed Achoch
- N'Guessan Yves Guichard Allou

**Encadré par :**
- Nacereddine Benzebouchi
"""
            )

    with gr.Tabs():
        # ==========================================================
        # ONGLET 1 : Analyse d'une image
        # ==========================================================
        with gr.Tab("Analyse d’une image"):
            gr.Markdown(
                """
### Description générale

Cette application permet de classifier une image **IRM cérébrale** en deux classes :
- **Tumeur**
- **Absence de tumeur**

Le modèle utilisé est **EfficientNet-B0**, pré-entraîné sur **ImageNet** puis **fine-tuné**
pour une tâche de classification binaire sur des images IRM cérébrales.

### Explicabilité du modèle

Une visualisation de type **Grad-CAM** est utilisée pour mettre en évidence les régions
de l’image qui ont le plus influencé la décision du modèle.

### Pseudo-segmentation

La pseudo-segmentation est obtenue en appliquant un **seuil** sur la carte d’attention.
Il ne s’agit pas d’une segmentation médicale exacte pixel par pixel, mais d’une
**approximation visuelle des régions importantes** pour la prédiction.
"""
            )

            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    input_image = gr.Image(
                        type="numpy",
                        label="Importer une image IRM",
                        height=320,
                    )

                    threshold = gr.Slider(
                        minimum=0.10,
                        maximum=0.90,
                        value=0.45,
                        step=0.05,
                        label="Seuil de pseudo-segmentation",
                        info=(
                            "Ce paramètre contrôle la sensibilité du masque extrait à partir de la carte d’attention. "
                            "Un seuil faible conserve une zone plus large mais plus bruitée. "
                            "Un seuil élevé conserve uniquement les zones les plus importantes."
                        ),
                    )

                    run_button = gr.Button("Lancer l’analyse", variant="primary")
                    clear_button = gr.Button("Réinitialiser")

                with gr.Column(scale=2):
                    prediction_box = gr.Markdown()

                    with gr.Row():
                        overlay_img = gr.Image(
                            label="Superposition Grad-CAM",
                            height=260,
                        )
                        mask_img = gr.Image(
                            label="Pseudo-segmentation",
                            height=260,
                        )

                    with gr.Row():
                        intersection_img = gr.Image(
                            label="Intersection",
                            height=260,
                        )
                        heatmap_img = gr.Image(
                            label="Carte thermique",
                            height=260,
                        )

                    gr.Markdown(
                        """
### Interprétation des résultats

- **Superposition Grad-CAM** : zones importantes superposées sur l’image  
- **Pseudo-segmentation** : masque binaire obtenu après seuillage  
- **Intersection** : partie de l’image conservée dans la zone détectée  
- **Carte thermique** : intensité de l’attention du modèle  

**Remarque :** l’image originale est déjà visible dans la zone d’importation à gauche.
"""
                    )

            run_button.click(
                fn=run_inference,
                inputs=[input_image, threshold],
                outputs=[
                    prediction_box,
                    overlay_img,
                    mask_img,
                    intersection_img,
                    heatmap_img,
                ],
            )

            clear_button.click(
                fn=lambda: (None, 0.45, "", None, None, None, None),
                outputs=[
                    input_image,
                    threshold,
                    prediction_box,
                    overlay_img,
                    mask_img,
                    intersection_img,
                    heatmap_img,
                ],
            )

        # ==========================================================
        # ONGLET 2 : Performance globale
        # ==========================================================
        with gr.Tab("Performance globale du modèle"):
            gr.Markdown(
                """
### Performance globale du modèle

Cet onglet présente les performances globales obtenues sur les données d’évaluation.

### Pourquoi la PR-AUC est importante ?

Dans ce projet, la **PR-AUC (Precision-Recall AUC)** est particulièrement pertinente car
le problème est **déséquilibré** et la détection correcte des tumeurs est prioritaire.

Une **PR-AUC élevée** indique que le modèle classe efficacement les cas tumoraux
par rapport aux cas sains sur l’ensemble des seuils possibles.
"""
            )

            gr.Markdown(
                """
### Résumé des performances

- **PR-AUC test : 0.9976**
- Le modèle présente une excellente capacité de séparation entre les classes.
- Les matrices de confusion montrent néanmoins que le **seuil de décision**
  influence le compromis entre faux négatifs et faux positifs.
- L’ajustement du seuil et l’utilisation du **TTA** améliorent la détection des tumeurs.
"""
            )

            with gr.Row():
                gr.Image(
                    value="app/prauc_curve.png",
                    label="Courbe Precision-Recall (test)",
                    height=360,
                )

            gr.Markdown(
                """
### Matrices de confusion

Les matrices de confusion ci-dessous comparent plusieurs configurations :
- **Seuil standard 0.5**
- **Seuil ajusté 0.3**
- **TTA + seuil 0.3**

Elles permettent de visualiser l’impact du choix du seuil sur les faux négatifs,
ce qui est particulièrement important dans un contexte médical.
"""
            )

            with gr.Row():
                gr.Image(
                    value="app/confusion_matrices.png",
                    label="Comparaison des matrices de confusion",
                    height=360,
                )

            gr.Markdown(
                """
### Courbes d’apprentissage

Les courbes suivantes présentent l’évolution de la **loss** et de l’**accuracy**
au cours de l’entraînement. Elles permettent d’observer la convergence du modèle
et l’effet de la transition méthodologique utilisée pendant l’apprentissage.
"""
            )

            with gr.Row():
                gr.Image(
                    value="app/training_curves.png",
                    label="Loss et Accuracy par époque",
                    height=360,
                )

            gr.Markdown(
                """
### Conclusion sur les performances

- Le modèle obtient des performances globales très élevées.
- La **PR-AUC proche de 1** confirme une très bonne qualité de classement.
- Les matrices de confusion montrent que le **réglage du seuil**
  améliore la sensibilité du modèle.
- L’approche combinant **classification + explicabilité** rend le système plus interprétable.
"""
            )


if __name__ == "__main__":
    demo.launch()