"""
app/app_gradio.py
Classification binaire de tumeurs cérébrales par IRM — interface Gradio.
"""

from pathlib import Path
import sys
import os

import numpy as np
from PIL import Image
import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brain_tumor_mri.inference.predict import load_model, predict_pil_image
from brain_tumor_mri.evaluation.gradcam import generate_gradcam_outputs


# =========================================================================
# Chargement du modèle
# =========================================================================
CHECKPOINT_PATH = APP_DIR / "artifacts" / "checkpoints" / "best_efficientnet_b0.pt"

print("PROJECT_ROOT =", PROJECT_ROOT)
print("APP_DIR =", APP_DIR)
print("CHECKPOINT_PATH =", CHECKPOINT_PATH)
print("Checkpoint exists =", CHECKPOINT_PATH.exists())

model = load_model(str(CHECKPOINT_PATH))


# =========================================================================
# CSS
# =========================================================================
CUSTOM_CSS = """
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}

.app-header {
    background: linear-gradient(135deg,
        var(--primary-50) 0%,
        var(--secondary-50) 100%);
    border: 1px solid var(--primary-200);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
}

.app-header .eyebrow {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--primary-700);
    font-weight: 600;
    margin-bottom: 4px;
}

.app-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0 0 8px 0;
    color: var(--body-text-color);
    line-height: 1.2;
}

.app-header .meta {
    font-size: 0.85rem;
    color: var(--neutral-600);
    line-height: 1.6;
}

.app-header .meta strong {
    color: var(--body-text-color);
}

.header-logo {
    display: flex;
    align-items: center;
    justify-content: center;
}

.card {
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 20px;
}

.card-title {
    font-weight: 600;
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--neutral-500);
    margin: 0 0 16px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-color-primary);
}

.result-panel {
    padding: 20px 24px;
    border-radius: 12px;
    border-left: 5px solid;
    margin-bottom: 16px;
    font-family: var(--font);
}

.result-panel.tumor {
    background: #fef2f2;
    border-left-color: #dc2626;
    color: #7f1d1d;
}

.result-panel.notumor {
    background: #f0fdf4;
    border-left-color: #16a34a;
    color: #14532d;
}

.result-panel.neutral {
    background: var(--neutral-100);
    border-left-color: var(--neutral-400);
    color: var(--neutral-700);
}

.result-eyebrow {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.75;
    font-weight: 600;
    margin-bottom: 6px;
}

.result-class {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0 0 16px 0;
    line-height: 1.2;
}

.result-metrics {
    display: flex;
    flex-wrap: wrap;
    gap: 28px;
}

.result-metric {
    display: flex;
    flex-direction: column;
    min-width: 100px;
}

.result-metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    opacity: 0.7;
    margin-bottom: 2px;
}

.result-metric-value {
    font-size: 1.2rem;
    font-weight: 700;
}

.metric-tile {
    text-align: center;
    padding: 20px 16px;
    background: var(--background-fill-primary);
    border-radius: 12px;
    border: 1px solid var(--border-color-primary);
}

.metric-tile .tile-value {
    font-size: 2.4rem;
    font-weight: 800;
    color: var(--primary-600);
    line-height: 1;
    margin-bottom: 6px;
    font-variant-numeric: tabular-nums;
}

.metric-tile .tile-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--neutral-500);
    font-weight: 600;
}

.metric-tile .tile-caption {
    font-size: 0.75rem;
    color: var(--neutral-400);
    margin-top: 4px;
}

.context-box {
    padding: 14px 18px;
    background: var(--neutral-50);
    border-left: 3px solid var(--primary-400);
    border-radius: 6px;
    margin-bottom: 16px;
    font-size: 0.9rem;
    color: var(--neutral-700);
    line-height: 1.55;
}

.tabs > .tab-nav button {
    font-weight: 500;
    padding: 10px 18px !important;
}

.output-image,
.input-image {
    border-radius: 8px;
}

.dark .result-panel.tumor {
    background: #450a0a;
    color: #fecaca;
}

.dark .result-panel.notumor {
    background: #052e16;
    color: #bbf7d0;
}

.dark .context-box {
    background: var(--neutral-900);
    color: var(--neutral-300);
}
"""


# =========================================================================
# Helpers
# =========================================================================
def build_result_html(result: dict | None) -> str:
    if result is None:
        return (
            '<div class="result-panel neutral">'
            '<div class="result-eyebrow">En attente</div>'
            '<div class="result-class">Aucune analyse</div>'
            '<div style="font-size:0.9rem; opacity:0.8;">'
            'Importez une image IRM puis cliquez sur « Lancer l’analyse ».'
            '</div>'
            '</div>'
        )

    is_tumor = result["pred_label"] == "tumor"
    css_class = "tumor" if is_tumor else "notumor"
    display_label = "Tumeur détectée" if is_tumor else "Aucune tumeur détectée"
    confidence_pct = f"{result['confidence'] * 100:.1f}%"
    tumor_prob_pct = f"{result['probabilities']['tumor'] * 100:.1f}%"

    return (
        f'<div class="result-panel {css_class}">'
        f'<div class="result-eyebrow">Résultat de la prédiction</div>'
        f'<div class="result-class">{display_label}</div>'
        f'<div class="result-metrics">'
        f'  <div class="result-metric">'
        f'    <span class="result-metric-label">Confiance</span>'
        f'    <span class="result-metric-value">{confidence_pct}</span>'
        f'  </div>'
        f'  <div class="result-metric">'
        f'    <span class="result-metric-label">Probabilité tumeur</span>'
        f'    <span class="result-metric-value">{tumor_prob_pct}</span>'
        f'  </div>'
        f'</div>'
        f'</div>'
    )


# =========================================================================
# Logique métier
# =========================================================================
def run_inference(image_np, cam_threshold):
    if image_np is None:
        return build_result_html(None), None, None, None, None

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

    return (
        build_result_html(result),
        gradcam_result["overlay"],
        gradcam_result["mask_rgb"],
        gradcam_result["intersection"],
        gradcam_result["heatmap_rgb"],
    )


def run_batch(file_paths, cam_threshold):
    if not file_paths:
        return [], [["Aucun fichier", "", "", ""]]

    gallery_items = []
    prediction_rows = []

    for file_path in file_paths:
        try:
            image = Image.open(file_path).convert("RGB")

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

            filename = os.path.basename(file_path)

            gallery_items.append(
                (
                    gradcam_result["overlay"],
                    f"{filename} | {result['pred_label']}",
                )
            )

            prediction_rows.append([
                filename,
                result["pred_label"],
                f"{result['confidence']:.4f}",
                f"{result['probabilities']['tumor']:.4f}",
            ])

        except Exception as e:
            filename = os.path.basename(file_path)
            error_img = np.zeros((224, 224, 3), dtype=np.uint8)
            gallery_items.append((error_img, f"{filename} | erreur"))
            prediction_rows.append([filename, "Erreur", "-", str(e)])

    return gallery_items, prediction_rows


def reset_analysis():
    return (
        None,
        0.45,
        build_result_html(None),
        None,
        None,
        None,
        None,
    )


def reset_batch():
    return None, 0.45, [], [["—", "—", "—", "—"]]


# =========================================================================
# Theme
# =========================================================================
THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="emerald",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)


# =========================================================================
# App
# =========================================================================
with gr.Blocks(
    title="Classification Binaire de Tumeurs Cérébrales par IRM",
    fill_width=True,
) as demo:

    with gr.Row(elem_classes="app-header"):
        with gr.Column(scale=1, min_width=160, elem_classes="header-logo"):
            gr.Image(
                value=str(APP_DIR / "logo_lacite.png"),
                show_label=False,
                container=False,
                height=110,
            )

        with gr.Column(scale=5):
            gr.HTML(
                """
                <div class="eyebrow">Collège La Cité — Sciences des données appliquées</div>
                <h1>Classification Binaire de Tumeurs Cérébrales par IRM</h1>
                <div class="meta">
                  <strong>Équipe :</strong>
                  Redouane Hamecha · Malick Diatara · Yazid Aloui · Omar El Medjoubi ·
                  Mohamed Achoch · N'Guessan Yves Guichard Allou<br>
                  <strong>Encadrement :</strong> Nacereddine Benzebouchi
                </div>
                """
            )

    with gr.Tabs():

        with gr.Tab("Analyse détaillée"):
            gr.HTML(
                '<div class="context-box">'
                'Importez une image IRM cérébrale. Le modèle <strong>EfficientNet-B0</strong> '
                'classe l’image en <em>tumeur</em> ou <em>absence de tumeur</em>, et '
                '<strong>Grad-CAM</strong> met en évidence les zones qui ont influencé la décision.'
                '</div>'
            )

            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    with gr.Group(elem_classes="card"):
                        gr.HTML('<div class="card-title">Entrée</div>')

                        input_image = gr.Image(
                            type="numpy",
                            label="Image IRM",
                            height=300,
                            elem_classes="input-image",
                        )

                        threshold = gr.Slider(
                            minimum=0.10,
                            maximum=0.90,
                            value=0.45,
                            step=0.05,
                            label="Seuil de pseudo-segmentation",
                            info=(
                                "Seuil appliqué sur la carte d’attention. "
                                "Bas = zone large et bruitée. "
                                "Haut = zones les plus saillantes uniquement."
                            ),
                        )

                        with gr.Row():
                            run_button = gr.Button(
                                "Lancer l’analyse",
                                variant="primary",
                                scale=2,
                            )
                            clear_button = gr.Button(
                                "Réinitialiser",
                                variant="secondary",
                                scale=1,
                            )

                with gr.Column(scale=2):
                    result_html = gr.HTML(build_result_html(None))

                    with gr.Group(elem_classes="card"):
                        gr.HTML('<div class="card-title">Visualisations Grad-CAM</div>')

                        with gr.Row():
                            overlay_img = gr.Image(
                                label="Superposition Grad-CAM",
                                height=220,
                                elem_classes="output-image",
                            )
                            mask_img = gr.Image(
                                label="Pseudo-segmentation",
                                height=220,
                                elem_classes="output-image",
                            )

                        with gr.Row():
                            intersection_img = gr.Image(
                                label="Intersection image × zone",
                                height=220,
                                elem_classes="output-image",
                            )
                            heatmap_img = gr.Image(
                                label="Carte thermique brute",
                                height=220,
                                elem_classes="output-image",
                            )

                        gr.HTML(
                            '<div class="context-box" style="margin-top:16px;margin-bottom:0;">'
                            '<strong>Lecture :</strong> '
                            'la <em>superposition</em> montre les zones influentes, '
                            'la <em>pseudo-segmentation</em> isole ces zones par seuillage, '
                            'l’<em>intersection</em> conserve uniquement la partie saillante de '
                            'l’image originale, et la <em>carte thermique</em> affiche l’intensité '
                            'brute de l’attention.'
                            '</div>'
                        )

            run_button.click(
                fn=run_inference,
                inputs=[input_image, threshold],
                outputs=[result_html, overlay_img, mask_img, intersection_img, heatmap_img],
            )

            clear_button.click(
                fn=reset_analysis,
                outputs=[
                    input_image,
                    threshold,
                    result_html,
                    overlay_img,
                    mask_img,
                    intersection_img,
                    heatmap_img,
                ],
            )

        with gr.Tab("Mode batch"):
            gr.HTML(
                '<div class="context-box">'
                'Importez plusieurs coupes IRM pour comparer rapidement les prédictions. '
                'Utile par exemple pour analyser plusieurs coupes d’un même cerveau.'
                '</div>'
            )

            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes="card"):
                        gr.HTML('<div class="card-title">Images à analyser</div>')

                        batch_files = gr.File(
                            file_count="multiple",
                            file_types=[".png", ".jpg", ".jpeg"],
                            label="Sélectionnez plusieurs fichiers",
                        )

                        batch_threshold = gr.Slider(
                            minimum=0.10,
                            maximum=0.90,
                            value=0.45,
                            step=0.05,
                            label="Seuil de pseudo-segmentation",
                            info="Même paramètre que dans l’analyse détaillée, appliqué à toutes les images.",
                        )

                        with gr.Row():
                            batch_button = gr.Button(
                                "Lancer la comparaison",
                                variant="primary",
                                scale=2,
                            )
                            batch_clear = gr.Button(
                                "Réinitialiser",
                                variant="secondary",
                                scale=1,
                            )

                with gr.Column(scale=2):
                    with gr.Group(elem_classes="card"):
                        gr.HTML('<div class="card-title">Galerie des superpositions Grad-CAM</div>')
                        batch_gallery = gr.Gallery(
                            show_label=False,
                            columns=3,
                            rows=2,
                            height=420,
                            object_fit="contain",
                        )

                    with gr.Group(elem_classes="card"):
                        gr.HTML('<div class="card-title">Tableau récapitulatif</div>')
                        batch_table = gr.Dataframe(
                            headers=["Image", "Classe prédite", "Confiance", "Probabilité tumeur"],
                            datatype=["str", "str", "str", "str"],
                            show_label=False,
                            interactive=False,
                            wrap=True,
                        )

            batch_button.click(
                fn=run_batch,
                inputs=[batch_files, batch_threshold],
                outputs=[batch_gallery, batch_table],
            )

            batch_clear.click(
                fn=reset_batch,
                outputs=[batch_files, batch_threshold, batch_gallery, batch_table],
            )

        with gr.Tab("Performances"):
            with gr.Row(equal_height=True):
                with gr.Column():
                    gr.HTML(
                        '<div class="metric-tile">'
                        '<div class="tile-value">0.9976</div>'
                        '<div class="tile-label">PR-AUC (test)</div>'
                        '<div class="tile-caption">qualité de classement</div>'
                        '</div>'
                    )
                with gr.Column():
                    gr.HTML(
                        '<div class="metric-tile">'
                        '<div class="tile-value">92%</div>'
                        '<div class="tile-label">Rappel (seuil 0.3 + TTA)</div>'
                        '<div class="tile-caption">267 tumeurs détectées sur 289</div>'
                        '</div>'
                    )
                with gr.Column():
                    gr.HTML(
                        '<div class="metric-tile">'
                        '<div class="tile-value">100%</div>'
                        '<div class="tile-label">Précision</div>'
                        '<div class="tile-caption">aucun faux positif</div>'
                        '</div>'
                    )

            with gr.Group(elem_classes="card"):
                gr.HTML('<div class="card-title">Courbe Precision-Recall</div>')
                gr.HTML(
                    '<div class="context-box">'
                    'La <strong>PR-AUC</strong> est la métrique clé dans un contexte '
                    'médical où détecter toutes les tumeurs compte plus que de minimiser '
                    'les fausses alertes. Une PR-AUC proche de 1 indique que le modèle '
                    'sépare très bien les classes sur tous les seuils.'
                    '</div>'
                )
                gr.Image(
                    value=str(APP_DIR / "prauc_curve.png"),
                    show_label=False,
                    height=380,
                    container=False,
                )

            with gr.Group(elem_classes="card"):
                gr.HTML('<div class="card-title">Matrices de confusion — impact du seuil</div>')
                gr.HTML(
                    '<div class="context-box">'
                    'Trois configurations comparées. Le passage du seuil 0.5 au seuil 0.3 '
                    '<strong>rattrape 11 tumeurs manquées</strong>. L’ajout du TTA '
                    '(Test-Time Augmentation) en rattrape 3 de plus, sans jamais créer de '
                    'faux positif.'
                    '</div>'
                )
                gr.Image(
                    value=str(APP_DIR / "confusion_matrices.png"),
                    show_label=False,
                    height=280,
                    container=False,
                )

            with gr.Group(elem_classes="card"):
                gr.HTML('<div class="card-title">Courbes d’apprentissage</div>')
                gr.HTML(
                    '<div class="context-box">'
                    'La ligne verticale marque la transition entre les phases '
                    '<em>A</em> (tête de classification seule) et <em>B</em> '
                    '(fine-tuning complet). Le pic de loss à l’époque 10 est normal : '
                    'le réseau réajuste tous ses poids d’un coup puis converge.'
                    '</div>'
                )
                gr.Image(
                    value=str(APP_DIR / "training_curves.png"),
                    show_label=False,
                    height=360,
                    container=False,
                )


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CUSTOM_CSS)