"""
app.py — CropGuard Gradio web demo
Run: python app.py
"""

import gradio as gr
import os
import sys
import tempfile
from pathlib import Path


import numpy as np
from PIL import Image

sys.path.insert(0, "src")

CHECKPOINT_PATH = os.environ.get("CHECKPOINT", "models/best_checkpoint.pth")
DEMO_MODE = not Path(CHECKPOINT_PATH).exists()

if not DEMO_MODE:
    from inference import CropGuardPredictor
    predictor = CropGuardPredictor(CHECKPOINT_PATH)
else:
    print("⚠️  No checkpoint found — running in DEMO mode with simulated predictions.")
    predictor = None

# ── Simulated predictions for demo mode (no trained model needed) ─────────────
DEMO_PREDICTIONS = {
    "default": {
        "disease": "Tomato___Late_blight",
        "confidence": 0.961,
        "risk": {"level": "Critical", "yield_loss_pct": 62},
        "top_k_predictions": [
            ("Tomato___Late_blight", 0.961),
            ("Tomato___Early_blight", 0.023),
            ("Tomato___Septoria_leaf_spot", 0.010),
            ("Tomato___healthy", 0.004),
            ("Tomato___Bacterial_spot", 0.002),
        ],
    }
}


def format_disease_name(raw: str) -> str:
    """Convert 'Tomato___Late_blight' → 'Tomato — Late blight'"""
    parts = raw.split("___")
    if len(parts) == 2:
        return f"{parts[0]} — {parts[1].replace('_', ' ')}"
    return raw.replace("_", " ")


def predict(image: Image.Image) -> tuple:
    """Main inference function called by Gradio."""
    if image is None:
        return None, "No image uploaded.", "", ""

    # Save to temp file for predictor
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    if DEMO_MODE:
        result = DEMO_PREDICTIONS["default"]
        gradcam_img = image   # Return original in demo mode
    else:
        result = predictor.predict(tmp_path, gradcam=True, top_k=5)
        gradcam_img = result["gradcam_image"] or image

    os.unlink(tmp_path)

    disease = format_disease_name(result["disease"])
    conf = result["confidence"]
    risk = result["risk"]

    # ── Prediction summary markdown ────────────────────────────────────────────
    risk_emoji = {"None": "✅", "Moderate": "⚠️", "High": "🔴", "Critical": "🚨"}.get(risk["level"], "⚠️")
    summary = f"""
## {risk_emoji} {disease}

| Field | Value |
|---|---|
| **Confidence** | {conf*100:.1f}% |
| **Risk level** | {risk['level']} |
| **Estimated yield loss** | {risk['yield_loss_pct']}% |
"""
    if risk["level"] != "None":
        summary += f"\n> **Recommendation:** Detected infection pattern consistent with *{disease}*. Suggest targeted treatment within 48h."
    else:
        summary += "\n> ✅ No disease detected. Leaf shows healthy growth patterns."

    # ── Top-5 table ────────────────────────────────────────────────────────────
    top5_md = "### Top-5 predictions\n\n| Rank | Disease | Confidence |\n|---|---|---|\n"
    for i, (name, prob) in enumerate(result["top_k_predictions"], 1):
        bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        top5_md += f"| {i} | {format_disease_name(name)} | {prob*100:.1f}% `{bar}` |\n"

    # ── Grad-CAM caption ──────────────────────────────────────────────────────
    gradcam_caption = (
        "🔴 Red = high model activation  ·  🔵 Blue = low activation  ·  "
        "Heatmap shows which leaf regions most influenced the prediction (Grad-CAM, Block7)"
        if not DEMO_MODE else
        "Demo mode — train a model and set CHECKPOINT env var for real Grad-CAM."
    )

    return gradcam_img, summary, top5_md, gradcam_caption


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="CropGuard — AI Crop Disease Detection",
    theme=gr.themes.Soft(primary_hue="green"),
) as demo:

    gr.Markdown("""
# 🌿 CropGuard — AI Crop Disease Detection
**EfficientNet-B3 + Grad-CAM explainability · Trained on PlantVillage (54,305 images · 38 classes)**

Upload a leaf image to get a disease prediction, confidence score, risk assessment, and Grad-CAM saliency map.
Research project aligned with the **AgroRisk** and **OPTIcut** agenda at De Montfort University.
""")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload leaf image", height=300)
            predict_btn = gr.Button("🔍 Analyse leaf", variant="primary", size="lg")

            gr.Examples(
                examples=[],
                inputs=image_input,
                label="Example images (add to assets/ folder)",
            )

        with gr.Column(scale=1):
            gradcam_output = gr.Image(label="Grad-CAM saliency map", height=300)
            gradcam_caption = gr.Markdown()

    with gr.Row():
        with gr.Column():
            prediction_output = gr.Markdown(label="Prediction")
        with gr.Column():
            top5_output = gr.Markdown(label="Top-5 classes")

    predict_btn.click(
        fn=predict,
        inputs=[image_input],
        outputs=[gradcam_output, prediction_output, top5_output, gradcam_caption],
    )

    image_input.change(
        fn=predict,
        inputs=[image_input],
        outputs=[gradcam_output, prediction_output, top5_output, gradcam_caption],
    )

    gr.Markdown("""
---
**Dataset**: [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) · 
**Paper**: Hughes & Salathé (2015) · 
**Architecture**: EfficientNet-B3 (timm) · 
**Explainability**: Grad-CAM (Selvaraju et al. 2017)
""")

if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)
