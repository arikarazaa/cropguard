"""
inference.py — Single-image inference with Grad-CAM explainability
CLI:
    python src/inference.py --image leaf.jpg --checkpoint models/best.pth --save_gradcam out.png

Python API:
    predictor = CropGuardPredictor("models/best.pth")
    result = predictor.predict("leaf.jpg", gradcam=True)
"""

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import CLASS_NAMES, get_risk, get_val_transforms
from gradcam import GradCAM, get_target_layer
from model import CropGuardModel


class CropGuardPredictor:
    """
    High-level inference API wrapping the trained model + Grad-CAM.

    Example:
        predictor = CropGuardPredictor("models/best_checkpoint.pth")
        result = predictor.predict("tomato_leaf.jpg", gradcam=True)
        result["gradcam_image"].save("output.png")
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        img_size: int = 224,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.transform = get_val_transforms(img_size)

        print(f"Loading model from {checkpoint_path} on {self.device}...")
        self.model = CropGuardModel.load_from_checkpoint(checkpoint_path, map_location=self.device)
        self.model.eval()
        self.model.to(self.device)

        # Grad-CAM — hooks last conv block
        target_layer = get_target_layer(self.model)
        self.gradcam = GradCAM(self.model, target_layer)

        print(f"Model ready. Classes: {len(CLASS_NAMES)}")

    def preprocess(self, image_path: str) -> tuple[torch.Tensor, np.ndarray]:
        """Load and preprocess image. Returns (tensor, original_rgb_array)."""
        original = np.array(Image.open(image_path).convert("RGB"))
        transformed = self.transform(image=original)["image"]
        tensor = transformed.unsqueeze(0).to(self.device)
        return tensor, original

    def predict(
        self,
        image_path: str,
        gradcam: bool = True,
        top_k: int = 5,
    ) -> dict:
        """
        Run inference on a single image.

        Returns dict with:
            - disease (str): Predicted class name
            - confidence (float): Softmax confidence of top class
            - risk (dict): {"level": str, "yield_loss_pct": int}
            - top_k_predictions (list): [(class_name, prob), ...]
            - gradcam_image (PIL.Image | None): Overlaid heatmap image
            - heatmap (np.ndarray | None): Raw [H, W] float heatmap
        """
        tensor, original = self.preprocess(image_path)

        with torch.set_grad_enabled(gradcam):
            if gradcam:
                heatmap, class_idx, probs = self.gradcam(tensor, class_idx=None)
            else:
                with torch.no_grad():
                    logits = self.model(tensor)
                    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
                    class_idx = probs.argmax()
                heatmap = None

        disease = CLASS_NAMES[class_idx]
        confidence = float(probs[class_idx])
        risk = get_risk(disease)

        # Top-K predictions
        top_indices = probs.argsort()[::-1][:top_k]
        top_k_predictions = [(CLASS_NAMES[i], float(probs[i])) for i in top_indices]

        # Grad-CAM overlay image
        gradcam_image = None
        if gradcam and heatmap is not None:
            original_resized = cv2.resize(original, (self.img_size, self.img_size))
            overlay = GradCAM.overlay(original_resized, heatmap, alpha=0.45)
            gradcam_image = GradCAM.to_pil(overlay)

        return {
            "disease": disease,
            "confidence": confidence,
            "risk": risk,
            "top_k_predictions": top_k_predictions,
            "gradcam_image": gradcam_image,
            "heatmap": heatmap,
            "class_idx": class_idx,
        }

    def batch_predict(self, image_paths: list[str], gradcam: bool = False) -> list[dict]:
        """Run inference on multiple images."""
        return [self.predict(p, gradcam=gradcam) for p in image_paths]


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="CropGuard inference + Grad-CAM")
    parser.add_argument("--image", required=True, help="Path to leaf image")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--save_gradcam", type=str, default=None, help="Save Grad-CAM overlay to this path")
    parser.add_argument("--no_gradcam", action="store_true", help="Skip Grad-CAM (faster)")
    parser.add_argument("--top_k", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()

    predictor = CropGuardPredictor(args.checkpoint)
    result = predictor.predict(args.image, gradcam=not args.no_gradcam, top_k=args.top_k)

    print("\n" + "="*50)
    print("  CROPGUARD PREDICTION")
    print("="*50)
    print(f"  Disease    : {result['disease']}")
    print(f"  Confidence : {result['confidence']*100:.1f}%")
    print(f"  Risk level : {result['risk']['level']} ({result['risk']['yield_loss_pct']}% yield loss)")
    print("\n  Top predictions:")
    for name, prob in result["top_k_predictions"]:
        bar = "█" * int(prob * 30)
        print(f"    {prob*100:5.1f}% {bar}  {name}")
    print("="*50)

    if result["gradcam_image"] and args.save_gradcam:
        result["gradcam_image"].save(args.save_gradcam)
        print(f"\nGrad-CAM saved to: {args.save_gradcam}")


if __name__ == "__main__":
    main()
