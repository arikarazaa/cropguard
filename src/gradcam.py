"""
gradcam.py — Grad-CAM implementation
Hooks into the last convolutional block of EfficientNet-B3 (or ViT) to produce
saliency maps that highlight which leaf regions drove the prediction.

Reference: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks
           via Gradient-based Localization" ICCV 2017.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Usage:
        cam = GradCAM(model, target_layer=model.backbone.blocks[-1])
        heatmap = cam(input_tensor, class_idx=None)  # None → argmax class
        overlay = cam.overlay(original_image, heatmap)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer

        self._gradients: torch.Tensor | None = None
        self._activations: torch.Tensor | None = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: int | None = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.

        Args:
            input_tensor: Preprocessed image tensor [1, C, H, W]
            class_idx: Target class index. None → uses predicted class.

        Returns:
            heatmap: np.ndarray [H, W] in [0, 1] — resized to input resolution.
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Pool gradients across spatial dimensions → [C]
        gradients = self._gradients       # [1, C, H', W']
        activations = self._activations   # [1, C, H', W']

        weights = gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, H', W']
        cam = F.relu(cam)

        # Upsample to input resolution
        H, W = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam, int(class_idx), probs.squeeze().detach().cpu().numpy()

    @staticmethod
    def apply_colormap(heatmap: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """Convert [0,1] heatmap to RGB colour image using OpenCV colourmap."""
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        coloured = cv2.applyColorMap(heatmap_uint8, colormap)
        return cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)

    @staticmethod
    def overlay(
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Blend original image with Grad-CAM heatmap.

        Args:
            original_image: np.ndarray [H, W, 3] uint8 RGB
            heatmap: np.ndarray [H, W] float in [0, 1]
            alpha: Heatmap opacity (0=hidden, 1=full)
            colormap: OpenCV colourmap (default: JET)

        Returns:
            Blended RGB image as np.ndarray [H, W, 3] uint8
        """
        coloured = GradCAM.apply_colormap(heatmap, colormap)

        # Resize heatmap to match original if needed
        if coloured.shape[:2] != original_image.shape[:2]:
            coloured = cv2.resize(coloured, (original_image.shape[1], original_image.shape[0]))

        blended = (alpha * coloured + (1 - alpha) * original_image).astype(np.uint8)
        return blended

    @staticmethod
    def to_pil(image_array: np.ndarray) -> Image.Image:
        """Convert np.ndarray [H, W, 3] uint8 to PIL Image."""
        return Image.fromarray(image_array.astype(np.uint8))


def get_target_layer(model) -> torch.nn.Module:
    """
    Auto-detect the last convolutional block for Grad-CAM hooks.
    Works for EfficientNet (timm) and torchvision ViT.
    """
    backbone = model.backbone

    # EfficientNet-B3 via timm: last block in backbone.blocks
    if hasattr(backbone, "blocks"):
        return backbone.blocks[-1]

    # ResNet: layer4
    if hasattr(backbone, "layer4"):
        return backbone.layer4[-1]

    # ViT: last transformer block
    if hasattr(backbone, "blocks"):
        return backbone.blocks[-1].norm1

    raise ValueError(
        f"Could not auto-detect target layer for {type(backbone).__name__}. "
        "Please pass target_layer manually."
    )
