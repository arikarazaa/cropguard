"""
Basic smoke tests — run with: pytest tests/
No trained checkpoint needed; uses random weights.
"""
import sys
sys.path.insert(0, "src")

import numpy as np
import pytest
import torch
import timm
import torch.nn as nn

from dataset import CLASS_NAMES, NUM_CLASSES, get_risk, get_val_transforms
from gradcam import GradCAM
from risk_scorer import assess_risk


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512), nn.SiLU(), nn.Linear(512, NUM_CLASSES)
        )
    def forward(self, x):
        return self.classifier(self.backbone(x))


def test_dataset_constants():
    assert NUM_CLASSES == 38
    assert len(CLASS_NAMES) == 38
    assert "Tomato___healthy" in CLASS_NAMES


def test_transforms_output_shape():
    t = get_val_transforms(224)
    img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    result = t(image=img)["image"]
    assert result.shape == (3, 224, 224)


def test_risk_healthy():
    ra = assess_risk("Tomato___healthy", 0.99)
    assert ra.level == "None"
    assert ra.yield_loss_pct == 0


def test_risk_critical():
    ra = assess_risk("Tomato___Late_blight", 0.96)
    assert ra.level == "Critical"
    assert ra.yield_loss_pct > 0


def test_risk_low_confidence_downgrades():
    levels = ["None","Low","Moderate","High","Critical"]
    ra_hi = assess_risk("Potato___Late_blight", 0.95)
    ra_lo = assess_risk("Potato___Late_blight", 0.50)
    assert levels.index(ra_lo.level) < levels.index(ra_hi.level)


def test_model_forward():
    model = _DummyModel()
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, NUM_CLASSES)


def test_gradcam_shape():
    model = _DummyModel()
    model.eval()
    cam = GradCAM(model, model.backbone.blocks[-1])
    heatmap, class_idx, probs = cam(torch.randn(1, 3, 224, 224))
    assert heatmap.shape == (224, 224)
    assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0
    assert 0 <= class_idx < NUM_CLASSES
    assert abs(probs.sum() - 1.0) < 1e-4
