"""
model.py — CropGuard classifier
EfficientNet-B3 backbone with progressive unfreezing and focal loss.
Supports swap to ViT-B/16 via --model flag.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from dataset import CLASS_NAMES, NUM_CLASSES


# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss — down-weights easy examples so the model focuses on hard ones.
    Especially useful for PlantVillage's class imbalance (healthy >> diseased).
    Paper: Lin et al. 2017, RetinaNet.
    """

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ── Model ─────────────────────────────────────────────────────────────────────
class CropGuardModel(pl.LightningModule):
    """
    EfficientNet-B3 (or ViT-B/16) fine-tuned on PlantVillage.

    Training strategy:
      1. Freeze backbone, train classification head for `warmup_epochs`.
      2. At epoch `unfreeze_epoch`, progressively unfreeze backbone blocks.
      3. Apply cosine LR schedule with warm-up.
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b3",
        num_classes: int = NUM_CLASSES,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        unfreeze_epoch: int = 10,
        max_epochs: int = 80,
        use_focal_loss: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,        # Remove default head
            drop_rate=dropout,
        )
        feature_dim = self.backbone.num_features

        # ── Classification head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        # ── Loss ──────────────────────────────────────────────────────────────
        self.criterion = FocalLoss(gamma=2.0) if use_focal_loss else nn.CrossEntropyLoss(label_smoothing=0.1)

        # ── Freeze backbone initially ─────────────────────────────────────────
        self._freeze_backbone()

        # ── Metrics storage ───────────────────────────────────────────────────
        self.val_preds = []
        self.val_targets = []

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()

        self.val_preds.extend(preds.cpu().numpy())
        self.val_targets.extend(labels.cpu().numpy())

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        if len(self.val_preds) == 0:
            return

        preds = np.array(self.val_preds)
        targets = np.array(self.val_targets)

        # Macro F1
        from sklearn.metrics import f1_score
        f1 = f1_score(targets, preds, average="macro", zero_division=0)
        self.log("val/f1_macro", f1, prog_bar=True)

        self.val_preds.clear()
        self.val_targets.clear()

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        preds = logits.argmax(dim=1)
        self.val_preds.extend(preds.cpu().numpy())
        self.val_targets.extend(labels.cpu().numpy())

    def on_test_epoch_end(self):
        preds = np.array(self.val_preds)
        targets = np.array(self.val_targets)
        print("\n" + "="*60)
        print("TEST SET RESULTS")
        print("="*60)
        print(classification_report(targets, preds, target_names=CLASS_NAMES, digits=3))
        self.val_preds.clear()
        self.val_targets.clear()

    def configure_optimizers(self):
        # Only head params initially (backbone frozen)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Cosine annealing LR
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6
        )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def on_epoch_start(self):
        # Progressive unfreezing at configured epoch
        if self.current_epoch == self.hparams.unfreeze_epoch:
            print(f"\n[Epoch {self.current_epoch}] Unfreezing backbone for fine-tuning...")
            self._unfreeze_backbone()
            # Update optimizer to include all params
            for param_group in self.optimizers().param_groups:
                param_group["lr"] = self.hparams.lr * 0.1   # Lower LR for backbone
