"""
train.py — CropGuard training entrypoint
Run:
    python src/train.py --data_dir data/plantvillage --epochs 80
"""

import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger

from dataset import PlantVillageDataModule
from model import CropGuardModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train CropGuard crop disease detector")

    # Data
    parser.add_argument("--data_dir", type=str, default="data/plantvillage", help="Path to PlantVillage dataset root")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.10)

    # Model
    parser.add_argument(
        "--model", type=str, default="efficientnet_b3",
        choices=["efficientnet_b3", "efficientnet_b4", "resnet50", "vit_b_16", "mobilenetv3_large_100"],
        help="Backbone architecture (timm model name)"
    )
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no_focal_loss", action="store_true", help="Use CE loss instead of Focal")

    # Training
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--unfreeze_epoch", type=int, default=10)
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--log_dir", type=str, default="results/logs")

    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    dm = PlantVillageDataModule(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CropGuardModel(
        model_name=args.model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        unfreeze_epoch=args.unfreeze_epoch,
        max_epochs=args.epochs,
        use_focal_loss=not args.no_focal_loss,
        dropout=args.dropout,
    )

    print(f"\n{'='*60}")
    print(f"  CropGuard Training")
    print(f"  Backbone : {args.model}")
    print(f"  Dataset  : {args.data_dir}")
    print(f"  Epochs   : {args.epochs} (patience={args.patience})")
    print(f"  LR       : {args.lr}  |  Batch: {args.batch_size}")
    print(f"{'='*60}\n")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.output_dir,
        filename=f"{args.model}_best_{{epoch:02d}}_{{val/f1_macro:.4f}}",
        monitor="val/f1_macro",
        mode="max",
        save_top_k=3,
        verbose=True,
    )

    early_stop_cb = EarlyStopping(
        monitor="val/f1_macro",
        patience=args.patience,
        mode="max",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = CSVLogger(args.log_dir, name=args.model)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",       # GPU if available, else CPU
        devices="auto",
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, RichProgressBar()],
        logger=logger,
        precision="16-mixed",     # AMP — 2× faster on modern GPUs
        deterministic=False,      # True hurts speed; False is fine for research
        log_every_n_steps=10,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.fit(model, dm)

    # ── Test with best checkpoint ─────────────────────────────────────────────
    print(f"\nBest checkpoint: {checkpoint_cb.best_model_path}")
    trainer.test(model, dm, ckpt_path="best")

    print(f"\nDone. Model saved to: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    main()
