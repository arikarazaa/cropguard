"""
dataset.py — PlantVillage DataModule
Handles download check, train/val/test splits, and augmentation.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── Class names (38 PlantVillage classes) ─────────────────────────────────────
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Apple___healthy", "Blueberry___healthy", "Cherry___Powdery_mildew",
    "Cherry___healthy", "Corn___Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
    "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]

NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# ── Risk mapping: disease class → estimated yield loss % ──────────────────────
RISK_MAP = {
    "Tomato___Late_blight": {"level": "Critical", "yield_loss_pct": 62},
    "Tomato___Early_blight": {"level": "High", "yield_loss_pct": 38},
    "Tomato___Bacterial_spot": {"level": "High", "yield_loss_pct": 35},
    "Potato___Late_blight": {"level": "Critical", "yield_loss_pct": 70},
    "Potato___Early_blight": {"level": "High", "yield_loss_pct": 40},
    "Corn___Northern_Leaf_Blight": {"level": "High", "yield_loss_pct": 30},
    "Corn___Cercospora_leaf_spot_Gray_leaf_spot": {"level": "Moderate", "yield_loss_pct": 24},
    "Apple___Apple_scab": {"level": "Moderate", "yield_loss_pct": 24},
    "Apple___Black_rot": {"level": "High", "yield_loss_pct": 45},
    "Grape___Black_rot": {"level": "Critical", "yield_loss_pct": 55},
}


def get_risk(class_name: str) -> dict:
    """Return risk level and estimated yield loss for a disease class."""
    if "healthy" in class_name.lower():
        return {"level": "None", "yield_loss_pct": 0}
    return RISK_MAP.get(class_name, {"level": "Moderate", "yield_loss_pct": 20})


# ── Augmentation pipelines ────────────────────────────────────────────────────
def get_train_transforms(img_size: int = 224) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(img_size, img_size, scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(img_size + 32, img_size + 32),
        A.CenterCrop(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────
class PlantVillageDataset(Dataset):
    """
    Expects directory structure:
        data_dir/
            Apple___Apple_scab/
                image1.jpg
                image2.jpg
            Apple___Black_rot/
                ...
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[A.Compose] = None,
        class_names: Optional[list] = None,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.class_names = class_names or CLASS_NAMES
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        self.samples: list[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        for class_dir in sorted(self.data_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name not in self.class_to_idx:
                continue
            label = self.class_to_idx[class_name]
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_exts:
                    self.samples.append((img_path, label))

        print(f"Loaded {len(self.samples)} images from {self.data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label


# ── DataModule ────────────────────────────────────────────────────────────────
class PlantVillageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/plantvillage",
        img_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.15,
        test_split: float = 0.10,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

    def setup(self, stage: Optional[str] = None):
        full_dataset = PlantVillageDataset(
            self.data_dir,
            transform=get_train_transforms(self.img_size),
        )

        n = len(full_dataset)
        n_test = int(n * self.test_split)
        n_val = int(n * self.val_split)
        n_train = n - n_val - n_test

        generator = torch.Generator().manual_seed(self.seed)
        self.train_ds, self.val_ds, self.test_ds = torch.utils.data.random_split(
            full_dataset, [n_train, n_val, n_test], generator=generator
        )

        # Override val/test transforms (no augmentation)
        self.val_ds.dataset.transform = get_val_transforms(self.img_size)
        self.test_ds.dataset.transform = get_val_transforms(self.img_size)

        print(f"Split → train: {n_train} | val: {n_val} | test: {n_test}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )
