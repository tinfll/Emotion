"""FER2013 dataset loading. Supports both the CSV format and the ImageFolder format."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
NUM_CLASSES = 7
IMG_SIZE = 48
FER_MEAN = [0.5077]
FER_STD = [0.2550]


def build_transforms(train: bool, augment: bool = True) -> transforms.Compose:
    if train and augment:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(FER_MEAN, FER_STD),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(FER_MEAN, FER_STD),
    ])


class FER2013CSV(Dataset):
    """Reads the canonical fer2013.csv (columns: emotion, pixels, Usage)."""

    SPLITS = {
        "train": "Training",
        "val": "PublicTest",
        "test": "PrivateTest",
    }

    def __init__(self, csv_path: str, split: str = "train", augment: bool = True):
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {list(self.SPLITS)}")
        df = pd.read_csv(csv_path)
        df = df[df["Usage"] == self.SPLITS[split]].reset_index(drop=True)
        self.labels = df["emotion"].astype(int).values
        self.pixels = df["pixels"].values
        self.transform = build_transforms(train=(split == "train"), augment=augment)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        arr = np.fromstring(self.pixels[idx], sep=" ", dtype=np.uint8).reshape(IMG_SIZE, IMG_SIZE)
        img = Image.fromarray(arr, mode="L")
        return self.transform(img), int(self.labels[idx])


class FER2013ImageFolder(Dataset):
    """Reads the Hugging Face / Kaggle 'images' layout: data/{train,test}/<emotion>/*.jpg"""

    EMOTION_TO_IDX = {name.lower(): i for i, name in enumerate(EMOTION_LABELS)}

    def __init__(self, root: str, split: str = "train", augment: bool = True):
        split_dir_map = {"train": "train", "val": "test", "test": "test"}
        root_path = Path(root) / split_dir_map[split]
        if not root_path.exists():
            raise FileNotFoundError(f"Split directory not found: {root_path}")
        self.samples: list[Tuple[Path, int]] = []
        for emo_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
            label = self.EMOTION_TO_IDX.get(emo_dir.name.lower())
            if label is None:
                continue
            for img_path in emo_dir.glob("*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((img_path, label))
        if not self.samples:
            raise RuntimeError(f"No images found under {root_path}")
        self.transform = build_transforms(train=(split == "train"), augment=augment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        return self.transform(img), label


def FER2013Dataset(data_dir: str, split: str = "train", augment: bool = True) -> Dataset:
    """Auto-detect CSV vs ImageFolder layout."""
    csv_path = Path(data_dir) / "fer2013.csv"
    if csv_path.exists():
        return FER2013CSV(str(csv_path), split=split, augment=augment)
    if (Path(data_dir) / "train").exists():
        return FER2013ImageFolder(data_dir, split=split, augment=augment)
    raise FileNotFoundError(
        f"Could not find FER2013 in '{data_dir}'. Expected either {csv_path} "
        f"or a 'train/'+'test/' folder. Run scripts/download_fer2013.py first."
    )


def build_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 0,
    augment: bool = True,
    val_split: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Returns (train_loader, val_loader, test_loader). For ImageFolder layout we
    derive a val split from the training set deterministically."""
    train_ds = FER2013Dataset(data_dir, split="train", augment=augment)
    try:
        val_ds = FER2013Dataset(data_dir, split="val", augment=False)
        test_ds = FER2013Dataset(data_dir, split="test", augment=False)
    except FileNotFoundError:
        val_ds = test_ds = FER2013Dataset(data_dir, split="test", augment=False)

    if isinstance(train_ds, FER2013ImageFolder) and val_ds is test_ds:
        n = len(train_ds)
        val_size = int(0.1 * n)
        gen = torch.Generator().manual_seed(42)
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [n - val_size, val_size], generator=gen)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader


def class_distribution(data_dir: str) -> dict[str, int]:
    ds = FER2013Dataset(data_dir, split="train", augment=False)
    if isinstance(ds, FER2013CSV):
        labels = ds.labels
    else:
        labels = [lbl for _, lbl in ds.samples]
    counts = np.bincount(np.asarray(labels), minlength=NUM_CLASSES)
    return {EMOTION_LABELS[i]: int(c) for i, c in enumerate(counts)}
