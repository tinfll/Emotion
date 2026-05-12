"""Inference utility. Includes a 'stylized image' preprocessing path tuned for
NPR / cel-shaded character renders (which differ a lot from FER2013 photos).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from .dataset import EMOTION_LABELS, FER_MEAN, FER_STD, IMG_SIZE, build_transforms
from .models import build_model


def _try_face_crop(img: Image.Image) -> Image.Image:
    """If OpenCV's Haar cascade is available, crop to the largest detected face."""
    try:
        import cv2
    except ImportError:
        return img
    gray = np.array(img.convert("L"))
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(40, 40))
    if len(faces) == 0:
        return img
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.15 * max(w, h))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(gray.shape[1], x + w + pad); y1 = min(gray.shape[0], y + h + pad)
    return img.crop((x0, y0, x1, y1))


def preprocess_image(img: Image.Image, *, face_crop: bool = True, stylized: bool = False) -> torch.Tensor:
    """Convert a PIL image to a (1, 1, 48, 48) tensor matching FER2013 normalisation.

    stylized=True applies extra steps that help on NPR/cel-shaded characters:
      - histogram equalisation (boosts shading detail flattened by toon shaders)
      - mild gaussian blur to reduce hard line-art aliasing
    """
    if face_crop:
        img = _try_face_crop(img)
    img = img.convert("L")
    if stylized:
        img = ImageOps.equalize(img)
        try:
            from PIL import ImageFilter
            img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
        except ImportError:
            pass
    tfm = build_transforms(train=False, augment=False)
    return tfm(img).unsqueeze(0)


@torch.no_grad()
def predict_image(checkpoint_path: str, img: Image.Image, *, device: Optional[str] = None,
                  face_crop: bool = True, stylized: bool = False) -> dict:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = build_model(
        ckpt.get("model_name", cfg.get("model_name", "DeeperCNN")),
        activation=cfg.get("activation", "relu"),
        dropout=cfg.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x = preprocess_image(img, face_crop=face_crop, stylized=stylized).to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(probs.argmax())
    return {
        "label": EMOTION_LABELS[idx],
        "label_idx": idx,
        "confidence": float(probs[idx]),
        "probabilities": {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(EMOTION_LABELS))},
    }
