from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from config import ALLOWED_EXTENSIONS


def validate_image_path(path: str | Path) -> Path:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Image not found: {file_path}")
    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image type '{file_path.suffix}'. "
            f"Use one of: {sorted(ALLOWED_EXTENSIONS)}"
        )
    try:
        with Image.open(file_path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Corrupt or unreadable image: {file_path}") from exc
    return file_path


def load_image(path: str | Path) -> np.ndarray:
    file_path = validate_image_path(path)
    image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not open image file: {file_path}")
    return image


def save_image(path: str | Path, image: np.ndarray) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise IOError(f"Failed to save image to {output_path}")
    return output_path


def as_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def as_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
