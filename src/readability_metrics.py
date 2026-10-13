from __future__ import annotations

import cv2
import numpy as np

from src.degradation_analysis import estimate_noise
from src.image_loader import as_grayscale


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def compute_readability_metrics(image: np.ndarray, decoded_results: list | None = None) -> dict:
    gray = as_grayscale(image)
    contrast = float(np.std(gray)) / 255.0
    laplacian = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    edge_strength = float(np.std(laplacian)) / 255.0

    # Foreground/background contrast and edge continuity are normalized to 0..1.
    contrast_norm = min(1.0, contrast * 1.8)
    edge_norm = min(1.0, edge_strength * 1.4)

    # Estimate module width continuity by looking at row-wise transitions over the barcode region.
    transitions = np.abs(np.diff(gray, axis=1)).mean() / 255.0
    module_norm = min(1.0, transitions * 3.0)

    noise_level = estimate_noise(gray) / 255.0
    noise_norm = max(0.0, 1.0 - min(1.0, noise_level * 2.5))

    illumination = 1.0 - min(1.0, abs(float(np.std(gray)) / 255.0))
    illumination_norm = max(0.0, illumination)

    decoder_bonus = 1.0 if decoded_results else 0.0

    score = (
        0.30 * contrast_norm
        + 0.20 * edge_norm
        + 0.15 * module_norm
        + 0.15 * noise_norm
        + 0.10 * illumination_norm
        + 0.10 * decoder_bonus
    ) * 100.0

    return {
        "score": round(float(score), 2),
        "foreground_background_contrast": round(float(contrast_norm * 100.0), 2),
        "edge_continuity": round(float(edge_norm * 100.0), 2),
        "module_width_estimate": round(float(module_norm * 100.0), 2),
        "noise_metric": round(float(noise_norm * 100.0), 2),
        "illumination_uniformity": round(float(illumination_norm * 100.0), 2),
        "decoder_bonus": decoder_bonus,
    }
