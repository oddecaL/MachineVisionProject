from __future__ import annotations

import cv2
import numpy as np

from src.image_loader import as_grayscale


def estimate_brightness(gray: np.ndarray) -> float:
    return float(np.mean(gray))


def estimate_global_contrast(gray: np.ndarray) -> float:
    return float(np.std(gray))


def estimate_local_contrast(gray: np.ndarray) -> float:
    laplacian = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    return float(np.std(laplacian))


def estimate_blur_score(gray: np.ndarray) -> float:
    laplacian = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    return float(np.var(laplacian))


def estimate_noise(gray: np.ndarray) -> float:
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    noise = gray.astype(np.float32) - blurred
    return float(np.std(noise))


def estimate_glare_percentage(gray: np.ndarray) -> float:
    bright_pixels = gray >= 245
    if bright_pixels.size == 0:
        return 0.0
    return float(np.mean(bright_pixels) * 100.0)


def estimate_glare_mask_percentage(image: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value_channel = hsv[:, :, 2]
    glare_mask = value_channel > 220
    if glare_mask.size == 0:
        return 0.0
    return float(np.mean(glare_mask) * 100.0)


def estimate_surface_curvature(image: np.ndarray) -> dict[str, float | str]:
    gray = as_grayscale(image)
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"curvature_score": 0.0, "surface_type": "unknown"}
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 250:
        return {"curvature_score": 0.0, "surface_type": "unknown"}
    perimeter = cv2.arcLength(largest, True)
    area = cv2.contourArea(largest)
    if perimeter == 0:
        return {"curvature_score": 0.0, "surface_type": "unknown"}
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    curvature_score = max(0.0, 1.0 - circularity)
    if curvature_score > 0.35:
        surface_type = "curved_surface"
    else:
        surface_type = "approximately_planar"
    return {"curvature_score": round(float(curvature_score), 4), "surface_type": surface_type}


def approximate_barcode_geometry(gray: np.ndarray) -> dict[str, float | int | None]:
    gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(gray_uint8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "orientation_deg": None,
            "width_px": None,
            "height_px": None,
            "bbox": None,
        }

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 40:
        return {
            "orientation_deg": None,
            "width_px": None,
            "height_px": None,
            "bbox": None,
        }

    x, y, w, h = cv2.boundingRect(largest)
    rect = cv2.minAreaRect(largest)
    angle = rect[-1]
    if angle < -45:
        angle += 90
    else:
        angle = angle

    return {
        "orientation_deg": float(angle),
        "width_px": int(w),
        "height_px": int(h),
        "bbox": (int(x), int(y), int(w), int(h)),
    }


def analyze_degradation(image: np.ndarray) -> dict:
    gray = as_grayscale(image)
    brightness = estimate_brightness(gray)
    global_contrast = estimate_global_contrast(gray)
    local_contrast = estimate_local_contrast(gray)
    blur_score = estimate_blur_score(gray)
    noise_estimate = estimate_noise(gray)
    glare_percentage = estimate_glare_percentage(gray)
    glare_mask_percentage = estimate_glare_mask_percentage(image)
    curvature = estimate_surface_curvature(image)
    geometry = approximate_barcode_geometry(gray)

    metrics = {
        "mean_brightness": float(round(brightness, 3)),
        "global_contrast": float(round(global_contrast, 3)),
        "local_contrast": float(round(local_contrast, 3)),
        "blur_score": float(round(blur_score, 3)),
        "noise_estimate": float(round(noise_estimate, 3)),
        "glare_percentage": float(round(glare_percentage, 3)),
        "glare_mask_percentage": float(round(glare_mask_percentage, 3)),
        "curvature_score": curvature["curvature_score"],
        "surface_type": curvature["surface_type"],
        "orientation_deg": geometry["orientation_deg"],
        "approx_width_px": geometry["width_px"],
        "approx_height_px": geometry["height_px"],
        "bbox": geometry["bbox"],
    }
    return metrics
