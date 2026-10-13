from __future__ import annotations

import cv2
import numpy as np

from config import DEFAULT_CLAHE_CLIP, DEFAULT_GAMMA


def detect_glare_mask(image: np.ndarray, threshold: int = 220) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    mask = (value > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def remove_glare(image: np.ndarray, threshold: int = 220) -> np.ndarray:
    glare_mask = detect_glare_mask(image, threshold)
    if np.count_nonzero(glare_mask) == 0:
        return image.copy()
    corrected = image.copy()
    corrected = cv2.inpaint(corrected, glare_mask, 3, cv2.INPAINT_TELEA)
    return corrected


def inpaint_missing_regions(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    low_contrast = np.abs(gray.astype(np.int16) - cv2.GaussianBlur(gray, (21, 21), 0)) > threshold
    mask = (low_contrast * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if np.count_nonzero(mask) == 0:
        return image.copy()
    return cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)


def estimate_curvature_mask(image: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(gray), 0.0
    largest = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest, True)
    area = cv2.contourArea(largest)
    if perimeter == 0:
        return np.zeros_like(gray), 0.0
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    score = float(max(0.0, 1.0 - circularity))
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest], -1, 255, thickness=2)
    return mask, score


def correct_curved_surface(image: np.ndarray) -> np.ndarray:
    mask, score = estimate_curvature_mask(image)
    if score < 0.15:
        return image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image.copy()
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 4:
        return image.copy()
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.float32)
    width = int(rect[1][0])
    height = int(rect[1][1])
    if width <= 0 or height <= 0:
        return image.copy()
    target = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(box, target)
    return cv2.warpPerspective(image, transform, (width, height))


def apply_clahe(image: np.ndarray, clip_limit: float = DEFAULT_CLAHE_CLIP, grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=grid_size)
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def apply_gamma(image: np.ndarray, gamma: float = DEFAULT_GAMMA) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


def apply_bilateral_denoise(image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_median_denoise(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(image, ksize)


def apply_sharpen(image: np.ndarray) -> np.ndarray:
    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1],
    ], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        10,
    )
    return cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)


def otsu_threshold(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)


def perspective_normalize(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image.copy()

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 500:
        return image.copy()

    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    if len(approx) != 4:
        return image.copy()

    corners = approx.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]
    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]

    width = max(np.linalg.norm(rect[1] - rect[2]), np.linalg.norm(rect[0] - rect[3]))
    height = max(np.linalg.norm(rect[2] - rect[3]), np.linalg.norm(rect[1] - rect[0]))
    target = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float32)

    transform = cv2.getPerspectiveTransform(rect, target)
    warped = cv2.warpPerspective(image, transform, (int(width), int(height)))
    return warped


def build_candidate_pipelines(image: np.ndarray) -> list[dict]:
    candidates: list[dict] = []

    candidates.append({
        "name": "Original",
        "description": "No enhancement. Baseline for comparison.",
        "image": image.copy(),
        "utility": 0.0,
    })
    candidates.append({
        "name": "CLAHE",
        "description": "Local contrast equalization to recover dim barcode contrast.",
        "image": apply_clahe(image),
        "utility": 0.0,
    })
    candidates.append({
        "name": "Gamma + CLAHE",
        "description": "Gamma correction with local contrast for dark or underexposed images.",
        "image": apply_clahe(apply_gamma(image, gamma=DEFAULT_GAMMA)),
        "utility": 0.0,
    })
    candidates.append({
        "name": "Bilateral Denoise + CLAHE",
        "description": "Reduce noise while preserving barcode edge structure.",
        "image": apply_clahe(apply_bilateral_denoise(image, d=9, sigma_color=75, sigma_space=75)),
        "utility": 0.0,
    })
    candidates.append({
        "name": "Sharpen + Threshold",
        "description": "Enhance weak edges and threshold for decoding-friendly binary output.",
        "image": otsu_threshold(apply_sharpen(image)),
        "utility": 0.0,
    })
    candidates.append({
        "name": "Perspective Normalized",
        "description": "Perspective correction when a barcode contour can be localized.",
        "image": perspective_normalize(image),
        "utility": 0.0,
    })
    candidates.append({
        "name": "Glare Reduction + Inpaint",
        "description": "Detect bright reflective glare and reconstruct the occluded barcode region using local interpolation.",
        "image": remove_glare(inpaint_missing_regions(image)),
        "utility": 0.0,
    })
    candidates.append({
        "name": "Curved Surface Correction + CLAHE",
        "description": "Estimate surface curvature and rectify the barcode region before local contrast enhancement.",
        "image": apply_clahe(correct_curved_surface(image)),
        "utility": 0.0,
    })
    candidates.append({
        "name": "Glare + Curvature + CLAHE",
        "description": "Combine glare suppression, curved-surface correction, and local contrast enhancement for difficult reflective packaging cases.",
        "image": apply_clahe(remove_glare(correct_curved_surface(image))),
        "utility": 0.0,
    })

    return candidates
