from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.adaptive_selector import select_best_candidate
from src.enhancement import build_candidate_pipelines
from src.image_loader import load_image


def generate_synthetic_degradation(image: np.ndarray, kind: str, severity: float = 0.5) -> np.ndarray:
    output = image.copy().astype(np.float32)
    h, w = output.shape[:2]

    if kind == "darkness":
        output = output * (1.0 - severity * 0.5)
    elif kind == "uneven_illumination":
        gradient = np.linspace(0.5, 1.6, h, dtype=np.float32)[:, None]
        output = output * gradient
    elif kind == "gaussian_noise":
        noise = np.random.normal(0, severity * 35, output.shape)
        output = output + noise
    elif kind == "salt_and_pepper":
        mask = np.random.rand(h, w, 3) < severity * 0.12
        output[mask] = 0 if np.random.rand() > 0.5 else 255
    elif kind == "gaussian_blur":
        kernel_size = max(3, int(3 + severity * 9))
        if kernel_size % 2 == 0:
            kernel_size += 1
        output = cv2.GaussianBlur(output, (kernel_size, kernel_size), 0)
    elif kind == "motion_blur":
        kernel_len = max(5, int(5 + severity * 18))
        kernel = np.zeros((kernel_len, kernel_len), np.float32)
        kernel[:, int(kernel_len / 2)] = 1.0 / kernel_len
        output = cv2.filter2D(output, -1, kernel)
    elif kind == "perspective_distortion":
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dest = np.float32([
            [w * 0.05, h * 0.12],
            [w * 0.92, h * 0.18],
            [w * 0.9, h * 0.92],
            [w * 0.06, h * 0.9],
        ])
        transform = cv2.getPerspectiveTransform(src, dest)
        output = cv2.warpPerspective(output, transform, (w, h))
    else:
        raise ValueError(f"Unsupported degradation kind: {kind}")

    output = np.clip(output, 0, 255).astype(np.uint8)
    return output


def run_synthetic_evaluation(image_path: str | Path, ground_truth: str, output_dir: str | Path = "results") -> list[dict]:
    image = load_image(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    degradation_types = [
        "darkness",
        "uneven_illumination",
        "gaussian_noise",
        "salt_and_pepper",
        "gaussian_blur",
        "motion_blur",
        "perspective_distortion",
    ]

    rows: list[dict] = []

    for kind in degradation_types:
        degraded = generate_synthetic_degradation(image, kind=kind, severity=0.5)
        candidates = build_candidate_pipelines(degraded)
        best = select_best_candidate(candidates, ground_truth=ground_truth)
        rows.append(
            {
                "degradation": kind,
                "best_pipeline": best["name"],
                "decoded": best["decoded"]["results"][0]["data"] if best["decoded"]["results"] else "",
                "success": bool(best["decoded"]["results"]),
                "readability_score": best["readability"]["score"],
                "ground_truth": ground_truth,
            }
        )

    csv_path = output_dir / "synthetic_eval_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "degradation",
            "best_pipeline",
            "decoded",
            "success",
            "readability_score",
            "ground_truth",
        ])
        writer.writeheader()
        writer.writerows(rows)

    # A simple success-rate bar chart for quick inspection.
    labels = [row["degradation"] for row in rows]
    success_values = [1 if row["success"] else 0 for row in rows]
    scores = [row["readability_score"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, success_values, color=["steelblue" if value else "tomato" for value in success_values])
    axes[0].set_title("Decoding success by degradation")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("Success (1 = yes)")

    axes[1].bar(labels, scores, color="darkorange")
    axes[1].set_title("Readability score by degradation")
    axes[1].set_ylabel("Score")

    for ax in axes:
        ax.tick_params(axis="x", rotation=30)

    plot_path = output_dir / "synthetic_eval_summary.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return rows
