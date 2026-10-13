from __future__ import annotations

from src.barcode_decoder import decode_barcode
from src.readability_metrics import compute_readability_metrics


def score_candidate(candidate: dict, ground_truth: str | None = None) -> float:
    decoded = decode_barcode(candidate["image"])
    metrics = compute_readability_metrics(candidate["image"], decoded["results"])

    utility = 0.0
    if decoded["success"]:
        utility += 60.0
    if ground_truth is not None:
        matches = any(item["data"] == ground_truth for item in decoded["results"])
        if matches:
            utility += 25.0
    utility += metrics["score"] * 0.35

    candidate["decoded"] = decoded
    candidate["readability"] = metrics
    candidate["utility"] = utility
    return utility


def select_best_candidate(candidates: list[dict], ground_truth: str | None = None) -> dict:
    if not candidates:
        raise ValueError("No candidate enhancement pipelines were provided.")

    for candidate in candidates:
        score_candidate(candidate, ground_truth=ground_truth)

    return max(candidates, key=lambda item: item["utility"])
