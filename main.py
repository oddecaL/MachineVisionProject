from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.adaptive_selector import select_best_candidate
from src.degradation_analysis import analyze_degradation
from src.enhancement import build_candidate_pipelines
from src.image_loader import load_image, save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BAR-ADAPT adaptive barcode restoration pipeline")
    parser.add_argument("--image", required=True, help="Path to an input barcode image")
    parser.add_argument("--ground-truth", default=None, help="Optional expected barcode text for validation")
    parser.add_argument("--output-dir", default="results", help="Folder where output images and summaries are saved")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = load_image(args.image)
    metrics = analyze_degradation(image)
    candidates = build_candidate_pipelines(image)
    best = select_best_candidate(candidates, ground_truth=args.ground_truth)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_image(output_dir / "best_enhancement.png", best["image"])

    summary = {
        "degradation_metrics": metrics,
        "selected_pipeline": best["name"],
        "description": best["description"],
        "decoded_results": best["decoded"]["results"],
        "readability_score": best["readability"]["score"],
        "processing_success": best["decoded"]["success"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
