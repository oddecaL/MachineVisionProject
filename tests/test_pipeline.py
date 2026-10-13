from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import qrcode

from src.barcode_decoder import decode_barcode
from src.degradation_analysis import analyze_degradation
from src.enhancement import build_candidate_pipelines


def create_test_barcode(path: Path = Path("tests/test_barcode.png")) -> Path:
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data("BAR-ADAPT-TEST-123")
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white")
    rgb = np.array(image.convert("RGB"))[:, :, ::-1]
    cv2.imwrite(str(path), rgb)
    return path


def test_decode_pipeline_returns_valid_result():
    path = create_test_barcode()
    decoded = decode_barcode(cv2.imread(str(path), cv2.IMREAD_COLOR))
    assert decoded["success"] is True
    assert decoded["results"]
    assert "BAR-ADAPT-TEST-123" in decoded["results"][0]["data"]


def test_degradation_analysis_returns_numeric_metrics():
    path = create_test_barcode()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    metrics = analyze_degradation(image)
    assert "mean_brightness" in metrics
    assert 0 <= metrics["mean_brightness"] <= 255
    assert "blur_score" in metrics
    assert "noise_estimate" in metrics


def test_candidate_pipeline_generation_has_multiple_candidates():
    path = create_test_barcode()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    pipelines = build_candidate_pipelines(image)
    assert len(pipelines) >= 4
    assert any(p["name"] == "Original" for p in pipelines)
    assert any(p["name"] == "CLAHE" for p in pipelines)
