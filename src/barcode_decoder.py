from __future__ import annotations

import cv2
import numpy as np
from pyzbar.pyzbar import decode


def decode_barcode(image: np.ndarray) -> dict:
    """Decode the best barcode found in the supplied image."""
    if image is None or image.size == 0:
        return {"success": False, "results": [], "error": "Empty image"}

    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        decoded = decode(rgb)
    except Exception as exc:  # pragma: no cover - runtime guard
        return {"success": False, "results": [], "error": str(exc)}

    results = []
    for item in decoded:
        data = item.data.decode("utf-8", errors="replace")
        results.append(
            {
                "type": item.type,
                "data": data,
                "quality": getattr(item, "quality", None),
                "bbox": [
                    getattr(item, "rect", None).left if getattr(item, "rect", None) else 0,
                    getattr(item, "rect", None).top if getattr(item, "rect", None) else 0,
                    getattr(item, "rect", None).width if getattr(item, "rect", None) else 0,
                    getattr(item, "rect", None).height if getattr(item, "rect", None) else 0,
                ],
            }
        )

    return {
        "success": bool(results),
        "results": results,
        "error": None if results else "No barcode decoded",
    }


def compare_decode_states(original_image: np.ndarray, enhanced_image: np.ndarray) -> dict:
    original = decode_barcode(original_image)
    enhanced = decode_barcode(enhanced_image)
    original_data = original["results"][0]["data"] if original["results"] else None
    enhanced_data = enhanced["results"][0]["data"] if enhanced["results"] else None

    return {
        "original": original,
        "enhanced": enhanced,
        "original_data": original_data,
        "enhanced_data": enhanced_data,
        "updated_decoding": enhanced["success"],
    }
