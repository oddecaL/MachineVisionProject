from __future__ import annotations

import io
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from config import IMAGES_DIR
from src.adaptive_selector import select_best_candidate
from src.barcode_decoder import decode_barcode
from src.degradation_analysis import analyze_degradation
from src.enhancement import build_candidate_pipelines
from src.image_loader import load_image
from src.readability_metrics import compute_readability_metrics


st.set_page_config(page_title="BAR-ADAPT", page_icon="📷", layout="wide")


def pick_source_image(uploaded_file):
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return image

    default_dir = IMAGES_DIR
    sample_files = sorted(default_dir.glob("*.*")) if default_dir.exists() else []
    if not sample_files:
        return None

    selected = st.selectbox("Choose sample image", [f.name for f in sample_files])
    return load_image(default_dir / selected)


st.title("BAR-ADAPT — Adaptive Barcode Restoration and Readability Assessment")
st.caption("Degradation-aware enhancement and evaluation for industrial barcode inspection.")

with st.sidebar:
    st.header("Input")
    uploaded_file = st.file_uploader("Upload barcode image", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded_file is not None:
        st.success(f"Loaded: {uploaded_file.name}")

    st.header("Processing")
    analyze_clicked = st.button("Analyze Image", type="primary")

if not analyze_clicked:
    st.info("Upload an image or select a sample, then click 'Analyze Image'.")
    st.stop()

image = pick_source_image(uploaded_file)
if image is None:
    st.warning("No image found. Add an image file to the images folder or upload one.")
    st.stop()

start = time.perf_counter()
original_decoded = decode_barcode(image)
metrics_before = analyze_degradation(image)
readability_before = compute_readability_metrics(image, original_decoded["results"])
candidates = build_candidate_pipelines(image)
best = select_best_candidate(candidates, ground_truth=None)
processing_time = time.perf_counter() - start

st.subheader("Original vs enhanced")
left_col, right_col = st.columns(2)
with left_col:
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB", caption="Original image")
with right_col:
    enhanced_rgb = cv2.cvtColor(best["image"], cv2.COLOR_BGR2RGB)
    st.image(enhanced_rgb, channels="RGB", caption=f"Best candidate: {best['name']}")

st.subheader("Degradation analysis")
metrics_df = {
    "Metric": [
        "Mean brightness",
        "Global contrast",
        "Local contrast",
        "Blur score (Laplacian variance)",
        "Noise estimate",
        "Glare %",
        "Glare mask %",
        "Surface curvature score",
        "Surface type",
        "Orientation",
        "Approx. width",
        "Approx. height",
    ],
    "Value": [
        metrics_before["mean_brightness"],
        metrics_before["global_contrast"],
        metrics_before["local_contrast"],
        metrics_before["blur_score"],
        metrics_before["noise_estimate"],
        metrics_before["glare_percentage"],
        metrics_before.get("glare_mask_percentage", 0.0),
        metrics_before.get("curvature_score", 0.0),
        metrics_before.get("surface_type", "unknown"),
        metrics_before["orientation_deg"],
        metrics_before["approx_width_px"],
        metrics_before["approx_height_px"],
    ],
}
st.table(metrics_df)

if metrics_before.get("glare_mask_percentage", 0.0) > 15.0:
    st.warning("Strong reflective glare is present. The system attempts glare suppression and inpainting before decoding.")
if metrics_before.get("curvature_score", 0.0) > 0.15:
    st.warning("A curved or wrapped surface is likely present. The pipeline includes surface-rectification attempts.")

st.subheader("Candidate pipeline comparison")
comparison_rows = []
for candidate in candidates:
    decoded = decode_barcode(candidate["image"])
    readability = compute_readability_metrics(candidate["image"], decoded["results"])
    comparison_rows.append({
        "Pipeline": candidate["name"],
        "Decoded": decoded["success"],
        "Decoded value": decoded["results"][0]["data"] if decoded["results"] else "",
        "Readability score": readability["score"],
    })
st.dataframe(comparison_rows)

st.subheader("Decoding and readability")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Original decode", "Success" if original_decoded["success"] else "Fail")
with col2:
    st.metric("Enhanced decode", "Success" if best["decoded"]["success"] else "Fail")
with col3:
    st.metric("Processing time", f"{processing_time:.2f}s")

st.metric("Readability score before", round(readability_before["score"], 2))
st.metric("Readability score after", round(best["readability"]["score"], 2))

st.subheader("Selected pipeline")
st.write(best["name"])
st.write(best["description"])
if best["decoded"]["results"]:
    st.write(f"Decoded value: {best['decoded']['results'][0]['data']}")
else:
    st.warning("No valid barcode could be decoded. A re-capture or different imaging setup may be required. The remaining unreadable regions may be too damaged for reliable reconstruction.")

st.subheader("Operational recommendation")
if best["decoded"]["success"]:
    st.success("The barcode is readable after enhancement. The current image is suitable for inspection or downstream tracking.")
else:
    st.error("The image remains unreadable. Recommended actions: improve illumination, reduce glare with diffused lighting, capture the barcode more normal to the surface, and re-capture from a flatter or less reflective region.")

st.subheader("Download")
_, buffer = cv2.imencode(".png", best["image"])
st.download_button(
    label="Download enhanced image",
    data=buffer.tobytes(),
    file_name="bar_adapt_enhanced.png",
    mime="image/png",
)
