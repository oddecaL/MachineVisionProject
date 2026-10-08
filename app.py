from __future__ import annotations

import csv
import hashlib
import io
import time
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
import zxingcpp
from PIL import Image, ImageOps


st.set_page_config(page_title="LumaScan · Machine Vision", page_icon="◉", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--bg:#0b1011;--panel:#11191a;--line:#243032;--muted:#8a9a98;--mint:#b8f36d;--ink:#eff5ed}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--ink)}.stApp{background:radial-gradient(ellipse at 75% 0%,#162324 0%,#0b1011 50%)}
header[data-testid="stHeader"]{background:transparent}.block-container{padding-top:2.4rem;max-width:1440px}
section[data-testid="stSidebar"]{background:#0e1516;border-right:1px solid #202b2c}.section-label{color:var(--mint);font:500 10px 'DM Mono';letter-spacing:.16em;text-transform:uppercase;margin:2px 0 12px}
.hero{display:flex;justify-content:space-between;align-items:end;border-bottom:1px solid var(--line);padding-bottom:22px;margin-bottom:24px}.eyebrow{color:var(--mint);font:500 10px 'DM Mono';letter-spacing:.17em;text-transform:uppercase}.hero h1{font:700 clamp(30px,4vw,48px) 'Manrope';letter-spacing:-.055em;margin:7px 0 4px}.hero p{color:var(--muted);margin:0;font-size:14px}.live-pill{border:1px solid #38503a;border-radius:99px;padding:8px 12px;color:var(--mint);font:11px 'DM Mono';white-space:nowrap}
.panel{background:linear-gradient(145deg,#131c1d,#101718);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin-bottom:14px}.metric{font:700 28px 'Manrope';letter-spacing:-.04em}.metric-label{font:10px 'DM Mono';text-transform:uppercase;letter-spacing:.12em;color:var(--muted);margin-top:4px}.metric-note{color:#b7c7bc;font-size:12px;margin-top:7px}
.result-row{padding:12px 0;border-bottom:1px solid #263132}.result-row:last-child{border:0}.code{font:12px 'DM Mono';overflow-wrap:anywhere;color:#eff5ed}.subtle{font-size:11px;color:var(--muted);margin-top:5px}
div[data-testid="stButton"] button{border-radius:8px;font-weight:600;min-height:42px}div[data-testid="stButton"] button[kind="primary"]{background:var(--mint);color:#0b1011;border:0}div[data-testid="stButton"] button[kind="primary"]:hover{background:#ceff8a;color:#0b1011}
.stTabs [data-baseweb="tab-list"]{gap:8px;border-bottom:1px solid var(--line)}.stTabs [data-baseweb="tab"]{font-size:13px}.stTabs [aria-selected="true"]{color:var(--mint)!important}
div[data-testid="stFileUploader"]{background:#11191a;border:1px dashed #3b4b4a;border-radius:12px;padding:12px}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def variants(image: np.ndarray, strength: int, mode: str):
    """Return multiple progressively enhanced views for difficult lighting."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5 + strength * .065, tileGridSize=(8, 8))
    eq = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    # Lift shadows with a gamma curve while preserving highlights.
    gamma = 1.65 - (strength * .055)
    lut = np.array([min(255, round((i / 255.0) ** (1 / gamma) * 255)) for i in range(256)], dtype=np.uint8)
    lifted = cv2.LUT(eq, lut)
    denoised = cv2.fastNlMeansDenoisingColored(lifted, None, 4 + strength // 3, 4 + strength // 3, 7, 21)
    sharp = cv2.addWeighted(denoised, 1.45, cv2.GaussianBlur(denoised, (0, 0), 2), -.45, 0)
    gray_eq = cv2.cvtColor(lifted, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
    # Gradient morphology from the reference project is retained as a localization aid.
    gx = cv2.Sobel(gray_eq, cv2.CV_32F, 1, 0, ksize=-1)
    gy = cv2.Sobel(gray_eq, cv2.CV_32F, 0, 1, ksize=-1)
    grad = cv2.convertScaleAbs(cv2.subtract(gx, gy))
    _, mask = cv2.threshold(cv2.blur(grad, (3, 3)), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    if mode == "Fast scan":
        return [("Enhanced", sharp), ("Original", image)]
    if mode == "Balanced":
        return [("Enhanced", sharp), ("CLAHE", lifted), ("Original", image)]
    return [("Enhanced", sharp), ("CLAHE", lifted), ("Adaptive threshold", adaptive), ("Gradient localization", mask), ("Original", image)]


def decode_image(image: np.ndarray, strength: int, mode: str):
    start = time.perf_counter()
    views = variants(image, strength, mode)
    found = {}
    for label, view in views:
        try:
            results = zxingcpp.read_barcodes(view, try_rotate=True, try_downscale=True, try_invert=True)
        except TypeError:  # Compatibility with earlier zxing-cpp bindings
            results = zxingcpp.read_barcodes(view)
        for r in results:
            if not r.valid:
                continue
            key = (str(r.format), r.text)
            if key not in found:
                pos = r.position
                points = [(int(pos.top_left.x), int(pos.top_left.y)), (int(pos.top_right.x), int(pos.top_right.y)),
                          (int(pos.bottom_right.x), int(pos.bottom_right.y)), (int(pos.bottom_left.x), int(pos.bottom_left.y))]
                found[key] = {"format": str(r.format), "text": r.text, "bytes": bytes(r.bytes).hex(), "points": points, "method": label}
    return list(found.values()), (time.perf_counter() - start) * 1000, views


def annotated(image: np.ndarray, results: list[dict]) -> np.ndarray:
    out = image.copy()
    for idx, r in enumerate(results, 1):
        pts = np.array(r["points"], dtype=np.int32)
        cv2.polylines(out, [pts], True, (109, 238, 130), 3, cv2.LINE_AA)
        x, y = pts[:, 0].min(), pts[:, 1].min()
        cv2.circle(out, (int(x), int(y)), 7, (109, 238, 130), -1, cv2.LINE_AA)
        cv2.putText(out, f"{idx} · {r['format']}", (int(x), max(22, int(y) - 10)), cv2.FONT_HERSHEY_SIMPLEX, .65, (109, 238, 130), 2, cv2.LINE_AA)
    return out


def image_bytes(image: np.ndarray, ext=".png") -> bytes:
    ok, data = cv2.imencode(ext, image)
    return data.tobytes() if ok else b""


if "last_scan" not in st.session_state:
    st.session_state.last_scan = None

with st.sidebar:
    st.markdown("<div style='font:800 15px Manrope;letter-spacing:-.03em;margin:7px 0 28px'>◉ &nbsp; LUMASCAN <span style='font:10px DM Mono;color:#8a9a98'>/ MV-01</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Capture source</div>", unsafe_allow_html=True)
    source = st.radio("Capture source", ["Upload image", "Camera snapshot"], label_visibility="collapsed")
    st.markdown("<div class='section-label' style='margin-top:24px'>Enhancement pipeline</div>", unsafe_allow_html=True)
    strength = st.slider("Shadow recovery", 1, 10, 6, help="Higher values lift darker regions more strongly.")
    mode = st.select_slider("Scan depth", options=["Fast scan", "Balanced", "Deep scan"], value="Balanced", help="Deep scan tries additional threshold and gradient views.")
    formats = st.multiselect("Barcode formats", ["All formats"], default=["All formats"], disabled=True, help="ZXing-C++ automatically detects supported formats.")
    st.markdown("<div class='panel' style='padding:13px;margin-top:26px'><div class='section-label'>Privacy</div><div style='font-size:12px;color:#b7c7bc;line-height:1.6'>Images stay in this local session. Nothing is uploaded to a server.</div></div>", unsafe_allow_html=True)

st.markdown("<div class='hero'><div><div class='eyebrow'>Machine vision · low-light barcode recovery</div><h1>See clearly.<br>Scan confidently.</h1><p>Enhance shadows, decode barcodes, and inspect every result.</p></div><div class='live-pill'>● &nbsp; LOCAL PROCESSING</div></div>", unsafe_allow_html=True)

if source == "Upload image":
    upload = st.file_uploader("Drop an image or browse files", type=["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"], label_visibility="collapsed")
    pil = Image.open(upload) if upload else None
else:
    camera = st.camera_input("Take a scanner snapshot")
    pil = Image.open(camera) if camera else None

if pil:
    frame = to_bgr(pil)
    h, w = frame.shape[:2]
    max_side = 2400
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    frame_id = hashlib.sha256(frame.tobytes()).hexdigest()
    if st.session_state.last_scan and st.session_state.last_scan.get("frame_id") != frame_id:
        st.session_state.last_scan = None
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.markdown("<div class='section-label'>Input frame</div>", unsafe_allow_html=True)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption(f"{w:,} × {h:,} px  ·  {len(pil.getbands())}-channel image")
    with right:
        st.markdown("<div class='section-label'>Scanner controls</div>", unsafe_allow_html=True)
        run = st.button("◉  Enhance & scan", type="primary", use_container_width=True)
        if run:
            with st.spinner("Recovering shadow detail and checking barcode views…"):
                results, elapsed, _ = decode_image(frame, strength, mode)
                record = {"time": datetime.now().astimezone().isoformat(timespec="seconds"), "results": results,
                          "elapsed_ms": round(elapsed, 1), "width": int(frame.shape[1]), "height": int(frame.shape[0]),
                          "image": annotated(frame, results), "frame_id": frame_id}
                st.session_state.last_scan = record
        scan = st.session_state.last_scan
        if scan:
            a, b = st.columns(2)
            with a:
                st.markdown(f"<div class='panel'><div class='metric'>{len(scan['results']):02d}</div><div class='metric-label'>Barcodes found</div><div class='metric-note'>{'Decode complete' if scan['results'] else 'Try deep scan or adjust shadow recovery'}</div></div>", unsafe_allow_html=True)
            with b:
                st.markdown(f"<div class='panel'><div class='metric'>{scan['elapsed_ms']:.0f}<span style='font-size:14px;color:#8a9a98'> ms</span></div><div class='metric-label'>Processing time</div><div class='metric-note'>{mode} · {strength}/10 recovery</div></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Decoded results</div>", unsafe_allow_html=True)
            if scan["results"]:
                for i, r in enumerate(scan["results"], 1):
                    st.markdown(f"<div class='section-label'>#{i} &nbsp; {r['format']} <span style='color:#8a9a98'>· via {r['method']}</span></div>", unsafe_allow_html=True)
                    st.code(r["text"], language=None, wrap_lines=True)
            else:
                st.info("No barcode decoded in this frame. Try **Deep scan**, raise shadow recovery, reduce glare, or move closer to the barcode.")
    if st.session_state.last_scan:
        st.markdown("<div class='section-label'>Detection overlay</div>", unsafe_allow_html=True)
        st.image(cv2.cvtColor(st.session_state.last_scan["image"], cv2.COLOR_BGR2RGB), use_container_width=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.download_button("Download annotated image", image_bytes(st.session_state.last_scan["image"]), "lumascan-annotated.png", "image/png", use_container_width=True)
        with c2:
            csv_buffer = io.StringIO()
            writer = csv.DictWriter(csv_buffer, fieldnames=["time", "format", "text", "method"])
            writer.writeheader()
            for r in st.session_state.last_scan["results"]:
                writer.writerow({"time": st.session_state.last_scan["time"], "format": r["format"], "text": r["text"], "method": r["method"]})
            st.download_button("Export results · CSV", csv_buffer.getvalue(), "lumascan-results.csv", "text/csv", use_container_width=True)
else:
    st.markdown("<div class='panel' style='text-align:center;padding:54px 20px;margin-top:20px'><div style='font:48px Manrope;color:#b8f36d'>◉</div><div style='font:600 18px Manrope;margin:12px 0 6px'>Ready when you are</div><div style='color:#8a9a98;font-size:13px'>Upload a barcode image or switch to camera snapshot to begin.</div></div>", unsafe_allow_html=True)

st.markdown("<div style='border-top:1px solid #243032;margin-top:30px;padding:16px 0;color:#71817f;font:10px DM Mono;display:flex;justify-content:space-between'><span>LUMASCAN · MACHINE VISION SCANNER</span><span>OpenCV enhancement + ZXing-C++ decoding</span></div>", unsafe_allow_html=True)
