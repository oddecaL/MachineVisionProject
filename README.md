# LumaScan — Machine Vision Low Light Scanner

A local browser-based barcode scanner for challenging lighting. LumaScan combines low-light image enhancement with multi-view barcode decoding and a visual inspection interface.

## Features

- Upload common image formats or capture a still frame from a webcam.
- Recover shadow detail with CLAHE, a configurable gamma curve, denoising, and sharpening.
- Decode using ZXing-C++, trying rotated, downscaled, and inverted views where supported.
- Choose Fast, Balanced, or Deep scan. Deep scan adds adaptive-threshold and Sobel-gradient morphology views based on the supplied OpenCV reference project.
- See barcode outlines, decoded text, format, processing time, and the view that decoded each symbol.
- Download an annotated PNG and export scan results as CSV.
- Runs locally; images are processed in the app process and are not sent to a hosted service.

## Requirements

- Python 3.10 or newer (64-bit recommended).
- A modern browser. Camera access requires the browser to allow access; localhost is treated as a secure context by modern browsers.

## Run on Windows

Open PowerShell in this folder and run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit opens the interface in your browser, usually at `http://localhost:8501`.

## Run on macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Use

1. Choose **Upload image** or **Camera snapshot** in the sidebar.
2. Set **Shadow recovery** higher for dark frames. Start with **Balanced**; select **Deep scan** for small, low-contrast, or difficult symbols.
3. Select **Enhance & scan**. Inspect the decoded payload and detection overlay.
4. Download the annotated image or CSV results as needed.

The camera option captures one still image; it does not continuously scan video. For best results, keep the barcode in focus, fill a reasonable portion of the frame, and avoid specular glare.

## How the vision pipeline works

The original project demonstrates horizontal-versus-vertical Sobel gradients, thresholding, morphology, and ZBar decoding. LumaScan keeps that gradient/morphology idea as one of its deeper search views. It adds a shadow-lifted CLAHE/gamma view, denoising and sharpening, adaptive thresholding, and a modern ZXing-C++ decoder. Multiple enhanced views are tried and duplicate format/payload pairs are collapsed into one result.

## Supported formats

ZXing-C++ supports common retail and machine-readable symbologies, including QR Code, Data Matrix, Aztec, PDF417, Code 128, Code 39, EAN/UPC, and ITF. Actual decode success depends on image quality, symbol size, focus, and damage.

## Project layout

```text
Machine Vision Low Light Scanner/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## License

This implementation is provided under the MIT License. ZXing-C++, OpenCV, Streamlit, NumPy, and Pillow retain their respective licenses.
