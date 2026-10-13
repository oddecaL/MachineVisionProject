# BAR-ADAPT

BAR-ADAPT is a degradation-aware image enhancement system for improving barcode readability in difficult industrial imaging conditions. The project combines image-quality analysis, candidate enhancement pipelines, barcode decoding, and a Streamlit dashboard for local inspection.

## Features

- Image ingestion for PNG, JPG, JPEG, and BMP files
- Degradation analysis: brightness, contrast, blur, noise, glare, surface-curvature cues, orientation, and approximate barcode size
- Multiple candidate enhancement pipelines, including CLAHE, gamma correction, denoising, sharpening, thresholding, perspective normalization, glare suppression, inpainting, and curved-surface correction
- Barcode decoding with `pyzbar`
- Reconstruction-oriented recovery for small missing or occluded barcode structures where local information remains available
- Readability scoring based on interpretable image and decoding metrics
- Reproducible synthetic evaluation pipeline and CSV export
- Local dashboard with Streamlit for inspection and download

## Project structure

```text
BAR-ADAPT/
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── config.py
├── images/
├── results/
├── src/
│   ├── __init__.py
│   ├── image_loader.py
│   ├── degradation_analysis.py
│   ├── enhancement.py
│   ├── barcode_decoder.py
│   ├── adaptive_selector.py
│   ├── readability_metrics.py
│   └── evaluation.py
├── tests/
│   └── test_pipeline.py
├── data/
└── results/
```

## Getting started

1. Create a Python 3.11+ environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. Or run a CLI analysis:

```bash
python main.py --image images/example.png
```

## Research framing and defensible novelty

The project does not claim a completely new barcode decoder or a revolutionary image-restoration algorithm. Instead, it adopts a realistic, defensible research position: a degradation-aware selection framework that chooses among established enhancement strategies based on barcode-specific evidence rather than image aesthetics alone.

Relevant existing work and tools include:

- ZXing, a mature and widely used barcode decoding library: https://github.com/zxing/zxing
- zxing-cpp, a modern C++/cross-platform decoder often used as a high-performance alternative: https://github.com/zxing-cpp/zxing-cpp
- pyzbar, a Python wrapper for barcode decoding that is practical for local prototype work: https://github.com/NaturalHistoryMuseum/pyzbar
- OpenCV image-processing pipelines, including CLAHE, thresholding, perspective transforms, denoising, and blur analysis: https://docs.opencv.org/
- Many industrial barcode readers from vendors such as Cognex, Zebra, and Honeywell already include built-in restoration, glare handling, and geometric correction, but their internal logic is proprietary and not directly reproducible in an academic project.

A realistic limitation that can be investigated is this: when a barcode image is degraded by uneven illumination, blur, glare, or compression-like artifacts, a single enhancement step does not reliably maximize decoding success. Existing decoders can read many clean or moderately degraded images, but the best enhancement branch depends on the degradation type and the barcode geometry. This makes a degradation-aware selector and barcode-specific validation pipeline a valid, testable experimental contribution.

This project’s narrow, defensible contribution is therefore not a claim that it invents barcode restoration from scratch. Instead, it provides a transparent, measurable framework to:

- estimate dominant degradation conditions;
- generate multiple enhancement candidates;
- select the best branch using decoder success and barcode-specific readability metrics;
- compare the chosen branch against the original and individual operations; and
- evaluate the effect through reproducible synthetic degradation experiments.

This is suitable for a third-year project because it is grounded in standard computer-vision tools, uses measurable outcomes, and is explicitly framed as a practical pipeline-selection contribution rather than a pseudo-novel filter.

## Notes

- This project is designed for offline use on a local machine.
- The enhancement and selection modules are intentionally modular so they can be tested independently.
- Barcode decoding may fail when a barcode is severely degraded, saturated, or missing information.
