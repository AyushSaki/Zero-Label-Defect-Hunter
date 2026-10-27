# Zero-Label Defect Hunter

**AI Manufacturing Quality Inspection — trained only on good products**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Zero-Label](https://img.shields.io/badge/Training-Good%20images%20only-orange)

---

## Project Overview

**Zero-Label Defect Hunter** is an industrial anomaly-detection system that learns what a *normal* manufactured product looks like using **only good product images**.  

When a new image is uploaded, the system:

1. Extracts deep visual features  
2. Compares them against a memory bank of normal patches  
3. Produces an **anomaly score**  
4. Generates an **anomaly heatmap** and overlay  
5. Returns a clear result: **NORMAL** or **SUSPICIOUS**

No defective images are required (or used) during training.

---

## Problem Statement

In real factories, collecting large numbers of labelled defective samples is expensive, slow, and sometimes impossible (rare defects).  

Zero-label / unsupervised anomaly detection solves this by training exclusively on the abundant “good” products and flagging anything that deviates from the learned normal distribution.

---

## How Zero-Label Anomaly Detection Works

1. **Feature extraction** – A pretrained Wide-ResNet50 extracts mid-level patch features from every good training image.  
2. **Memory bank** – All normal patch features are stored (with optional random projection + coreset reduction).  
3. **Nearest-neighbour scoring** – At inference time each patch of a new image is scored by its distance to the nearest normal patches.  
4. **Image-level score** – The maximum patch score becomes the anomaly score.  
5. **Threshold** – Calibrated on a held-out set of good images (e.g. 90th percentile).  
6. **Heatmap** – The spatial score map is upsampled and colour-coded for localisation.

This is a practical, simplified PatchCore-style pipeline that runs fully offline.

---

## Architecture

```
GOOD PRODUCT IMAGES
        │
        ▼
 Jupyter Notebook / CLI
        │
        ▼
 Anomaly Detection Model  (Wide-ResNet50 + Memory Bank)
        │
        ▼
   Saved Model (.pkl)
        │
        ▼
   Flask Application
        │
  Upload new product
        │
        ▼
 Image Processing → Anomaly Score + Heatmap
        │
        ▼
 Inspection Result (NORMAL / SUSPICIOUS)
```

---

## Technology Stack

| Layer              | Tools                                      |
|--------------------|--------------------------------------------|
| Language           | Python 3.10 – 3.12                         |
| Deep Learning      | PyTorch, torchvision                       |
| Classical ML       | scikit-learn (NearestNeighbors, projection)|
| Image processing   | OpenCV, Pillow, NumPy                      |
| Web                | Flask, Bootstrap 5, HTML/CSS               |
| Notebook           | JupyterLab / Jupyter Notebook              |
| Packaging          | pip / conda                                |

---

## Dataset

The repository ships with a **synthetic bottle dataset** so you can run the demo immediately.

```
data/
├── train/good/          # 40 good bottles (training)
├── validation/good/     # 10 good bottles (threshold calibration)
└── test/
    ├── good/            # 12 good bottles
    └── defective/       # 15 synthetic defects (scratch, dent, stain, crack, spot)
```

> **Important**  
> Defective images are **never** used for training.  
> They exist only for evaluation and demo purposes.  
> The dataset is clearly synthetic — replace it with MVTec AD (bottle category) or your own factory images for serious experiments.

### Recommended sizes for real projects

| Purpose        | Good images | Defective (test only) |
|----------------|-------------|------------------------|
| Quick demo     | 20–50       | 10–20                  |
| Solid prototype| 100–300     | 50                     |
| Strong experiment | 500–2000+ | 100+                |

---

## Installation

### Option A – pip (recommended for most users)

```bash
# Clone or unzip the project
cd Zero-Label-Defect-Hunter

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# CPU-only (fastest install on laptops)
pip install -r requirements-cpu.txt

# OR default (will pull CUDA wheels if available)
pip install -r requirements.txt

# OR explicit GPU (CUDA 12.1)
pip install -r requirements-gpu.txt

# Optional: notebook / dev tools
pip install -r requirements-dev.txt
```

### Option B – Conda / Anaconda

```bash
conda env create -f environment.yml
conda activate defect-hunter
```

---

## Model Training

```bash
# From project root
python defect_hunter.py --data data --out models/defect_hunter_model.pkl
```

Or open the notebook:

```bash
jupyter lab notebooks/defect_detection.ipynb
# or
jupyter notebook notebooks/defect_detection.ipynb
```

The notebook walks through all ten required sections (imports → evaluation).

---

## Flask Application

```bash
python app.py
```

Then open **http://127.0.0.1:5000**

- Upload any product image  
- Click **INSPECT**  
- View original / heatmap / overlay + anomaly score + NORMAL / SUSPICIOUS  
- All inspections are logged to `results/inspection_history.csv`

---

## Project Structure

```
Zero-Label-Defect-Hunter/
├── app.py                      # Flask entry point
├── defect_hunter.py            # Core anomaly-detection logic
├── requirements.txt            # Main dependencies
├── requirements-cpu.txt        # CPU-only PyTorch
├── requirements-gpu.txt        # CUDA 12.1 PyTorch
├── requirements-dev.txt        # Jupyter + linting extras
├── environment.yml             # Conda environment
├── README.md
├── LICENSE
├── .gitignore
│
├── notebooks/
│   └── defect_detection.ipynb
│
├── data/
│   ├── train/good/
│   ├── validation/good/
│   └── test/{good,defective}/
│
├── models/                     # Saved memory-bank model
├── uploads/                    # Runtime uploads
├── results/
│   ├── inspection_history.csv
│   └── predictions/
│
├── templates/
│   ├── index.html
│   └── history.html
│
└── static/
    ├── css/style.css
    └── results/                # Generated heatmaps
```

---

## Heatmap Explanation

| Image    | Meaning                                              |
|----------|------------------------------------------------------|
| Original | Uploaded product                                     |
| Heatmap  | Warm colours = regions far from the normal memory bank |
| Overlay  | Heatmap blended on the original for easy localisation |

---

## Results & Evaluation

After training you can evaluate on the held-out test set (good + defective).  
Typical metrics computed in the notebook:

- Accuracy, Precision, Recall, F1  
- ROC-AUC (image-level)  
- Qualitative heatmap inspection

Because the sample dataset is synthetic and small, treat the numbers as a **sanity check**, not a production benchmark. Swap in MVTec AD for proper numbers.

---

## Limitations

- Current implementation is a lightweight PatchCore-style baseline (not the full original paper with greedy coreset).  
- Sensitive to large changes in lighting, camera angle or product pose.  
- Single product category in the demo (bottles).  
- Heatmap resolution is limited by the backbone feature-map size.  
- Not a certified industrial inspection system — always keep a human in the loop.

---

## Future Improvements

- Full PatchCore coreset + locally-aware patch scoring  
- Support for multiple product categories  
- Pixel-level AUROC when ground-truth masks are available (MVTec)  
- Real-time webcam / line-camera mode  
- Model quantisation & ONNX export for edge devices  
- User authentication & multi-user history  
- Active learning loop for hard negatives

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Acknowledgements

- MVTec AD dataset (for serious benchmarking)  
- PatchCore paper (Roth et al., CVPR 2022)  
- PyTorch & torchvision maintainers

---

## Using the official MVTec AD Bottle dataset

1. Download **Bottle (148 MB)** from:
   https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads

2. Place `bottle.zip` next to this project (or anywhere).

3. Integrate + optionally train:

```bash
# Activate your venv first
source .venv/bin/activate

# Integrate dataset only
python integrate_mvtec_bottle.py ~/Downloads/bottle.zip

# Or integrate AND train in one step
python integrate_mvtec_bottle.py ~/Downloads/bottle.zip --train
```

4. Start the app:

```bash
python app.py
# If port 5000 is busy (AirPlay on macOS):
python -c "from app import app, get_hunter; get_hunter(); app.run(host='0.0.0.0', port=5001, debug=True)"
```

Open http://127.0.0.1:5001

MVTec bottle structure used:

```
bottle/
├── train/good/          → data/train/good/     (training ONLY)
├── test/good/           → split into validation + test/good
└── test/broken_* / contamination/  → data/test/defective/
```

License: CC BY-NC-SA 4.0 (non-commercial / research / portfolio OK).
