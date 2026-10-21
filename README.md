# Noise-Resilient ECG Signal Processing & Arrhythmia Classification Platform

An educational research platform that combines classical digital signal processing with self-supervised deep learning for noise-robust ECG analysis on PhysioNet/MIT-BIH recordings.

The system performs three core tasks:

1. Cleans noisy ECG signals using tiered filter pipelines (bandpass, notch, wavelet)
2. Detects every heartbeat with a custom Pan-Tompkins-inspired QRS detector
3. Classifies each detected beat into AAMI 5-class arrhythmia categories (N, S, V, F, Q) using a 1D ResNet encoder pre-trained with contrastive learning on noise-augmented beats

**EDUCATIONAL PROTOTYPE — NOT A MEDICAL DEVICE — NOT FOR CLINICAL USE.** All outputs are indicative only and require human review. The system makes no diagnostic claims.

---

## Prerequisites

- **Python**: 3.12 or higher
- **OS**: Linux, macOS, or Windows (PowerShell commands shown; use the equivalent on your shell)
- **Disk**: ~2 GB free for data, model, and dependencies
- **GPU**: Not required for inference or the dashboard. Only needed if retraining the ML model (free Kaggle GPU is sufficient).

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/arushiagrawal-123/ECD_DSP123.git
cd ECD_DSP123
```

### 2. Create a virtual environment (recommended)

**Linux / macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install the package in editable mode

```bash
pip install -e .
```

This installs all required dependencies (NumPy, SciPy, wfdb, PyWavelets, Streamlit, Plotly, onnxruntime, ReportLab, Pydantic, and others) and registers the `ecg_monitor` package so Python finds it from anywhere in the repo.

### 4. Download the MIT-BIH ECG data

The signal files are large and not committed to the repo. Download them from PhysioNet into the `data/` directory.

**Linux / macOS (bash):**
```bash
cd data
for id in 100 101 105; do
    for ext in dat hea atr; do
        curl -L -o ${id}.${ext} https://physionet.org/files/mitdb/1.0.0/${id}.${ext}
    done
done
cd ..
```

**Windows PowerShell:**
```powershell
cd data
foreach ($id in 100, 101, 105) {
    foreach ($ext in "dat", "hea", "atr") {
        Invoke-WebRequest -Uri "https://physionet.org/files/mitdb/1.0.0/$id.$ext" -OutFile "$id.$ext"
    }
}
cd ..
```

Each `.dat` file should be approximately **1.95 MB**. If any file is smaller, the download was truncated — delete it and re-download.

> **Note:** If PhysioNet downloads time out from your region, open the direct URLs in your browser and use "Save As" to download them manually into the `data/` folder.

### 5. Place the trained ML model

The pre-trained ONNX model is not committed to the repo. You have two options:

**Option A: Use the provided model** (if a teammate shared it with you)

Place `beat_classifier.onnx` and `metrics.json` inside the `models/` folder at the repo root.

**Option B: Train from scratch**

See the "Training the ML Model" section below. Takes ~90 minutes on a free Kaggle GPU.

Verify the model is in place:

```bash
ls models/
```

You should see `beat_classifier.onnx` (~3.8 MB) and `metrics.json`.

---

## Running the Dashboard

```bash
streamlit run app/main.py
```

Open the URL printed in the terminal (usually `http://localhost:8501`) in a browser.

> **Important:** Use the sidebar navigation on the left, under "ECG Dashboard". The top URL links (e.g. `/beat_classification`) are Streamlit's auto-generated page routes and will not work because they bypass the shared record loading.

### Dashboard Pages

| Page | What it does |
|---|---|
| **Overview** | Load a record, view raw vs filtered signal, inspect quality score, see detected R-peaks |
| **Live Replay** | Stream the recording in chunks to simulate a real-time monitor |
| **Beat Classification** | Run the ONNX model on every detected beat; see per-beat AAMI labels overlaid on the ECG with confidence scores |
| **Experiments** | Launch noise robustness sweeps and detector comparison benchmarks |

### Beat Classification Page

On this page you will see:

- A caption line showing the number of R-peaks detected
- The ECG plot with colored markers on each beat (green: N, orange: S, red: V, purple: F, gray: Q)
- Hover tooltips with predicted class and confidence
- A beat distribution table and confidence histogram
- Offline training benchmark panels loaded from `models/metrics.json` (label efficiency, noise robustness, confusion matrix)

For a demo with more class diversity, download record 208 or 221 (both have significant ventricular ectopic beats):

```bash
cd data
for id in 208 221; do
    for ext in dat hea atr; do
        curl -L -o ${id}.${ext} https://physionet.org/files/mitdb/1.0.0/${id}.${ext}
    done
done
cd ..
```

---

## Running Experiments from the Command Line

Three experiment scripts live in the `experiments/` directory. Each produces a CSV result file in `results/`.

```bash
python experiments/run_exp1_noise.py       # Noise robustness sweep across records 100, 101, 105
python experiments/run_exp2_pipeline.py    # Compare Basic vs Standard vs Enhanced filter pipelines
python experiments/run_exp3_detectors.py   # Benchmark custom QRS detector against NeuroKit2 and BioSPPy
```

---

## Running the Test Suite

Fast unit tests only (no data files required):

```bash
pytest
```

Full test suite including integration tests that require cached MIT-BIH records:

```bash
pytest -m "integration or not integration"
```

---

## Training the ML Model (Kaggle)

The ML component is trained externally on Kaggle because it requires a GPU. All other components run on CPU.

### Steps

1. Create a new notebook on [kaggle.com](https://kaggle.com) and enable GPU (P100 or T4)
2. Add the **MIT-BIH Arrhythmia Database** by Mondejar as an input dataset (search Kaggle for `mondejar mitbih-database`)
3. Paste the entire contents of `kaggle_train_ssl_ecg.py` into one code cell
4. Run the cell. Expected runtime: 90–120 minutes
5. From Kaggle's Output tab, download:
   - `beat_classifier.onnx`
   - `metrics.json`
   - `ssl_loss_curve.png`
   - `label_efficiency.png`
   - `noise_robustness.png`
6. Place `beat_classifier.onnx` and `metrics.json` into the `models/` folder of the repo
7. Restart the Streamlit dashboard

The script performs: SimCLR pre-training on noise-augmented beats, supervised fine-tuning on AAMI DS1 labels (at 1%, 10%, 100% label fractions), evaluation on DS2 with macro-F1, noise robustness sweep, and ONNX export.

---

## Repository Structure

```
ECD_DSP123/
├── app/                          # Streamlit dashboard
│   ├── main.py                   # Entry point
│   └── pages/                    # Dashboard pages
│       ├── overview.py
│       ├── streaming.py
│       ├── experiments.py
│       └── beat_classification.py
├── src/ecg_monitor/              # Core library
│   ├── acquisition/              # Load MIT-BIH records via wfdb
│   ├── artifacts/                # Noise generators (baseline wander, powerline, EMG, motion)
│   ├── preprocessing/            # Filter pipelines (Basic, Standard, Enhanced)
│   ├── quality/                  # Signal quality scoring
│   ├── detection/                # QRS / R-peak detector
│   ├── evaluation/               # Peak matching against ground-truth annotations
│   ├── heart_rate/               # HR and HRV statistics
│   ├── alerts/                   # Rule-based alert engine
│   ├── streaming/                # Live replay simulator
│   ├── reporting/                # PDF, CSV, JSON report generation
│   ├── experiments/              # Experiment orchestration
│   ├── learned/                  # ML inference wrapper (ONNX runtime)
│   └── core/                     # Shared Pydantic data models
├── tests/                        # Unit and integration tests
├── data/                         # MIT-BIH recordings (not in git)
├── models/                       # Trained ONNX model (not in git)
├── results/                      # Experiment output CSVs
├── config.yaml                   # Project configuration
├── kaggle_train_ssl_ecg.py       # ML training script (run on Kaggle)
├── pyproject.toml                # Package build configuration
└── README.md
```

---

## Troubleshooting

**"Running load_ecg_record(...)" hangs on dashboard load**

The record's `.dat` file is missing from `data/`. Only `.hea` and `.atr` (text files) are tiny and may be committed; the `.dat` is binary and must be downloaded. Check `ls data/` and verify all three files exist for the record you are loading.

**"Failed to load record X: operands could not be broadcast together"**

The `.dat` file is truncated. Check its size — it should be approximately 1.95 MB. Re-download. If PhysioNet keeps timing out, download via a browser instead of `curl` or `Invoke-WebRequest`.

**Beat Classification page shows "Model unavailable"**

The ONNX model is missing from `models/`. Either copy `beat_classifier.onnx` into `models/`, or train it on Kaggle (see above).

**Beat Classification page is blank**

Streamlit cached a stale version. Clear the Python bytecode cache and restart:

```bash
# Linux / macOS
rm -rf app/pages/__pycache__ src/ecg_monitor/learned/__pycache__

# Windows PowerShell
Remove-Item -Recurse -Force app\pages\__pycache__
Remove-Item -Recurse -Force src\ecg_monitor\learned\__pycache__
```

Then stop Streamlit (Ctrl+C) and run `streamlit run app/main.py` again. Hard refresh the browser (Ctrl+Shift+R).

**Top URL links (e.g. `/beat_classification`) do not work**

Those are Streamlit's auto-detected pages and bypass the shared record loader. Use the sidebar navigation under "ECG Dashboard" instead.

**pip install fails on PyTorch or CUDA packages**

PyTorch is only required for training on Kaggle. The local install uses lightweight CPU inference via `onnxruntime`. If PyTorch is listed as a hard dependency in `pyproject.toml`, remove it before installing locally — or install everything with `pip install -e . --no-deps` and then install only the inference-time packages manually.

---

## Documentation

- **PROJECT_REPORT.pdf** — Full technical project report (overview, architecture, pipeline, experiments, results, limitations)
- **ML_EXPLAINED.pdf** — Deep-dive explainer of the machine-learning component

---

## License

MIT License. See `LICENSE` file for details.

## Citation

Based on the PhysioNet MIT-BIH Arrhythmia Database:

> Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45–50 (May–June 2001).
> Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215–e220 (2000).

## Disclaimer

This is an educational research prototype. All classifications and alerts are candidates for human review, not clinical diagnoses. The system is not approved for medical use and should never be used as a substitute for professional medical judgment.