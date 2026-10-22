# Noise-Resilient ECG Signal Processing & Arrhythmia Classification Platform

An educational research platform that combines classical digital signal processing with self-supervised deep learning for noise-robust ECG analysis on PhysioNet/MIT-BIH recordings.

The system performs three core tasks:

1. Cleans noisy ECG signals using tiered filter pipelines (bandpass, notch, wavelet denoising)
2. Detects every heartbeat with a custom Pan-Tompkins-inspired QRS detector
3. Classifies each detected beat into AAMI 5-class arrhythmia categories (N, S, V, F, Q) using a 1D ResNet encoder pre-trained with contrastive learning on noise-augmented beats

**EDUCATIONAL PROTOTYPE — NOT A MEDICAL DEVICE — NOT FOR CLINICAL USE.** All outputs are indicative only and require human review. The system makes no diagnostic claims.

---

## Prerequisites

- Python 3.12 or higher
- Linux, macOS, or Windows
- About 500 MB free disk space for data and dependencies
- No GPU required. CPU inference works fine for the dashboard.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/arushiagrawal-123/ECD_DSP123.git
cd ECD_DSP123
```

The repository already contains the trained ONNX model (`models/beat_classifier.onnx`, 3.8 MB) and the training metrics (`models/metrics.json`). You do not need to train anything yourself to run the dashboard.

### 2. Create a virtual environment

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

### 3. Install the package

```bash
pip install -e ".[dev]"
```

This installs all runtime dependencies (NumPy, SciPy, wfdb, PyWavelets, Streamlit, Plotly, onnxruntime, ReportLab, Pydantic) plus development tools (pytest, ruff, mypy) and registers the `ecg_monitor` package in editable mode.

### 4. Download MIT-BIH recordings

The MIT-BIH signal files are not committed to the repo (they're binary and large). Download them from PhysioNet into the `data/` directory.

**Linux / macOS:**
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

Each `.dat` file should be around **1.95 MB**. If the download is smaller, it was truncated. Delete and retry, or download via a browser.

That's it. The dashboard is ready to run.

---

## Running the Dashboard

```bash
streamlit run app/main.py
```

Open the URL printed in the terminal (usually `http://localhost:8501`).

> **Use the sidebar navigation on the left**, under "ECG Dashboard". The top URL links (e.g. `/beat_classification`) are Streamlit's auto-generated routes and bypass the shared record loader, so they will not work.

### Pages

| Page | Purpose |
|---|---|
| **Overview** | Load a record, view raw vs filtered signal, inspect quality score, see detected R-peaks |
| **Live Replay** | Stream the recording in chunks to simulate a real-time monitor |
| **Beat Classification** | Run the ONNX model on every detected beat; see per-beat AAMI labels overlaid on the ECG with confidence scores |
| **Experiments** | Launch noise robustness sweeps and detector comparison benchmarks |

### Beat Classification Page in Detail

For every detected R-peak the model predicts one of five AAMI classes with a confidence score:

| Code | Color | Meaning |
|---|---|---|
| N | green | Normal / bundle-branch block beat |
| S | orange | Supraventricular ectopic (atrial) |
| V | red | Ventricular ectopic (PVC) |
| F | purple | Fusion beat |
| Q | gray | Unknown or paced |

The page displays:

- A caption showing how many R-peaks were detected
- ECG plot with colored markers on each beat and hover tooltips showing class + confidence
- Beat distribution table and confidence histogram
- Offline benchmark panels loaded from `models/metrics.json`: label efficiency bar chart, noise robustness line chart, confusion matrix heatmap

For a demo showing class diversity, download record 208 or 221 (both contain significant ventricular ectopic beats):

```bash
cd data
for id in 208 221; do
    for ext in dat hea atr; do
        curl -L -o ${id}.${ext} https://physionet.org/files/mitdb/1.0.0/${id}.${ext}
    done
done
cd ..
```

Record 100 is almost entirely normal beats, so expect mostly green dots.

---

## Running Experiments from the CLI

Three experiment scripts in `experiments/`, each writing a CSV to `results/`:

```bash
python experiments/run_exp1_noise.py       # Noise robustness sweep across records 100, 101, 105
python experiments/run_exp2_pipeline.py    # Compare Basic vs Standard vs Enhanced filter pipelines
python experiments/run_exp3_detectors.py   # Benchmark the custom QRS detector against NeuroKit2 and BioSPPy
```

---

## Running the Tests

Fast unit tests only:

```bash
pytest
```

Full suite including integration tests (requires MIT-BIH data in `data/`):

```bash
pytest -m "integration or not integration"
```

---

## Retraining the ML Model (Optional)

The ONNX model shipped with the repo was trained on Kaggle. If you want to retrain or modify the training, follow these steps.

1. Create a new notebook on [kaggle.com](https://kaggle.com) with GPU enabled (P100 or T4)
2. Add the MIT-BIH Arrhythmia Database as an input dataset (search `mondejar mitbih-database`)
3. Paste the training script into one cell and run. Expected runtime: 90–120 minutes
4. Download `beat_classifier.onnx` and `metrics.json` from Kaggle's Output tab
5. Replace the files in `models/` with the new ones
6. Restart the Streamlit dashboard

The training performs: SimCLR contrastive pre-training on noise-augmented beats, supervised fine-tuning on AAMI DS1 labels at 1%, 10%, and 100% label fractions, evaluation on DS2 with macro-F1, a noise robustness sweep, and ONNX export.

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
│   ├── acquisition/              # Load MIT-BIH records (wfdb)
│   ├── artifacts/                # Noise generators: baseline wander, powerline, EMG, motion
│   ├── preprocessing/            # Filter pipelines: Basic, Standard, Enhanced
│   ├── quality/                  # Signal quality scoring (reference-based + reference-free)
│   ├── detection/                # QRS / R-peak detector
│   ├── evaluation/               # Peak matching against ground-truth annotations
│   ├── heart_rate/               # HR and HRV statistics
│   ├── alerts/                   # Rule-based alert engine
│   ├── streaming/                # Live replay simulator
│   ├── reporting/                # PDF, CSV, JSON report generation
│   ├── experiments/              # Experiment orchestration
│   ├── learned/                  # ML inference (ONNX runtime wrapper)
│   └── core/                     # Shared Pydantic data models
├── tests/                        # Unit and integration tests
├── data/                         # MIT-BIH recordings (not in git, download yourself)
├── models/                       # Trained model (committed to git)
│   ├── beat_classifier.onnx
│   └── metrics.json
├── results/                      # Experiment output CSVs (not in git)
├── config.yaml                   # Project configuration
├── pyproject.toml                # Package build configuration
├── PROJECT_STATE.md              # Build log and architectural decisions
└── README.md
```

---

## Troubleshooting

**"Running load_ecg_record(...)" hangs on dashboard load**
The record's `.dat` file is missing from `data/`. Run `ls data/` and confirm you have `.dat`, `.hea`, and `.atr` for the record you want to load.

**"Failed to load record X: operands could not be broadcast together"**
The `.dat` file is truncated. Check its size — it should be about 1.95 MB. Re-download; if `curl` keeps timing out, download via a browser.

**Beat Classification page shows "Model unavailable"**
The ONNX model is missing from `models/`. It ships with the repo, so this usually means someone deleted it. Pull the latest version from git or re-clone.

**Beat Classification page is blank**
Streamlit cached stale bytecode. Clear it and restart:

```bash
# Linux / macOS
rm -rf app/pages/__pycache__ src/ecg_monitor/learned/__pycache__

# Windows PowerShell
Remove-Item -Recurse -Force app\pages\__pycache__
Remove-Item -Recurse -Force src\ecg_monitor\learned\__pycache__
```

Stop Streamlit (Ctrl+C), run `streamlit run app/main.py` again, and hard-refresh the browser (Ctrl+Shift+R).

**Top URL links do not work**
Those are Streamlit's auto-detected pages. They bypass the shared record loader. Use the sidebar navigation under "ECG Dashboard" instead.

---

## License

MIT License (as declared in `pyproject.toml`).

## Citation

Based on the PhysioNet MIT-BIH Arrhythmia Database:

> Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45–50 (May–June 2001).
>
> Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215–e220 (2000).