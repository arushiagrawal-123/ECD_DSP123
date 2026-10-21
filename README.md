# ECG Signal Processing & Arrhythmia Classification Platform

An educational research platform that combines classical DSP with self-supervised deep learning for noise-robust ECG analysis on MIT-BIH recordings.

The system does three things:
1. Cleans noisy ECG signals using tiered filter pipelines (bandpass, notch, wavelet)
2. Detects heartbeats with a custom Pan-Tompkins-based QRS detector
3. Classifies every detected beat into AAMI arrhythmia categories (N, S, V, F, Q) using a 1D ResNet encoder pre-trained with contrastive learning

**This is not a medical device.** All outputs are indicative only and require human review.

---

## Prerequisites

- Python 3.12+
- Windows, Linux, or macOS
- About 2 GB free disk space (for data + model)

---

## Quick setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd ECD_DSP123
```

### 2. Create a virtual environment (optional but recommended)

Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux / macOS:
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install the package in editable mode

```bash
pip install -e .
```

This installs all dependencies (numpy, scipy, wfdb, pytorch, onnxruntime, streamlit, plotly, etc.) and registers the `ecg_monitor` package so Python finds it from anywhere.

### 4. Download the MIT-BIH data

The repo ships without the signal files (they're ~2MB each, too big for git). Download them manually.

For the three default records:

```powershell
cd data
Invoke-WebRequest -Uri https://physionet.org/files/mitdb/1.0.0/100.dat -OutFile 100.dat
Invoke-WebRequest -Uri https://physionet.org/files/mitdb/1.0.0/100.hea -OutFile 100.hea
Invoke-WebRequest -Uri https://physionet.org/files/mitdb/1.0.0/100.atr -OutFile 100.atr
Invoke-WebRequest -Uri https://physionet.org/files/mitdb/1.0.0/101.dat -OutFile 101.dat
Invoke-WebRequest -Uri https://physionet.org/files/mitdb/1.0.0/101.hea -OutFile 101.hea
Invoke-WebRequest -Uri https://physionet.org/files/mitdb/1.0.0/101.atr -OutFile 101.atr
Invoke-WebRequest -Uri https://physionet.org/files/mitdb/1.0.0/105.dat -OutFile 105.dat
Invoke-WebRequest -Uri https://physionet.org/files/mitdb/1.0.0/105.hea -OutFile 105.hea
Invoke-WebRequest -Uri https://physionet.org/files/mitdb/1.0.0/105.atr -OutFile 105.atr
cd ..
```

Each `.dat` file should be around 1.95 MB. If it's smaller, the download got truncated. Re-download or use Chrome.

### 5. Drop the trained model into `models/`

The repo doesn't ship the trained ONNX model either (it's 3.8 MB). Two options:

**Option A: Use the pre-trained model (if you have it)**

Copy `beat_classifier.onnx` and `metrics.json` into `C:\Users\Siddharth\ECD_DSP123\models\` (or wherever you cloned to).

**Option B: Train it yourself on Kaggle**

See the "Training" section below. Takes about 90 minutes on a free Kaggle P100 GPU.

Verify the model folder:
```powershell
dir models
```

Should list `beat_classifier.onnx` (~3.8 MB) and `metrics.json`.

---

## Running the dashboard

```bash
streamlit run app/main.py
```

Open the URL it prints (usually http://localhost:8501) in your browser.

**Use the left sidebar for navigation**, not the top URL links. The top links are Streamlit's auto-detected pages and don't work because they don't load a record.

**Pages**:
- **Overview**: load a record, see raw vs filtered signal, quality score, R-peak detection
- **Live Replay**: streaming simulator for the real-time demo
- **Beat Classification**: the ML page. Loads the ONNX model and classifies every beat
- **Experiments**: run noise sweeps and detector benchmarks

**On the Beat Classification page you'll see**:
- Line showing "Detected N R-peaks at 360 Hz"
- ECG plot with colored dots on each beat (green=normal, orange=S, red=V, purple=F, gray=Q)
- Hover over any dot to see predicted class + confidence
- Beat distribution table and confidence histogram below
- Offline training benchmarks (label efficiency, noise robustness, confusion matrix) further down

---

## Running experiments from the command line

Three experiment scripts live in `experiments/`. Each produces a CSV in `results/`.

```bash
python experiments/run_exp1_noise.py     # noise robustness sweep
python experiments/run_exp2_pipeline.py   # pipeline comparison
python experiments/run_exp3_detectors.py  # detector benchmark
```

---

## Running tests

```bash
pytest                        # unit tests only (fast)
pytest -m "integration or not integration"   # all tests including integration
```

---

## Training the ML model (Kaggle, ~90 min)

The ML component (beat classifier) was trained externally on Kaggle because it needs a GPU. Everything else runs on CPU.

**Steps**:

1. Create a new Kaggle notebook with GPU accelerator (P100 or T4)
2. Add input dataset: search for `mitbih-database` by mondejar
3. Paste the entire contents of `kaggle_train_ssl_ecg.py` into one code cell
4. Hit Run. Wait 90 minutes.
5. From the Kaggle output tab, download:
   - `beat_classifier.onnx`
   - `metrics.json`
   - `ssl_loss_curve.png`, `label_efficiency.png`, `noise_robustness.png`
6. Drop the first two into `models/` locally
7. Restart Streamlit

Full training details are in `TRAINING_GUIDE.md`.

---

## Folder structure

```
ECD_DSP123/
├── app/                      # Streamlit dashboard
│   ├── main.py
│   └── pages/
│       ├── overview.py
│       ├── streaming.py
│       ├── experiments.py
│       └── beat_classification.py   # ML inference page
├── src/ecg_monitor/          # Core library
│   ├── acquisition/          # Load MIT-BIH records (wfdb)
│   ├── artifacts/            # Noise generators (baseline wander, powerline, EMG, motion)
│   ├── preprocessing/        # Filter pipelines (Basic, Standard, Enhanced)
│   ├── quality/              # Signal quality scoring
│   ├── detection/            # QRS/R-peak detector
│   ├── evaluation/           # Peak matching against ground truth
│   ├── heart_rate/           # HR and HRV statistics
│   ├── alerts/               # Rule-based alert engine
│   ├── streaming/            # Live replay simulator
│   ├── reporting/            # PDF/CSV/JSON report generation
│   ├── experiments/          # Experiment orchestration
│   ├── learned/              # ML inference (ONNX wrapper)
│   └── core/                 # Shared data models
├── tests/                    # Unit and integration tests
├── data/                     # MIT-BIH recordings (not in git)
├── models/                   # Trained ONNX model (not in git)
├── results/                  # Experiment output CSVs
├── config.yaml               # Project configuration
├── kaggle_train_ssl_ecg.py   # ML training script
├── pyproject.toml            # Package config
└── README.md
```

---

## Common issues

**"Running load_ecg_record(...)" hangs forever on dashboard load**
→ The record's `.dat` file is missing. Check `dir data`. If only `.hea` and `.atr` are present, re-download the `.dat`.

**"Failed to load record X: operands could not be broadcast together"**
→ The record's `.dat` is truncated. File size should be ~1.95 MB. Re-download.

**Beat Classification page shows "Model unavailable"**
→ `models/beat_classifier.onnx` is missing. Either train on Kaggle or copy the model file in.

**Beat Classification page is completely blank**
→ Clear Streamlit's bytecode cache:
```powershell
Remove-Item -Recurse -Force app\pages\__pycache__
Remove-Item -Recurse -Force src\ecg_monitor\learned\__pycache__
```
Then restart Streamlit with Ctrl+C and `streamlit run app/main.py`.

**PhysioNet downloads keep timing out**
→ Download the files in Chrome instead. Right-click the direct URL and "Save As" into the `data/` folder.

**pip install fails on PyTorch**
→ PyTorch isn't a runtime dependency, only needed for training on Kaggle. The local install uses onnxruntime (lightweight CPU inference). If pyproject.toml lists torch, remove it or install with `pip install -e . --no-deps` then install everything else manually.

---

## License

MIT. See LICENSE file.

## Disclaimer

Educational prototype. All classifications and alerts are candidates for human review, not clinical diagnoses. Not approved for medical use.