import re

with open('/tmp/PROJECT_STATE.md', 'r') as f:
    text = f.read()

# 1. New Current State and Project Summary
top_section = """# PROJECT_STATE.md
# ECG Monitor — Single Source of Truth

> **Read this file first.** It contains everything needed to understand, extend,
> or resume work on this project — architecture, data contracts, module ownership,
> and a chronological build log. Assume zero other context.

---

## Current State

The core ECG analysis pipeline (Phases 1–15) is fully implemented, strictly typed, and rigorously tested (100% coverage). The originally planned Phase 16 (FastAPI/React/Docker/CI) was skipped by decision in favor of completing a comprehensive dashboard bugfix pass (Issues 1–6) and a full project audit, both of which are now complete. Additionally, the Phase 7 enhancement — a Pan-Tompkins++ style explicit 3-level adaptive thresholding upgrade with Hilbert Transform envelope refinement — has been implemented, validated, and merged into the main detector, showing improved robustness on noisy records. The project is fully mature and ready for final submission or evaluation.

---

## Project Summary

**What it is:**
An educational, rule-based signal-processing research platform for analyzing physiological ECG data. It evaluates standard PhysioNet/MIT-BIH ECG recordings without relying on any black-box AI/ML models, strictly utilizing classical digital signal processing (DSP) techniques.

**What it does end-to-end:**
1. **Acquisition:** Loads PhysioNet records and their expert annotations via `wfdb`.
2. **Simulation:** Injects realistic noise artifacts (baseline wander, powerline interference, motion, EMG) for robustness testing.
3. **Preprocessing:** Applies a tiered filter pipeline (Basic, Standard, Enhanced with wavelet denoising) to clean the signal.
4. **Quality Assessment:** Computes SNR, RMSE, and correlation against clean references to categorize signal quality as GOOD, FAIR, or POOR.
5. **Detection:** Extracts QRS complexes (R-peaks) using a highly robust custom Pan-Tompkins++ algorithm with a 3-level adaptive threshold and Hilbert transform envelope refinement.
6. **Feature Extraction:** Computes instantaneous heart rate (HR) and heart rate variability (HRV) metrics (SDNN, RMSSD, CoV).
7. **Alert Engine:** Continuously evaluates RR intervals against clinical rules to flag candidate events (e.g., Tachycardia, Bradycardia, Irregular Rhythm). Signals classified as POOR quality proactively suppress clinical alerts to avoid false positives, instead flagging a "Low Signal Quality" event.
8. **Reporting & Dashboard:** Provides a real-time streaming simulation UI via Streamlit and generates comprehensive PDF/CSV session reports.

**Validated Results:**
* **Detection Accuracy:** The custom detector achieves an F1 score of ~0.984 with highly accurate timing error (-0.96 ms), performing competitively against production-grade libraries like NeuroKit2 and BioSPPy.
* **Robustness:** Exceptional resilience against baseline wander and powerline interference (retaining ~0.984 F1 even at 150% noise amplitude). Hilbert refinement specifically tightens timing error on noisy records (e.g. Record 105 F1 boosts to 0.967). High-amplitude motion artifacts remain the primary failure mode.
* **Alert Engine:** Successfully detects bradycardia and tachycardia episodes accurately without raising false alarms on clean physiological baselines.

**Tech Stack:**
Python 3.12+, Hatchling, Pydantic (data contracts), NumPy/SciPy (DSP), Streamlit (dashboard), ReportLab (PDFs), pytest (testing).

**Known Limitations:**
The system is an educational engineering prototype. Alert flags are strictly rule-based candidates meant for human review; it is not a certified medical device and makes no diagnostic claims. The dataset parser currently assumes standard single-segment MIT-BIH `.atr` annotations.
"""

# 2. Architecture & Pipeline
arch_pipeline = """---

## Architecture & Pipeline

Build order below maps directly to implementation phases:

| Phase | Component | Description |
|-------|-----------|-------------|
| 1  | **Scaffolding** | Folder structure, pyproject.toml, Pydantic models, config schema, config.yaml, this file |
| 2  | **Data Acquisition** | MIT-BIH/PhysioNet loader via `wfdb` → `ECGPackage` |
| 3  | **Noise Simulation** | Baseline wander, 50 Hz powerline, EMG, motion artifacts — config-driven, reproducible via seed |
| 4  | **Preprocessing** | Three tiers: Basic (bandpass), Standard (bandpass+notch), Enhanced (baseline corr+notch+bandpass+wavelet) |
| 5  | **Signal Quality** | SNR, RMSE, correlation vs clean reference → combined score → GOOD/FAIR/POOR |
| 6  | **FFT Analysis** | Frequency-domain magnitude spectra before and after filtering |
| 7  | **QRS Detection** | Custom Pan-Tompkins++ detector with 3-level adaptive thresholding and Hilbert Transform envelope refinement |
| 8  | **Evaluation** | Match detected peaks to reference annotations (tolerance window) → Sensitivity, PPV, F1, timing error |
| 9  | **Heart Rate / HRV** | Instantaneous/avg/min/max HR, RR intervals, SDNN, RMSSD, RR CoV |
| 10 | **Alert Engine** | Rule-based tachycardia/bradycardia/irregular-RR candidates; POOR quality → "Low Signal Quality" event |
| 11 | **Experiment Runner** | Noise-level vs accuracy curves, pipeline comparison, detector vs NeuroKit2/BioSPPy benchmark |
| 12 | **Streaming Simulation** | Chunk-fed replay; latency/throughput/real-time factor tracking |
| 13 | **Reporting** | Auto-generated PDF (ReportLab), CSV, JSON per experiment/session |
| 14 | **Streamlit Dashboard** | Record/config selection, ECG plots, HR/quality display, FFT, alerts log, experiments page |
| 15 | **Test Suite & Audit** | 100% pytest coverage across all modules, comprehensive bugfix pass (Issues 1-6) |
| 16 | **Project Cleanup** | Removal of dead code, old prototypes, unneeded artifacts; FastAPI/React polish skipped by decision |
"""

# 3. Data Contract
data_contract = """---

## Data Contract

These exact field names and types are the canonical shared contract.
**Never rename or restructure without updating all pipeline stages.**

### ECGPackage

```python
class ECGPackage(BaseModel):
    record_id: str
    sampling_rate: int
    raw_signal: np.ndarray
    filtered_signal: np.ndarray
    quality: SignalQuality        # label: str, snr_db: float, rmse: float, correlation: float
    reference_annotations: ReferenceAnnotations  # sample_indices: list[int], symbols: list[str]
```

### ExperimentResult

```python
class ExperimentResult(BaseModel):
    timestamp: datetime
    config: ExperimentConfig
    snr_db: float | None
    rmse: float | None
    correlation: float | None
    sensitivity: float
    ppv: float
    f1: float
    timing_error_mean_ms: float
    timing_error_std_ms: float
    hr_average_bpm: float
    hrv_sdnn_ms: float
    quality: SignalQuality | None
    hr_stats: HeartRateStats | None
    hrv_stats: RRVariability | None
    detected_peaks: list[int]
    alerts: list[AlertEvent]
```

### AlertEvent

```python
class AlertEvent(BaseModel):
    event_id: str          # UUID, auto-generated
    event_type: str        # e.g. 'tachycardia_candidate', 'low_signal_quality'
    timestamp: datetime    # UTC
    duration: float        # seconds
    severity: str          # 'LOW' | 'MEDIUM' | 'HIGH'
    heart_rate: float | None
    signal_quality: str    # 'GOOD' | 'FAIR' | 'POOR'
    reason: str            # Human-readable explanation
    parameters: dict[str, Any]  # Rule-specific values used
```

All models live in `src/ecg_monitor/core/models.py` or `src/ecg_monitor/experiments/models.py`.
"""

# 4. Module Ownership Map
ownership_map = """---

## Module Ownership Map

| Folder | Responsibility |
|--------|---------------|
| `src/ecg_monitor/acquisition/` | PhysioNet/MIT-BIH record loading via `wfdb`; produces `ECGPackage` |
| `src/ecg_monitor/artifacts/` | Noise and artifact injection (baseline wander, powerline, EMG, motion) |
| `src/ecg_monitor/preprocessing/` | Tiered filter pipelines: Basic, Standard, Enhanced (wavelet) |
| `src/ecg_monitor/quality/` | SNR, RMSE, correlation computation; GOOD/FAIR/POOR labelling |
| `src/ecg_monitor/detection/` | Custom Pan-Tompkins++ QRS / R-peak detector |
| `src/ecg_monitor/evaluation/` | Peak-matching, Sensitivity/PPV/F1, timing-error statistics |
| `src/ecg_monitor/heart_rate/` | RR interval extraction, HR stats, SDNN/RMSSD/CoV |
| `src/ecg_monitor/alerts/` | Rule-based alert engine; `AlertEvent` generation |
| `src/ecg_monitor/streaming/` | Chunk-fed replay simulation; latency/throughput tracking |
| `src/ecg_monitor/reporting/` | PDF (ReportLab), CSV, JSON report generation |
| `src/ecg_monitor/visualization/` | Matplotlib/Plotly plot helpers used by dashboard and reports |
| `src/ecg_monitor/config/` | Pydantic config schema; `load_config()` from YAML |
| `src/ecg_monitor/core/` | Canonical Pydantic data models (ECGPackage, AlertEvent) |
| `src/ecg_monitor/experiments/` | Experiment configurations and result models |
| `app/` | Streamlit dashboard entry point and page modules |
| `experiments/` | Experiment runner scripts; outputs go to `results/` |
| `tests/` | pytest suite mirroring `src/ecg_monitor/` structure |
| `data/` | Raw PhysioNet records (git-ignored; populated at runtime via `wfdb`) |
| `results/` | Generated PDFs, CSVs, JSONs (git-ignored) |
"""

# 5. Open Questions
open_questions = """---

## Open Questions / Flags

* **OQ-1 — `SignalQuality` placeholder NaN values:** **Resolved** (Phase 5). `score_signal()` now fully computes metrics.
* **OQ-2 — Multi-segment MIT-BIH records:** **Still Open**. `wfdb.rdrecord` defaults to single segment; handling true multi-segment records (e.g. 207) is deferred.
* **OQ-3 — Annotation file for non-MIT-BIH databases:** **Still Open**. Downloader assumes `.atr`; `AcquisitionConfig.annotation_ext` needs explicit hook for other datasets.
* **OQ-4 — Config schema field rename (`probability` → `activity_probability`):** **Resolved** (Phase 3). Config and schema updated to disambiguate.
* **OQ-5 — EMG activity gating:** **Still Open**. Currently uses a single contiguous block; random scattered bursts deferred if needed for future experiments.
* **OQ-6 — Signal Quality empirical thresholds:** **Resolved** (Phase 11). Experimental sweeps confirmed thresholds correlate correctly with QRS detector degradation.
"""

# 6. Constraints
constraints = """---

## Constraints (apply to every phase)

- **No AI/ML models** — classical DSP only.
- **No database** — file I/O (JSON/CSV/PDF) only.
- **No medical-device language** — use "candidate", "rule-based", "educational prototype", "human review recommended".
- **Highly modular** — one responsibility per file/function; no god-files.
- **Type-hinted throughout** (Python 3.12+, strict mypy).
- **Config-driven** — nothing hardcoded that the data contract marks as configurable; all parameters live in `config.yaml`.
- **Pydantic models** for all structured data — never pass loose dicts between modules.
- **Minimal inline comments** — only docstrings where purpose isn't obvious from name/signature.
"""

# Extract build log (everything from "## Build Log" downwards)
# We will strip out standalone `## Current State` and `## Open Questions / Flags` lines
# and their immediate paragraphs to avoid noise, since they're consolidated at the top.

build_log_start = text.find('## Build Log')
if build_log_start == -1:
    print("Could not find ## Build Log")
    exit(1)

raw_build_log = text[build_log_start:]

# We want to remove blocks like:
# ## Current State\n\nPhase X complete. ...\n\n---
# and ## Open Questions / Flags ... ---

# Better yet, since we have strict ### Phase headers, we can just split by them.
# Or use regex to remove '## Current State' up to the next '---'
clean_build_log = re.sub(r'## Current State\n.*?\n---', '', raw_build_log, flags=re.DOTALL)
clean_build_log = re.sub(r'## Open Questions / Flags\n.*?\n---', '', clean_build_log, flags=re.DOTALL)
clean_build_log = re.sub(r'## Open Questions / Flags \(updated\)\n.*?\n---', '', clean_build_log, flags=re.DOTALL)
# There's also the standalone block for OQ-6 inside Phase 5:
# Wait, let's leave OQ-6 text as is inside Phase 5 because it's part of the phase notes.

# Let's clean up any double newlines created by the substitution
clean_build_log = re.sub(r'\n{3,}', '\n\n', clean_build_log)

final_text = top_section + "\n" + arch_pipeline + "\n" + data_contract + "\n" + ownership_map + "\n" + open_questions + "\n" + constraints + "\n---\n\n" + clean_build_log

with open('PROJECT_STATE_NEW.md', 'w') as f:
    f.write(final_text)

print("SUCCESS")
