import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path("src")))

from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.config.schema import load_config
from ecg_monitor.detection.detector import QRSDetector
from ecg_monitor.evaluation.evaluator import evaluate_detection
from ecg_monitor.preprocessing.pipeline import preprocess, PreprocessingTier

def run_eval():
    config = load_config("config.yaml")
    data_dir = Path("data")
    
    records = ["100", "101", "105"]
    
    print("Record | Hilbert | Sens  | PPV   | F1    | Mean Err | Std Err")
    print("-" * 65)
    
    for rec in records:
        pkg = load_record(rec, data_dir)
        fs = pkg.sampling_rate
        
        filtered = preprocess(pkg.raw_signal, fs, config.filters, PreprocessingTier.ENHANCED)
        pkg.filtered_signal = filtered
        
        for use_hilbert in [False, True]:
            config.detection.use_hilbert_refinement = use_hilbert
            detector = QRSDetector(config.detection, fs)
            peaks = detector.detect(filtered)
            
            res = evaluate_detection(peaks, pkg, config)
            
            hilbert_str = "ON " if use_hilbert else "OFF"
            print(f"{rec:6} | {hilbert_str:7} | {res.sensitivity:.3f} | {res.ppv:.3f} | {res.f1:.3f} | {res.timing_error_ms.mean_ms:8.2f} | {res.timing_error_ms.std_ms:7.2f}")

if __name__ == "__main__":
    run_eval()
