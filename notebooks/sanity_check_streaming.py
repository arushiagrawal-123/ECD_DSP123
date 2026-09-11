"""
Sanity check script for Phase 12 Streaming / Real-Time Replay Module.

Loads a real ECG record, replays it chunk-by-chunk through the StreamingEngine,
and prints a live ticker of the state.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.config.schema import load_config
from ecg_monitor.streaming.engine import StreamingEngine


def main() -> None:
    data_dir = Path(__file__).parents[1] / "data"
    config_path = Path(__file__).parents[1] / "config.yaml"
    config = load_config(config_path)

    # Use a small chunk size for a more "live" feel
    config.streaming.chunk_duration_s = 0.5
    
    print("Loading Record 100 for live replay...")
    pkg = load_record("100", data_dir)
    
    # We will just replay the first 30 seconds to keep it brief
    duration_s = 30
    fs = pkg.sampling_rate
    signal = pkg.raw_signal[:duration_s * fs]
    
    engine = StreamingEngine(config, fs)
    
    chunk_size = int(config.streaming.chunk_duration_s * fs)
    n_samples = len(signal)
    
    print(f"Starting replay of {duration_s} seconds (Chunk = {config.streaming.chunk_duration_s}s)...\n")
    print(f"{'Time':>6} | {'Quality':>7} | {'HR (BPM)':>8} | {'Cumulative Peaks':>16} | Alerts")
    print("-" * 75)
    
    snapshots = []
    
    for i in range(0, n_samples, chunk_size):
        chunk = signal[i:i + chunk_size]
        
        # In a real GUI this step would be asynchronous.
        snap = engine.step(chunk)
        snapshots.append(snap)
        
        hr_str = f"{snap.current_heart_rate_bpm:.1f}" if snap.current_heart_rate_bpm else "---"
        alerts_str = ", ".join(a.event_type for a in snap.new_alerts) if snap.new_alerts else ""
        
        print(f"{snap.elapsed_simulated_time_s:5.1f}s | {snap.current_signal_quality:>7} | {hr_str:>8} | {len(snap.cumulative_peaks):>16} | {alerts_str}")
        
        # Sleep briefly to simulate live viewing if desired (commented out for speed)
        # time.sleep(0.1)

    print("-" * 75)
    
    # Compute performance
    total_processing_time = sum(s for s in [0.0] if s) # we tracked this in engine!
    # Wait, the engine tracks total processing time natively. 
    # Let's just ask the engine to do a full replay to get the exact performance object,
    # or we can manually reconstruct it since we stepped manually.
    # We'll just calculate it here:
    total_wall_time = engine._total_processing_time_s
    rtf = duration_s / total_wall_time if total_wall_time > 0 else float('inf')
    
    print(f"\nReplay Complete!")
    print(f"Simulated Duration : {duration_s:.1f} seconds")
    print(f"Wall-Clock Time    : {total_wall_time:.3f} seconds")
    print(f"Real-Time Factor   : {rtf:.1f}x (Processed {rtf:.1f} seconds of data per 1 second of wall-clock time)")


if __name__ == "__main__":
    main()
