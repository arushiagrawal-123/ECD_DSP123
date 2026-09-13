import sys
import os
sys.path.insert(0, os.path.abspath("."))
from ecg_monitor.preprocessing.pipeline import PreprocessingTier
from app.utils import get_root_config, load_ecg_record
from app.pages.overview import run_static_analysis

config = get_root_config()
tier = PreprocessingTier.ENHANCED

# Load 100
pkg100 = load_ecg_record("100")
res100 = run_static_analysis(pkg100.record_id, pkg100, config, tier)
print(f"100 -> sensitivity={res100['eval_metrics'].sensitivity}, alerts={len(res100['alerts'])}")

# Load 101
pkg101 = load_ecg_record("101")
res101 = run_static_analysis(pkg101.record_id, pkg101, config, tier)
print(f"101 -> sensitivity={res101['eval_metrics'].sensitivity}, alerts={len(res101['alerts'])}")
