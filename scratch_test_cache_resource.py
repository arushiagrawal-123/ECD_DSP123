import sys
import os
sys.path.insert(0, os.path.abspath("."))
import streamlit as st
from app.utils import load_ecg_record

p100 = load_ecg_record("100")
p101 = load_ecg_record("101")

print(p100.record_id, p101.record_id)
