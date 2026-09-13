import streamlit as st
from app.utils import load_ecg_record

pkg100 = load_ecg_record("100")
pkg101 = load_ecg_record("101")
st.write(f"100 ID: {pkg100.record_id}")
st.write(f"101 ID: {pkg101.record_id}")
