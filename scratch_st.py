import streamlit as st
import pandas as pd
from datetime import datetime

alert_dicts = [
    {
        "Time (ISO)": datetime.now().strftime("%H:%M:%S"),
        "Type": "irregular_rr_candidate",
        "HR": "75.0",
        "Reason": "Irregular heart rate detected.",
    }
]

# Write a fake app
st.dataframe(pd.DataFrame(alert_dicts), hide_index=True)
