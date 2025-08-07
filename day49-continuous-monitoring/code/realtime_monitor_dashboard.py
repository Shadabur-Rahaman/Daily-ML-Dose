import streamlit as st
import pandas as pd

st.title("Real-Time ML Monitoring")

@st.cache
def load_metrics():  
    return pd.read_csv("live_metrics.csv")

df = load_metrics()
st.line_chart(df[["accuracy", "precision", "recall"]])
