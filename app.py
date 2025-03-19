import streamlit as st
import os
from pathlib import Path
import streamlit.components.v1 as components

st.set_page_config(page_title="Data Science Pipeline Visualizations", layout="wide")

st.title("Data Science Pipeline Visualizations")

# Define the path to visualizations
vis_path = Path("results/visualizations")

# Get all HTML files
html_files = list(vis_path.glob("*.html"))

# Create tabs for different visualization categories
tabs = {
    "Distribution Plots": [f for f in html_files if "distribution" in f.name],
    "Box Plots": [f for f in html_files if "box_plot" in f.name],
    "PCA & Scatter": [f for f in html_files if any(x in f.name for x in ["pca", "scatter"])],
    "Correlation & Relationships": [f for f in html_files if any(x in f.name for x in ["correlation", "relationships"])]
}

# Create tabs
selected_tab = st.sidebar.radio("Select Visualization Category", list(tabs.keys()))

st.header(selected_tab)

# Display files in the selected category
for file in tabs[selected_tab]:
    with st.expander(f"{file.stem}", expanded=True):
        with open(file, 'r', encoding='utf-8') as f:
            html_content = f.read()
            components.html(html_content, height=600, scrolling=True)

# Display metrics
st.sidebar.header("Model Metrics")
try:
    import pandas as pd
    metrics_df = pd.read_csv("results/metrics/metrics.csv")
    st.sidebar.dataframe(metrics_df)
    
    importance_df = pd.read_csv("results/metrics/feature_importance.csv")
    st.sidebar.header("Feature Importance")
    st.sidebar.dataframe(importance_df)
except Exception as e:
    st.sidebar.warning("Could not load metrics files") 