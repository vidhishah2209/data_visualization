import streamlit as st
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit.components.v1 as components

# Set page config
st.set_page_config(
    page_title="Credit Card Fraud Analysis",
    page_icon="💳",
    layout="wide"
)

# Title and description
st.title("Credit Card Fraud Analysis Dashboard")
st.markdown("""
This dashboard displays various visualizations and insights from the credit card fraud detection pipeline.
""")

# Load data
@st.cache_data
def load_data():
    data_path = "data/raw/creditCardFraud_28011964_120214 (2).csv"
    return pd.read_csv(data_path)

# Load visualizations
@st.cache_data
def load_visualizations():
    viz_dir = Path("results/visualizations")
    visualizations = {}
    if viz_dir.exists():
        for file in viz_dir.glob("*.html"):
            with open(file, 'r', encoding='utf-8') as f:
                visualizations[file.stem] = f.read()
    return visualizations

# Load metrics
@st.cache_data
def load_metrics():
    metrics_dir = Path("results/metrics")
    metrics = {}
    if metrics_dir.exists():
        for file in metrics_dir.glob("*.json"):
            metrics[file.stem] = pd.read_json(file)
    return metrics

# Main app
def main():
    # Load data and results
    data = load_data()
    visualizations = load_visualizations()
    metrics = load_metrics()

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a section",
        ["Data Overview", "Feature Analysis", "Model Performance", "Predictions"]
    )

    if page == "Data Overview":
        st.header("Data Overview")
        
        # Display basic statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(data))
        with col2:
            st.metric("Features", len(data.columns))
        with col3:
            st.metric("Fraud Rate", f"{(data['default payment next month'].mean() * 100):.2f}%")

        # Display data sample
        st.subheader("Data Sample")
        st.dataframe(data.head())

        # Display correlation heatmap if available
        if "correlation_heatmap" in visualizations:
            st.subheader("Correlation Heatmap")
            st.components.v1.html(visualizations["correlation_heatmap"], height=600)

    elif page == "Feature Analysis":
        st.header("Feature Analysis")
        
        # Display distribution plots
        if "distribution_plots" in visualizations:
            st.subheader("Feature Distributions")
            st.components.v1.html(visualizations["distribution_plots"], height=600)

        # Display box plots
        if "box_plots" in visualizations:
            st.subheader("Box Plots")
            st.components.v1.html(visualizations["box_plots"], height=600)

        # Display PCA visualization
        if "pca_3d" in visualizations:
            st.subheader("3D PCA Visualization")
            st.components.v1.html(visualizations["pca_3d"], height=600)

    elif page == "Model Performance":
        st.header("Model Performance")
        
        # Display model metrics
        if "model_metrics" in metrics:
            st.subheader("Model Metrics")
            st.dataframe(metrics["model_metrics"])

        # Display feature importance plots
        if "feature_importance" in visualizations:
            st.subheader("Feature Importance")
            st.components.v1.html(visualizations["feature_importance"], height=600)

        # Display ROC curves
        if "roc_curves" in visualizations:
            st.subheader("ROC Curves")
            st.components.v1.html(visualizations["roc_curves"], height=600)

    elif page == "Predictions":
        st.header("Predictions")
        
        # Add prediction interface
        st.subheader("Make a Prediction")
        
        # Create input fields for features
        col1, col2 = st.columns(2)
        with col1:
            limit_bal = st.number_input("Credit Limit", min_value=0, max_value=1000000)
            age = st.number_input("Age", min_value=18, max_value=100)
            education = st.selectbox("Education", [1, 2, 3, 4])
            marriage = st.selectbox("Marriage", [1, 2, 3])
        
        with col2:
            pay_0 = st.selectbox("Payment Status (Month 1)", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
            pay_2 = st.selectbox("Payment Status (Month 2)", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
            pay_3 = st.selectbox("Payment Status (Month 3)", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
            bill_amt1 = st.number_input("Bill Amount (Month 1)", min_value=-1000000, max_value=1000000)

        if st.button("Predict"):
            # Here you would load the model and make predictions
            st.success("Prediction completed!")
            st.info("This is a placeholder. To implement actual predictions, you would need to load the trained model.")

if __name__ == "__main__":
    main() 