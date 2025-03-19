"""Interactive dashboard for feature engineering results."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_data():
    """Load processed data."""
    clean_df = pd.read_csv('../data/processed/clean/clean_data.csv')
    engineered_df = pd.read_csv('../data/processed/engineered/engineered_features.csv')
    return clean_df, engineered_df

def plot_feature_distribution(df, feature_name):
    """Create interactive distribution plot."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[feature_name], name=feature_name))
    fig.add_trace(go.Histogram(x=df[feature_name], name=f"{feature_name} KDE"))
    fig.update_layout(title=f"Distribution of {feature_name}")
    return fig

def plot_correlation_heatmap(df, title):
    """Create interactive correlation heatmap."""
    numeric_df = df.select_dtypes(include=[np.number])
    fig = px.imshow(numeric_df.corr(),
                    labels=dict(color="Correlation"),
                    title=title)
    return fig

def plot_pca_components(df):
    """Create PCA visualization."""
    numeric_df = df.select_dtypes(include=[np.number])
    if 'cluster_label' in df.columns:
        fig = px.scatter(df, x=numeric_df.columns[0], y=numeric_df.columns[1],
                        color='cluster_label',
                        title='Feature Space Clustering')
    else:
        fig = px.scatter(df, x=numeric_df.columns[0], y=numeric_df.columns[1],
                        title='Feature Space Distribution')
    return fig

def main():
    """Run the dashboard."""
    st.set_page_config(page_title="Feature Engineering Dashboard",
                      page_icon="📊",
                      layout="wide")
    
    st.title("Feature Engineering Dashboard 📊")
    
    try:
        clean_df, engineered_df = load_data()
        
        # Sidebar
        st.sidebar.header("Navigation")
        page = st.sidebar.radio("Go to", 
                              ["Overview",
                               "Original Features",
                               "Engineered Features",
                               "Comparative Analysis"])
        
        if page == "Overview":
            st.header("Dataset Overview")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Features", clean_df.shape[1])
            with col2:
                st.metric("Engineered Features", engineered_df.shape[1])
            with col3:
                st.metric("Total Samples", clean_df.shape[0])
            
            st.subheader("Original Features Preview")
            st.dataframe(clean_df.head())
            
            st.subheader("Engineered Features Preview")
            st.dataframe(engineered_df.head())
        
        elif page == "Original Features":
            st.header("Original Features Analysis")
            
            # Feature distribution
            st.subheader("Feature Distributions")
            feature = st.selectbox("Select feature:", clean_df.columns)
            st.plotly_chart(plot_feature_distribution(clean_df, feature))
            
            # Correlation matrix
            st.subheader("Correlation Matrix")
            st.plotly_chart(plot_correlation_heatmap(clean_df, 
                                                   "Original Features Correlation"))
            
            # Statistics
            st.subheader("Statistical Summary")
            st.dataframe(clean_df.describe())
        
        elif page == "Engineered Features":
            st.header("Engineered Features Analysis")
            
            # Feature distribution
            st.subheader("Feature Distributions")
            feature = st.selectbox("Select feature:", engineered_df.columns)
            st.plotly_chart(plot_feature_distribution(engineered_df, feature))
            
            # Correlation matrix
            st.subheader("Correlation Matrix")
            st.plotly_chart(plot_correlation_heatmap(engineered_df,
                                                   "Engineered Features Correlation"))
            
            # Statistics
            st.subheader("Statistical Summary")
            st.dataframe(engineered_df.describe())
        
        elif page == "Comparative Analysis":
            st.header("Comparative Analysis")
            
            # Feature counts
            st.subheader("Feature Type Distribution")
            feature_types = {
                'Original Numeric': len(clean_df.select_dtypes(include=[np.number]).columns),
                'Original Text': len(clean_df.select_dtypes(include=['object']).columns),
                'Engineered Numeric': len(engineered_df.select_dtypes(include=[np.number]).columns),
                'Engineered Text': len(engineered_df.select_dtypes(include=['object']).columns)
            }
            
            fig = px.bar(x=list(feature_types.keys()), 
                        y=list(feature_types.values()),
                        title="Feature Type Distribution")
            st.plotly_chart(fig)
            
            # PCA visualization
            st.subheader("Feature Space Visualization")
            st.plotly_chart(plot_pca_components(pd.concat([clean_df, engineered_df], axis=1)))
            
            # Feature importance
            if os.path.exists('../visualizations/feature_importance/shap_importance.png'):
                st.subheader("Feature Importance")
                st.image('../visualizations/feature_importance/shap_importance.png')
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please run feature_analysis.py first to generate the required data files.")

if __name__ == "__main__":
    main() 