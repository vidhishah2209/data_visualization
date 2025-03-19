# -*- coding: utf-8 -*-
"""Feature analysis and visualization script."""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_engineering.feature_generator import FeatureGenerator

def create_output_dirs():
    """Create output directories."""
    dirs = [
        '../data/processed/clean',
        '../data/processed/engineered',
        '../visualizations/distributions',
        '../visualizations/correlations',
        '../visualizations/clusters',
        '../visualizations/feature_importance',
        '../visualizations/pca'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def generate_features(data):
    """Generate features using the FeatureGenerator."""
    config = {
        'feature_engineering': {
            'use_llm': True,
            'llm_provider': 'gemini',
            'llm_model': 'gemini-pro',
            'text_embeddings': True,
            'polynomial_features': True,
            'clustering_features': True,
            'embedding_model': 'text-embedding-ada-002'
        }
    }
    
    column_types = {
        'numeric': ['numeric1', 'numeric2'],
        'text': ['text']
    }
    
    generator = FeatureGenerator(config)
    return generator.generate_features(data, column_types, is_training=True)

def separate_features(original_df, processed_df):
    """Separate original and engineered features."""
    # Keep original features in clean dataset
    clean_df = original_df.copy()
    
    # Get engineered features
    engineered_features = [col for col in processed_df.columns if col not in original_df.columns]
    engineered_df = processed_df[engineered_features]
    
    return clean_df, engineered_df

def plot_feature_distributions(df, output_dir, title_prefix=""):
    """Plot distributions for all numeric features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Matplotlib/Seaborn plots
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f"{title_prefix}Distribution of {col}")
        plt.savefig(f"{output_dir}/{col}_dist.png")
        plt.close()
    
    # Plotly interactive plots
    for col in numeric_cols:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[col], name=col))
        fig.add_trace(go.Histogram(x=df[col], name=f"{col} KDE"))
        fig.update_layout(title=f"{title_prefix}Interactive Distribution of {col}")
        fig.write_html(f"{output_dir}/{col}_dist_interactive.html")

def plot_correlation_matrix(df, output_path, title="Correlation Matrix"):
    """Plot correlation matrix."""
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Matplotlib/Seaborn correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{output_path}_static.png")
    plt.close()
    
    # Plotly interactive correlation matrix
    fig = px.imshow(numeric_df.corr(),
                    labels=dict(color="Correlation"),
                    title=title)
    fig.write_html(f"{output_path}_interactive.html")

def plot_cluster_analysis(df, output_dir):
    """Plot cluster analysis visualizations."""
    if 'cluster_label' in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # PCA for dimensionality reduction
        pca = PCA(n_components=2)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df[numeric_cols])
        data_pca = pca.fit_transform(data_scaled)
        
        # Static plot
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], 
                            c=df['cluster_label'], cmap='viridis')
        plt.colorbar(scatter)
        plt.title('Cluster Visualization (PCA)')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.savefig(f"{output_dir}/clusters_pca.png")
        plt.close()
        
        # Interactive plot
        fig = px.scatter(x=data_pca[:, 0], y=data_pca[:, 1],
                        color=df['cluster_label'].astype(str),
                        title='Interactive Cluster Visualization (PCA)')
        fig.write_html(f"{output_dir}/clusters_pca_interactive.html")

def plot_feature_importance(df, output_dir):
    """Plot feature importance using SHAP values."""
    try:
        import xgboost as xgb
        
        # Train a simple model
        X = df.select_dtypes(include=[np.number])
        y = X.iloc[:, 0]  # Using first column as dummy target
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
        model.fit(X, y)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Plot SHAP summary
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, show=False)
        plt.title('Feature Importance (SHAP)')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_importance.png")
        plt.close()
    except Exception as e:
        print(f"Error in feature importance plotting: {str(e)}")

def plot_pca_explained_variance(df, output_dir):
    """Plot PCA explained variance ratio."""
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Perform PCA
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(numeric_df)
    pca = PCA()
    pca.fit(data_scaled)
    
    # Create cumulative variance plot
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    
    # Static plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), 
             cumulative_variance_ratio, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Explained Variance Ratio')
    plt.grid(True)
    plt.savefig(f"{output_dir}/pca_variance.png")
    plt.close()
    
    # Interactive plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(cumulative_variance_ratio) + 1)),
                            y=cumulative_variance_ratio,
                            mode='lines+markers',
                            name='Cumulative Variance'))
    fig.update_layout(title='PCA Explained Variance Ratio (Interactive)',
                     xaxis_title='Number of Components',
                     yaxis_title='Cumulative Explained Variance Ratio')
    fig.write_html(f"{output_dir}/pca_variance_interactive.html")

def main():
    """Run feature analysis."""
    # Create output directories
    create_output_dirs()
    
    # Create sample data
    data = pd.DataFrame({
        'numeric1': np.random.randn(100),
        'numeric2': np.random.randn(100),
        'text': ['Sample text ' + str(i) for i in range(100)]
    })
    
    # Generate features
    print("Generating features...")
    processed_df = generate_features(data)
    
    # Separate features
    clean_df, engineered_df = separate_features(data, processed_df)
    
    # Save datasets
    clean_df.to_csv('../data/processed/clean/clean_data.csv', index=False)
    engineered_df.to_csv('../data/processed/engineered/engineered_features.csv', index=False)
    
    print("\nCreating visualizations...")
    
    # Plot distributions
    plot_feature_distributions(clean_df, '../visualizations/distributions', "Original ")
    plot_feature_distributions(engineered_df, '../visualizations/distributions', "Engineered ")
    
    # Plot correlation matrices
    plot_correlation_matrix(clean_df, '../visualizations/correlations/original_corr', 
                          "Original Features Correlation")
    plot_correlation_matrix(engineered_df, '../visualizations/correlations/engineered_corr',
                          "Engineered Features Correlation")
    
    # Plot cluster analysis
    plot_cluster_analysis(processed_df, '../visualizations/clusters')
    
    # Plot feature importance
    plot_feature_importance(processed_df, '../visualizations/feature_importance')
    
    # Plot PCA analysis
    plot_pca_explained_variance(processed_df, '../visualizations/pca')
    
    print("\nAnalysis complete! Check the following directories for results:")
    print("- Clean data: ../data/processed/clean/")
    print("- Engineered features: ../data/processed/engineered/")
    print("- Visualizations:")
    print("  - Distributions: ../visualizations/distributions/")
    print("  - Correlations: ../visualizations/correlations/")
    print("  - Clusters: ../visualizations/clusters/")
    print("  - Feature Importance: ../visualizations/feature_importance/")
    print("  - PCA Analysis: ../visualizations/pca/")
    
    # Print shapes
    print("\nDataset shapes:")
    print(f"Original data: {clean_df.shape}")
    print(f"Engineered features: {engineered_df.shape}")
    print(f"Total processed features: {processed_df.shape}")

if __name__ == '__main__':
    main() 