# -*- coding: utf-8 -*-
"""Process any dataset with automatic feature engineering."""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_engineering.feature_generator import FeatureGenerator
from examples.feature_analysis import (create_output_dirs, plot_feature_distributions,
                                    plot_correlation_matrix, plot_cluster_analysis,
                                    plot_feature_importance, plot_pca_explained_variance,
                                    separate_features)

def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Automatically detect column types in the dataset."""
    column_types = {
        'numeric': [],
        'text': [],
        'categorical': [],
        'datetime': [],
        'boolean': []
    }
    
    for column in df.columns:
        # Check if numeric
        if pd.api.types.is_numeric_dtype(df[column]):
            if df[column].nunique() <= 10:  # Few unique values might be categorical
                column_types['categorical'].append(column)
            else:
                column_types['numeric'].append(column)
        # Check if datetime
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            column_types['datetime'].append(column)
        # Check if boolean
        elif df[column].dtype == bool:
            column_types['boolean'].append(column)
        # Check if categorical (few unique values)
        elif df[column].nunique() <= 20:
            column_types['categorical'].append(column)
        # Otherwise treat as text
        else:
            column_types['text'].append(column)
    
    return {k: v for k, v in column_types.items() if v}  # Remove empty types

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Basic preprocessing of the dataset."""
    # Handle missing values
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].mean())
        else:
            df[column] = df[column].fillna('missing')
    
    # Convert datetime columns to numeric features
    for column in df.select_dtypes(include=['datetime64']).columns:
        df[f'{column}_year'] = df[column].dt.year
        df[f'{column}_month'] = df[column].dt.month
        df[f'{column}_day'] = df[column].dt.day
        df[f'{column}_dayofweek'] = df[column].dt.dayofweek
        df = df.drop(columns=[column])
    
    return df

def main():
    """Process input dataset."""
    parser = argparse.ArgumentParser(description='Process a dataset with feature engineering')
    parser.add_argument('--input', type=str, help='Path to input CSV file', required=True)
    parser.add_argument('--target', type=str, help='Target column name (optional)')
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs('data/processed/clean', exist_ok=True)
    os.makedirs('data/processed/engineered', exist_ok=True)
    os.makedirs('visualizations/distributions', exist_ok=True)
    os.makedirs('visualizations/correlations', exist_ok=True)
    os.makedirs('visualizations/clusters', exist_ok=True)
    os.makedirs('visualizations/feature_importance', exist_ok=True)
    os.makedirs('visualizations/pca', exist_ok=True)
    
    # Load and preprocess data
    print(f"\nLoading dataset from {args.input}...")
    data = pd.read_csv(args.input)
    print(f"Original dataset shape: {data.shape}")
    
    # Save original data
    data.to_csv('data/raw/original_data.csv', index=False)
    
    # Preprocess data
    print("\nPreprocessing dataset...")
    processed_data = preprocess_dataset(data)
    
    # Detect column types
    print("\nDetecting column types...")
    column_types = detect_column_types(processed_data)
    print("\nDetected column types:")
    for type_name, columns in column_types.items():
        print(f"{type_name}: {len(columns)} columns")
        print(f"Examples: {columns[:5]}")
    
    # Configure feature generator
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
    
    # Generate features
    print("\nGenerating features...")
    generator = FeatureGenerator(config)
    processed_df = generator.generate_features(processed_data, column_types, is_training=True)
    
    # Separate features
    clean_df, engineered_df = separate_features(data, processed_df)
    
    # Save datasets
    clean_df.to_csv('data/processed/clean/clean_data.csv', index=False)
    engineered_df.to_csv('data/processed/engineered/engineered_features.csv', index=False)
    
    print("\nCreating visualizations...")
    
    # Plot distributions
    plot_feature_distributions(clean_df, 'visualizations/distributions', "Original ")
    plot_feature_distributions(engineered_df, 'visualizations/distributions', "Engineered ")
    
    # Plot correlation matrices
    plot_correlation_matrix(clean_df, 'visualizations/correlations/original_corr', 
                          "Original Features Correlation")
    plot_correlation_matrix(engineered_df, 'visualizations/correlations/engineered_corr',
                          "Engineered Features Correlation")
    
    # Plot cluster analysis
    plot_cluster_analysis(processed_df, 'visualizations/clusters')
    
    # Plot feature importance
    if args.target:
        target_data = data[args.target] if args.target in data else None
        if target_data is not None:
            plot_feature_importance(processed_df, 'visualizations/feature_importance')
    
    # Plot PCA analysis
    plot_pca_explained_variance(processed_df, 'visualizations/pca')
    
    print("\nAnalysis complete! Check the following directories for results:")
    print("- Original data: data/raw/original_data.csv")
    print("- Clean data: data/processed/clean/clean_data.csv")
    print("- Engineered features: data/processed/engineered/engineered_features.csv")
    print("- Visualizations:")
    print("  - Distributions: visualizations/distributions/")
    print("  - Correlations: visualizations/correlations/")
    print("  - Clusters: visualizations/clusters/")
    print("  - Feature Importance: visualizations/feature_importance/")
    print("  - PCA Analysis: visualizations/pca/")
    
    # Print shapes
    print("\nDataset shapes:")
    print(f"Original data: {clean_df.shape}")
    print(f"Engineered features: {engineered_df.shape}")
    print(f"Total processed features: {processed_df.shape}")
    
    print("\nTo view the interactive dashboard, run:")
    print("streamlit run examples/dashboard.py")

if __name__ == '__main__':
    main() 