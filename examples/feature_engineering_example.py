# -*- coding: utf-8 -*-
"""Example script for using the feature engineering module."""

import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engineering.feature_generator import FeatureGenerator

def main():
    """Run feature engineering example."""
    
    # Create sample data (replace this with your actual data)
    data = pd.DataFrame({
        'numeric1': np.random.randn(100),
        'numeric2': np.random.randn(100),
        'text': ['Sample text ' + str(i) for i in range(100)]
    })
    
    # Save raw data
    os.makedirs('../data/raw', exist_ok=True)
    data.to_csv('../data/raw/sample_data.csv', index=False)
    
    # Define configuration
    config = {
        'feature_engineering': {
            'use_llm': True,
            'llm_provider': 'gemini',  # Using Gemini as LLM provider
            'llm_model': 'gemini-pro',
            'text_embeddings': True,
            'polynomial_features': True,
            'clustering_features': True,
            'embedding_model': 'text-embedding-ada-002'
        }
    }
    
    # Define column types
    column_types = {
        'numeric': ['numeric1', 'numeric2'],
        'text': ['text']
    }
    
    # Initialize feature generator
    generator = FeatureGenerator(config)
    
    # Generate features
    print("Generating features...")
    result_df = generator.generate_features(
        df=data,
        column_types=column_types,
        is_training=True
    )
    
    # Save processed data
    os.makedirs('../data/processed', exist_ok=True)
    result_df.to_csv('../data/processed/processed_data.csv', index=False)
    
    print("\nFeature generation complete!")
    print(f"Original shape: {data.shape}")
    print(f"Processed shape: {result_df.shape}")
    print("\nGenerated features:")
    for col in result_df.columns:
        if col not in data.columns:
            print(f"- {col}")

if __name__ == '__main__':
    main() 