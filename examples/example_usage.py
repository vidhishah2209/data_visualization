import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from pipeline import DataSciencePipeline

def create_sample_dataset(n_samples: int = 1000, n_features: int = 20) -> pd.DataFrame:
    """Create a sample dataset for demonstration."""
    # Generate synthetic data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add some missing values
    mask = np.random.rand(*df.shape) < 0.05
    df[mask] = np.nan
    
    # Add some outliers
    for col in feature_names:
        outliers = np.random.rand(len(df)) < 0.02
        df.loc[outliers, col] = df[col].mean() + np.random.randn(outliers.sum()) * 10
    
    return df

def main():
    """Demonstrate the usage of the data science pipeline."""
    # Create sample dataset
    print("Creating sample dataset...")
    data = create_sample_dataset()
    
    # Save sample dataset
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(data_dir / 'sample_dataset.csv', index=False)
    
    # Initialize and run pipeline
    print("Initializing pipeline...")
    pipeline = DataSciencePipeline()
    
    print("Running pipeline...")
    pipeline.run('data/raw/sample_dataset.csv')
    
    print("Pipeline completed successfully!")
    print("\nResults are saved in the 'results' directory:")
    print("- Data insights: results/insights/")
    print("- Visualizations: results/visualizations/")
    print("- Model metrics: results/metrics/")
    print("- Saved models: results/models/")

if __name__ == "__main__":
    main() 