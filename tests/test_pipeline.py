import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import yaml
from sklearn.datasets import make_classification
from pipeline import DataSciencePipeline
from preprocessing.data_cleaner import DataCleaner
from feature_engineering.advanced_features import AdvancedFeatureEngineering
from models.model_trainer import ModelTrainer
from visualization.advanced_plots import AdvancedVisualizer
from explainability.model_explainer import ModelExplainer

class TestDataSciencePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Load configuration
        with open('config.yaml', 'r') as f:
            cls.config = yaml.safe_load(f)
        
        # Create test data
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            random_state=42
        )
        
        # Create DataFrame with missing values and outliers
        feature_names = [f'feature_{i}' for i in range(10)]
        cls.test_data = pd.DataFrame(X, columns=feature_names)
        cls.test_data['target'] = y
        
        # Add missing values
        mask = np.random.rand(*cls.test_data.shape) < 0.1
        cls.test_data[mask] = np.nan
        
        # Add outliers
        for col in feature_names:
            outliers = np.random.rand(len(cls.test_data)) < 0.05
            cls.test_data.loc[outliers, col] = cls.test_data[col].mean() + np.random.randn(outliers.sum()) * 10
        
        # Create test directories
        cls.test_dirs = ['data/raw', 'data/processed', 'results']
        for dir_name in cls.test_dirs:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Remove test directories
        for dir_name in cls.test_dirs:
            shutil.rmtree(dir_name)
    
    def test_data_cleaner(self):
        """Test data cleaning functionality."""
        cleaner = DataCleaner(self.config)
        cleaned_data = cleaner.clean_data(self.test_data.copy())
        
        # Check if missing values are handled
        self.assertFalse(cleaned_data.isnull().any().any())
        
        # Check if data shape is preserved
        self.assertEqual(cleaned_data.shape, self.test_data.shape)
        
        # Check if target column is preserved
        self.assertIn('target', cleaned_data.columns)
    
    def test_feature_engineering(self):
        """Test feature engineering functionality."""
        engineer = AdvancedFeatureEngineering(self.config)
        engineered_data = engineer.transform(self.test_data.copy())
        
        # Check if original features are preserved
        for col in self.test_data.columns:
            self.assertIn(col, engineered_data.columns)
        
        # Check if new features are created
        self.assertGreater(len(engineered_data.columns), len(self.test_data.columns))
    
    def test_model_trainer(self):
        """Test model training functionality."""
        trainer = ModelTrainer(self.config)
        
        # Prepare data
        X = self.test_data.drop('target', axis=1)
        y = self.test_data['target']
        
        # Train models
        trainer.train_models(X, y)
        
        # Check if models are trained
        self.assertGreater(len(trainer.models), 0)
        
        # Check if metrics are generated
        self.assertGreater(len(trainer.metrics), 0)
    
    def test_visualizer(self):
        """Test visualization functionality."""
        visualizer = AdvancedVisualizer(self.config)
        
        # Generate visualizations
        visualizer.create_all_visualizations(self.test_data, 'target')
        
        # Check if visualization files are created
        viz_dir = Path(self.config['output']['visualization_dir'])
        self.assertGreater(len(list(viz_dir.glob('*.html'))), 0)
    
    def test_model_explainer(self):
        """Test model explanation functionality."""
        explainer = ModelExplainer(self.config)
        
        # Train a simple model for testing
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = self.test_data.drop('target', axis=1)
        y = self.test_data['target']
        model.fit(X, y)
        
        # Generate explanations
        explainer.explain_model(model, X, y)
        
        # Check if explanation files are created
        viz_dir = Path(self.config['output']['visualization_dir'])
        self.assertGreater(len(list(viz_dir.glob('*_explanation_*.html'))), 0)
    
    def test_end_to_end_pipeline(self):
        """Test the complete pipeline."""
        pipeline = DataSciencePipeline()
        
        # Save test data
        self.test_data.to_csv('data/raw/test_data.csv', index=False)
        
        # Run pipeline
        pipeline.run('data/raw/test_data.csv')
        
        # Check if results are generated
        results_dir = Path('results')
        self.assertTrue((results_dir / 'insights').exists())
        self.assertTrue((results_dir / 'visualizations').exists())
        self.assertTrue((results_dir / 'metrics').exists())
        self.assertTrue((results_dir / 'models').exists())

if __name__ == '__main__':
    unittest.main() 