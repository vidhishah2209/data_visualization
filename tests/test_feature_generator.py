# -*- coding: utf-8 -*-
"""Test feature generator module."""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engineering.feature_generator import FeatureGenerator

class TestFeatureGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.config = {
            'feature_engineering': {
                'polynomial_features': True,
                'clustering_features': True
            }
        }
        
        # Create sample data
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'numeric1': np.random.randn(100),
            'numeric2': np.random.randn(100),
            'numeric3': np.random.randn(100)
        })
        
        # Define column types
        self.column_types = {
            'numeric': ['numeric1', 'numeric2', 'numeric3']
        }
        
        # Initialize feature generator
        self.feature_generator = FeatureGenerator(self.config)
    
    def test_generate_features(self):
        """Test feature generation pipeline."""
        # Generate features
        result_df = self.feature_generator.generate_features(
            self.sample_data,
            self.column_types,
            is_training=True
        )
        
        # Check if result is a DataFrame
        self.assertIsInstance(result_df, pd.DataFrame)
        
        # Check if new features were generated
        self.assertTrue(len(result_df.columns) > len(self.sample_data.columns))
        
        # Check if polynomial features exist
        self.assertTrue(any('numeric1 numeric2' in col for col in result_df.columns))
        
        # Check if clustering features exist
        self.assertIn('cluster_label', result_df.columns)
        self.assertTrue(any('cluster_distance' in col for col in result_df.columns))
        
        print("Feature generation test completed successfully")
    
    def test_generate_polynomial_features(self):
        """Test polynomial feature generation."""
        result_df = self.feature_generator.generate_polynomial_features(
            self.sample_data,
            self.column_types['numeric'],
            is_training=True
        )
        
        # Check if polynomial features were generated
        self.assertTrue(len(result_df.columns) > len(self.sample_data.columns))
        self.assertTrue(any('numeric1 numeric2' in col for col in result_df.columns))
        
        print("Polynomial feature generation test completed successfully")
    
    def test_generate_clustering_features(self):
        """Test clustering feature generation."""
        result_df = self.feature_generator.generate_clustering_features(
            self.sample_data,
            self.column_types['numeric'],
            is_training=True
        )
        
        # Check if clustering features were generated
        self.assertIn('cluster_label', result_df.columns)
        self.assertTrue(any('cluster_distance' in col for col in result_df.columns))
        
        print("Clustering feature generation test completed successfully")

if __name__ == '__main__':
    unittest.main() 