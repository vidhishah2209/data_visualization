"""Test minimal feature generator."""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engineering.feature_generator import FeatureGenerator

class TestMinimalFeatureGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.config = {
            'feature_engineering': {
                'polynomial_features': True
            }
        }
        
        # Create sample data
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'numeric1': [1, 2, 3],
            'numeric2': [4, 5, 6]
        })
        
        # Define column types
        self.column_types = {
            'numeric': ['numeric1', 'numeric2']
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
        self.assertTrue(any('1_2' in col for col in result_df.columns))
        
        print("Feature generation test completed successfully")

if __name__ == '__main__':
    unittest.main() 