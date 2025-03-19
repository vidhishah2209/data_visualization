import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engineering.simple_generator import SimpleFeatureGenerator

class TestSimpleFeatureGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.config = {'test': True}
        self.generator = SimpleFeatureGenerator(self.config)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'category': ['A', 'B', 'A', 'B', 'A']
        })
    
    def test_generate_features(self):
        """Test feature generation."""
        result = self.generator.generate_features(self.sample_data)
        
        # Check if result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check if new feature was added
        self.assertIn('numeric_squared', result.columns)
        
        # Check if squared values are correct
        np.testing.assert_array_equal(
            result['numeric_squared'],
            self.sample_data['numeric'] ** 2
        )
        
        print("Simple feature generation test completed successfully")

if __name__ == '__main__':
    unittest.main() 