"""Simple feature generator for testing."""

import pandas as pd
import numpy as np
from typing import Dict

class SimpleFeatureGenerator:
    """A simplified version of the feature generator for testing."""
    
    def __init__(self, config: Dict):
        """Initialize with configuration."""
        self.config = config
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate simple features."""
        result = df.copy()
        
        # Add a simple feature
        if len(df.select_dtypes(include=['number']).columns) > 0:
            numeric_col = df.select_dtypes(include=['number']).columns[0]
            result[f'{numeric_col}_squared'] = df[numeric_col] ** 2
        
        return result 