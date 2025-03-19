import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from feature_engineering.feature_generator import FeatureGenerator
    print("Successfully imported FeatureGenerator")
except Exception as e:
    print(f"Error importing FeatureGenerator: {str(e)}")
    raise 