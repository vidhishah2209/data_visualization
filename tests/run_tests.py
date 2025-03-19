import unittest
import sys
from pathlib import Path
import shutil
import logging

def setup_test_environment():
    """Set up the test environment."""
    # Create necessary directories
    test_dirs = ['data/raw', 'data/processed', 'results']
    for dir_name in test_dirs:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('test_pipeline.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def cleanup_test_environment():
    """Clean up the test environment."""
    # Remove test directories
    test_dirs = ['data/raw', 'data/processed', 'results']
    for dir_name in test_dirs:
        if Path(dir_name).exists():
            shutil.rmtree(dir_name)

def run_tests():
    """Run all tests."""
    # Set up test environment
    setup_test_environment()
    
    try:
        # Discover and run tests
        loader = unittest.TestLoader()
        start_dir = Path(__file__).parent
        suite = loader.discover(start_dir, pattern='test_*.py')
        
        # Run tests with verbosity
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Return success status
        return result.wasSuccessful()
    
    finally:
        # Clean up test environment
        cleanup_test_environment()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1) 