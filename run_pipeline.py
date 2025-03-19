import yaml
from pipeline import DataSciencePipeline
import os

def main():
    # Load configuration
    print("Loading configuration...")
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = DataSciencePipeline(config)
    
    # Run pipeline
    print("Running pipeline...")
    pipeline.run(config['data']['input_path'])
    
    print("Pipeline completed successfully!")
    print("Results can be found in the results directory.")

if __name__ == "__main__":
    main() 