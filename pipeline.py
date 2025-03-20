import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import google.generativeai as genai
import logging
from pathlib import Path
import joblib
from visualization.advanced_plots import AdvancedVisualizer
from feature_engineering.feature_generator import FeatureGenerator
from preprocessing.data_cleaner import DataCleaner
from preprocessing.data_processor import DataProcessor
from models.model_trainer import ModelTrainer
from models.model_explainer import ModelExplainer
from typing import Dict, Any, Tuple
import yaml

logger = logging.getLogger(__name__)

class DataSciencePipeline:
    def __init__(self, config_path: str):
        """Initialize the pipeline with configuration."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.setup_gemini()
        
        # Initialize components
        self.visualizer = AdvancedVisualizer(self.config)
        self.feature_generator = FeatureGenerator(self.config)
        self.data_cleaner = DataCleaner(self.config)
        self.data_processor = DataProcessor(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.model_explainer = ModelExplainer(self.config)
        
        # Setup output directories
        self.output_dir = Path(self.config.get('output_dir', 'results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
        
    def setup_logging(self):
        """Set up logging configuration."""
        log_dir = Path(self.config.get('log_dir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'pipeline.log'),
                logging.StreamHandler()
            ]
        )
        
    def setup_gemini(self):
        """Set up Gemini API."""
        genai.configure(api_key=self.config['llm_api']['gemini_api_key'])
        self.model = genai.GenerativeModel(self.config['feature_engineering']['llm_model'])
        
    def get_gemini_recommendations(self, df):
        """Get preprocessing recommendations from Gemini."""
        # Default recommendations
        default_recommendations = {
            'scaling_method': 'standard',
            'threshold': '0.01',
            'n_components': str(min(3, df.shape[1]))  # Ensure n_components doesn't exceed number of features
        }
        
        if not self.config['llm_api'].get('use_llm', True):  # Check if LLM usage is enabled
            logger.info("LLM usage is disabled, using default recommendations")
            return default_recommendations
            
        prompt = f"""
        Analyze this dataset and recommend preprocessing parameters:
        - Number of features: {df.shape[1]}
        - Number of samples: {df.shape[0]}
        - Missing values: {df.isnull().sum().sum()}
        - Feature types: {df.dtypes}
        
        Please recommend:
        1. Scaling method (standard, minmax, or robust)
        2. Feature selection threshold (0.0 to 1.0)
        3. Number of PCA components (1 to {df.shape[1]})
        
        Format your response EXACTLY as follows (one parameter per line):
        scaling_method: standard
        threshold: 0.01
        n_components: 3
        """
        
        try:
            response = self.model.generate_content(prompt)
            recommendations = default_recommendations.copy()  # Start with defaults
            
            # Parse response
            for line in response.text.strip().split('\n'):
                line = line.strip()
                if ':' in line:
                    parts = line.split(':', 1)  # Split on first occurrence only
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        if key in recommendations:
                            recommendations[key] = value
            
            logger.info(f"Gemini recommendations: {recommendations}")
            return recommendations
            
        except Exception as e:
            logger.warning(f"Error getting Gemini recommendations: {str(e)}. Using default values.")
            return default_recommendations
    
    def load_data(self) -> pd.DataFrame:
        """Load data from configured source."""
        data_path = self.config['data']['input_path']
        logger.info(f"Loading data from {data_path}")
        
        try:
            data = pd.read_csv(data_path)
            logger.info(f"Loaded data with shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess the data using all preprocessing components."""
        logger.info("Starting data preprocessing")
        
        # Clean data
        cleaned_data = self.data_cleaner.clean_data(data)
        
        # Process data
        processed_data = self.data_processor.process_data(cleaned_data)
        
        # Generate additional features
        if self.config.get('feature_engineering', {}).get('enabled', False):
            processed_data = self.feature_generator.generate_features(processed_data)
        
        # Split features and target
        target_col = self.config['data']['target_column']
        X = processed_data.drop(columns=[target_col])
        y = processed_data[target_col]
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Train and evaluate models."""
        logger.info("Starting model training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train models
        models, metrics = self.model_trainer.train_models(
            X_train, X_test, y_train, y_test
        )
        
        return models, metrics
    
    def generate_explanations(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Generate model explanations."""
        logger.info("Starting explanation generation")
        
        explanations = self.model_explainer.generate_explanations(models, X, y)
        return explanations
    
    def create_visualizations(self, data: pd.DataFrame, models: Dict[str, Any], 
                            explanations: Dict[str, Any]) -> None:
        """Create visualizations using the advanced visualizer."""
        logger.info("Creating visualizations")
        
        self.visualizer.create_all_plots(data, models, explanations)
    
    def save_results(self, models: Dict[str, Any], metrics: Dict[str, float], 
                    explanations: Dict[str, Any], cleaned_data: pd.DataFrame) -> None:
        """Save all results."""
        logger.info("Saving results")
        
        # Save cleaned data
        cleaned_data.to_csv(self.output_dir / 'cleaned_data.csv', index=False)
        
        # Save metrics
        pd.DataFrame([metrics]).to_csv(self.output_dir / 'metrics.csv', index=False)
        
        # Save model explanations
        self.model_explainer.save_explanations(explanations)
        
        # Save models if configured
        if self.config.get('save_models', True):
            model_dir = self.output_dir / 'models'
            model_dir.mkdir(exist_ok=True)
            for name, model in models.items():
                joblib.dump(model, model_dir / f'{name}.joblib')
    
    def run(self):
        """Run the complete pipeline."""
        try:
            # Load data
            data = self.load_data()
            
            # Preprocess data
            X, y = self.preprocess_data(data)
            
            # Train models
            models, metrics = self.train_models(X, y)
            
            # Generate explanations
            explanations = self.generate_explanations(models, X, y)
            
            # Create visualizations
            self.create_visualizations(data, models, explanations)
            
            # Save results
            self.save_results(models, metrics, explanations, data)
            
            logger.info("Pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

if __name__ == "__main__":
    pipeline = DataSciencePipeline("config.yaml")
    pipeline.run() 