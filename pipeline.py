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

class DataSciencePipeline:
    def __init__(self, config):
        """Initialize the pipeline with configuration."""
        self.config = config
        self.setup_logging()
        self.setup_gemini()
        self.visualizer = AdvancedVisualizer(config)
        
    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=self.config['logging']['level'],
            format=self.config['logging']['format'],
            filename=self.config['logging']['file']
        )
        self.logger = logging.getLogger(__name__)
        
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
            self.logger.info("LLM usage is disabled, using default recommendations")
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
            
            self.logger.info(f"Gemini recommendations: {recommendations}")
            return recommendations
            
        except Exception as e:
            self.logger.warning(f"Error getting Gemini recommendations: {str(e)}. Using default values.")
            return default_recommendations
    
    def load_data(self, file_path):
        """Load and validate data."""
        self.logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Validate target column
        target_col = self.config['data']['target_column']
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
            
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data based on configuration and Gemini recommendations."""
        self.logger.info("Starting data preprocessing")
        
        # Store target column
        target_col = self.config['data']['target_column']
        
        # Remove rows where target is missing
        df = df.dropna(subset=[target_col])
        
        # Store target column and convert to numeric if needed
        y = pd.to_numeric(df[target_col], errors='coerce')
        # Drop any remaining NaN values in target after conversion
        mask = ~y.isna()
        y = y[mask]
        X = df.drop(columns=[target_col]).loc[mask]
        
        # Get Gemini recommendations
        recommendations = self.get_gemini_recommendations(X)
        
        # Handle missing values first
        if self.config['preprocessing']['handle_missing_values']:
            strategy = self.config['preprocessing']['missing_value_strategy']
            if strategy == 'llm':
                # Use Gemini to suggest imputation strategy
                prompt = f"Suggest imputation strategy for missing values in this dataset: {X.isnull().sum()}"
                response = self.model.generate_content(prompt)
                strategy = response.text.strip().lower()
            
            if strategy == 'mean':
                X = X.fillna(X.mean())
            elif strategy == 'median':
                X = X.fillna(X.median())
            elif strategy == 'mode':
                X = X.fillna(X.mode().iloc[0])
            else:  # Default to mean if strategy is not recognized
                X = X.fillna(X.mean())
        else:
            # Always handle missing values with mean imputation if not explicitly configured
            X = X.fillna(X.mean())
        
        # Handle outliers
        if self.config['preprocessing']['handle_outliers']:
            method = self.config['preprocessing']['outlier_detection_method']
            if method == 'isolation_forest':
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outliers = iso_forest.fit_predict(X)
                # Keep track of indices and ensure X and y stay aligned
                X = X.reset_index(drop=True)
                y = pd.Series(y).reset_index(drop=True)
                mask = outliers == 1
                X = X[mask].reset_index(drop=True)
                y = y[mask].reset_index(drop=True)
        
        # Feature selection
        if self.config['preprocessing']['feature_selection']['method'] == 'variance_threshold':
            threshold = float(recommendations['threshold'])
            selector = VarianceThreshold(threshold=threshold)
            selector.fit(X)
            selected_features = X.columns[selector.get_support()].tolist()
            X = X[selected_features]
        
        # Scaling
        if self.config['preprocessing']['scaling']:
            scaling_method = recommendations['scaling_method']
            if scaling_method == 'standard':
                scaler = StandardScaler()
            elif scaling_method == 'minmax':
                scaler = MinMaxScaler()
            else:
                scaler = RobustScaler()
            
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        # Dimensionality reduction
        if self.config['preprocessing']['dimensionality_reduction']['method'] == 'pca':
            n_components = int(recommendations['n_components'])
            n_components = min(n_components, X.shape[1])  # Ensure n_components doesn't exceed number of features
            pca = PCA(n_components=n_components)
            X = pd.DataFrame(
                pca.fit_transform(X),
                columns=[f'PC{i+1}' for i in range(n_components)]
            )
        
        # Combine features and target
        df = X.copy()
        df[target_col] = y
        
        return df
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate models."""
        self.logger.info("Starting model training")
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Train Random Forest as an example
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        return model, metrics
    
    def save_results(self, model, metrics, df):
        """Save results and visualizations."""
        self.logger.info("Saving results")
        
        # Create necessary directories
        os.makedirs(self.config['output']['metrics_dir'], exist_ok=True)
        os.makedirs(self.config['output']['model_dir'], exist_ok=True)
        os.makedirs(self.config['output']['plots_dir'], exist_ok=True)
        os.makedirs(self.config['output']['visualization_dir'], exist_ok=True)
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(self.config['output']['metrics_dir'], 'metrics.csv'), index=False)
        
        # Save feature importance
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': df.drop(columns=[self.config['data']['target_column']]).columns,
                'importance': model.feature_importances_
            })
            importance_df.to_csv(os.path.join(self.config['output']['metrics_dir'], 'feature_importance.csv'), index=False)
            
        # Save model
        joblib.dump(model, os.path.join(self.config['output']['model_dir'], 'model.joblib'))
        
        # Create and save visualizations
        self.visualizer.create_all_visualizations(df, self.config['data']['target_column'])
    
    def run(self, data_path):
        """Run the complete pipeline."""
        try:
            # Load data
            df = self.load_data(data_path)
            
            # Preprocess data
            df = self.preprocess_data(df)
            
            # Save processed data
            os.makedirs(self.config['data']['output_path'], exist_ok=True)
            df.to_csv(os.path.join(self.config['data']['output_path'], 'cleaned_data.csv'), index=False)
            
            # Split data
            X = df.drop(columns=[self.config['data']['target_column']])
            y = df[self.config['data']['target_column']]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config['data']['test_size'],
                random_state=self.config['data']['random_state']
            )
            
            # Train models
            model, metrics = self.train_models(X_train, X_test, y_train, y_test)
            
            # Save results
            self.save_results(model, metrics, df)
            
            self.logger.info("Pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise 