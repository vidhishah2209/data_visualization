import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import logging
from pathlib import Path
import joblib
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data cleaner with configuration."""
        self.config = config
        self.scaler = None
        self.imputer = None
        self.variance_threshold = None
        self.pca = None
        self.feature_names = None
        self.preprocessing_config = config.get('preprocessing', {})
        self.label_encoders = {}
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all cleaning and preprocessing steps."""
        logger.info("Starting data cleaning pipeline")
        
        # Store original feature names
        self.feature_names = data.columns.tolist()
        
        # Identify numeric and categorical columns
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        # Handle missing values
        if self.preprocessing_config.get('handle_missing_values', True):
            strategy = self.preprocessing_config.get('missing_value_strategy', 'mean')
            data = self._handle_missing_values(data, strategy, numeric_cols, categorical_cols)
        
        # Handle outliers (only for numeric columns)
        if self.preprocessing_config.get('handle_outliers', True):
            method = self.preprocessing_config.get('outlier_detection_method', 'isolation_forest')
            data[numeric_cols] = self._handle_outliers(data[numeric_cols], method)
        
        # Encode categorical variables
        data = self._encode_categorical_variables(data, categorical_cols)
        
        # Scale numeric features
        data[numeric_cols] = self._scale_features(data[numeric_cols])
        
        # Apply feature selection
        data = self._select_features(data)
        
        # Apply dimensionality reduction if configured
        if self.config['preprocessing'].get('dimensionality_reduction', {}).get('enabled', False):
            data = self._reduce_dimensions(data)
        
        logger.info(f"Data cleaning completed. Final shape: {data.shape}")
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame, strategy: str, 
                             numeric_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
        """Handle missing values using specified strategy."""
        logger.info("Handling missing values")
        
        # Handle numeric columns
        if len(numeric_cols) > 0:
            if strategy == 'mean':
                data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
            elif strategy == 'median':
                data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
            else:  # Default to mean
                data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        # Handle categorical columns
        if len(categorical_cols) > 0:
            data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """Handle outliers using specified method."""
        logger.info("Handling outliers")
        
        if method == 'isolation_forest':
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(data)
            mask = outliers == 1
            data = data[mask].reset_index(drop=True)
        
        return data
    
    def _encode_categorical_variables(self, data: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """Encode categorical variables using label encoding."""
        logger.info("Encoding categorical variables")
        
        for col in categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            self.label_encoders[col] = le
        
        return data
    
    def _scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features using configured method."""
        logger.info("Scaling features")
        
        scaling_method = self.config['preprocessing'].get('scaling_method', 'standard')
        
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        scaled_data = self.scaler.fit_transform(data)
        return pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
    
    def _select_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection techniques."""
        logger.info("Applying feature selection")
        
        selection_config = self.config['preprocessing'].get('feature_selection', {})
        
        if selection_config.get('method') == 'variance_threshold':
            self.variance_threshold = VarianceThreshold(
                threshold=selection_config.get('threshold', 0.01)
            )
            selected_features = self.variance_threshold.fit_transform(data)
            
            # Get selected feature names
            feature_names = data.columns[self.variance_threshold.get_support()].tolist()
            return pd.DataFrame(selected_features, columns=feature_names, index=data.index)
        
        return data
    
    def _reduce_dimensions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply dimensionality reduction techniques."""
        logger.info("Applying dimensionality reduction")
        
        dim_reduction_config = self.config['preprocessing'].get('dimensionality_reduction', {})
        
        if dim_reduction_config.get('method') == 'pca':
            n_components = dim_reduction_config.get('n_components', min(data.shape[1], 3))
            self.pca = PCA(n_components=n_components)
            reduced_features = self.pca.fit_transform(data)
            
            # Create feature names
            feature_names = [f"PC{i+1}" for i in range(n_components)]
            return pd.DataFrame(reduced_features, columns=feature_names, index=data.index)
        
        return data
    
    def save_preprocessors(self, output_dir: str) -> None:
        """Save preprocessor objects for later use."""
        logger.info("Saving preprocessor objects")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.scaler is not None:
            joblib.dump(self.scaler, output_path / 'scaler.joblib')
        if self.imputer is not None:
            joblib.dump(self.imputer, output_path / 'imputer.joblib')
        if self.variance_threshold is not None:
            joblib.dump(self.variance_threshold, output_path / 'variance_threshold.joblib')
        if self.pca is not None:
            joblib.dump(self.pca, output_path / 'pca.joblib')
        if self.label_encoders:
            joblib.dump(self.label_encoders, output_path / 'label_encoders.joblib')
    
    def load_preprocessors(self, input_dir: str) -> None:
        """Load preprocessor objects from disk."""
        logger.info("Loading preprocessor objects")
        
        input_path = Path(input_dir)
        
        if (input_path / 'scaler.joblib').exists():
            self.scaler = joblib.load(input_path / 'scaler.joblib')
        if (input_path / 'imputer.joblib').exists():
            self.imputer = joblib.load(input_path / 'imputer.joblib')
        if (input_path / 'variance_threshold.joblib').exists():
            self.variance_threshold = joblib.load(input_path / 'variance_threshold.joblib')
        if (input_path / 'pca.joblib').exists():
            self.pca = joblib.load(input_path / 'pca.joblib')
        if (input_path / 'label_encoders.joblib').exists():
            self.label_encoders = joblib.load(input_path / 'label_encoders.joblib') 