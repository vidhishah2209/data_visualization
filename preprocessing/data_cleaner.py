import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import logging
from pathlib import Path
import joblib

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
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all cleaning and preprocessing steps."""
        logger.info("Starting data cleaning pipeline")
        
        # Store original feature names
        self.feature_names = data.columns.tolist()
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Handle outliers
        data = self._handle_outliers(data)
        
        # Scale features
        data = self._scale_features(data)
        
        # Apply feature selection
        data = self._select_features(data)
        
        # Apply dimensionality reduction if configured
        if self.config['preprocessing']['dimensionality_reduction']['enabled']:
            data = self._reduce_dimensions(data)
        
        logger.info(f"Data cleaning completed. Final shape: {data.shape}")
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using configured strategy."""
        logger.info("Handling missing values")
        
        missing_strategy = self.config['preprocessing']['missing_value_strategy']
        
        if missing_strategy == 'knn':
            self.imputer = KNNImputer(n_neighbors=5)
            data_imputed = self.imputer.fit_transform(data)
            return pd.DataFrame(data_imputed, columns=data.columns, index=data.index)
        
        elif missing_strategy == 'llm':
            # Use LLM-based imputation for text columns
            text_cols = data.select_dtypes(include=['object']).columns
            numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
            
            # Handle numerical columns with KNN
            if len(numerical_cols) > 0:
                self.imputer = KNNImputer(n_neighbors=5)
                data[numerical_cols] = self.imputer.fit_transform(data[numerical_cols])
            
            # Handle text columns with mode
            for col in text_cols:
                data[col].fillna(data[col].mode()[0], inplace=True)
            
            return data
        
        else:
            # Use simple imputation strategies
            self.imputer = SimpleImputer(strategy=missing_strategy)
            data_imputed = self.imputer.fit_transform(data)
            return pd.DataFrame(data_imputed, columns=data.columns, index=data.index)
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using configured method."""
        logger.info("Handling outliers")
        
        outlier_method = self.config['preprocessing']['outlier_detection_method']
        numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
        
        if outlier_method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(data[numerical_cols])
            
            # Replace outliers with median
            for col in numerical_cols:
                data.loc[outliers == -1, col] = data[col].median()
        
        elif outlier_method == 'z_score':
            for col in numerical_cols:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                data.loc[z_scores > 3, col] = data[col].median()
        
        elif outlier_method == 'iqr':
            for col in numerical_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data.loc[(data[col] < lower_bound) | (data[col] > upper_bound), col] = data[col].median()
        
        return data
    
    def _scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features using configured method."""
        logger.info("Scaling features")
        
        scaling_method = self.config['preprocessing']['scaling_method']
        numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
        
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        
        data[numerical_cols] = self.scaler.fit_transform(data[numerical_cols])
        return data
    
    def _select_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection techniques."""
        logger.info("Applying feature selection")
        
        selection_config = self.config['preprocessing']['feature_selection']
        
        if selection_config['method'] == 'variance_threshold':
            self.variance_threshold = VarianceThreshold(
                threshold=selection_config['threshold']
            )
            selected_features = self.variance_threshold.fit_transform(data)
            
            # Get selected feature names
            feature_names = data.columns[self.variance_threshold.get_support()].tolist()
            return pd.DataFrame(selected_features, columns=feature_names, index=data.index)
        
        return data
    
    def _reduce_dimensions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply dimensionality reduction techniques."""
        logger.info("Applying dimensionality reduction")
        
        dim_reduction_config = self.config['preprocessing']['dimensionality_reduction']
        
        if dim_reduction_config['method'] == 'pca':
            self.pca = PCA(n_components=dim_reduction_config['n_components'])
            reduced_features = self.pca.fit_transform(data)
            
            # Create feature names
            feature_names = [f"PC{i+1}" for i in range(dim_reduction_config['n_components'])]
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