import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple, Any, Optional
import logging
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.ensemble import IsolationForest
from scipy import stats
import warnings
from datetime import datetime
import re
import json
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import timer_decorator, logger, parallelize_dataframe, get_optimal_dtypes

class DataProcessor:
    """
    Class for automated data preprocessing.
    
    Features:
    - Auto-detection of data types
    - Missing value imputation
    - Outlier detection and handling
    - Feature scaling
    - Categorical encoding
    - Datetime feature extraction
    - Text cleaning
    - Feature selection
    - Dimensionality reduction
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the DataProcessor with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.preprocessing_config = config.get('preprocessing', {})
        self.data_config = config.get('data', {})
        self.general_config = config.get('general', {})
        
        # Initialize attributes
        self.categorical_encoders = {}
        self.scalers = {}
        self.imputers = {}
        self.outlier_detectors = {}
        self.column_types = {}
        self.transformations_applied = {}
        
        # Set up parallel processing
        self.n_jobs = self.general_config.get('n_jobs', -1)
        
        # Suppress warnings
        warnings.filterwarnings('ignore')
        
        self.scaler = None
        self.pca = None
        self.feature_selector = None
    
    @timer_decorator
    def process_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Main method to process the data.
        
        Args:
            df: Input DataFrame
            is_training: Whether this is training data or not
            
        Returns:
            Processed DataFrame
        """
        logger.info(f"Starting data processing on DataFrame with shape {df.shape}")
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Store target column if it exists
        target_col = self.data_config.get('target_column')
        if target_col and target_col in result_df.columns:
            target_values = result_df[target_col].copy()
        
        # Auto-detect column types if enabled
        if self.preprocessing_config.get('auto_detect_types', True):
            result_df = self.detect_column_types(result_df)
        
        # Handle missing values if enabled
        if self.preprocessing_config.get('handle_missing_values', True):
            result_df = self.handle_missing_values(result_df, is_training)
        
        # Handle outliers if enabled
        if self.preprocessing_config.get('handle_outliers', True):
            result_df = self.handle_outliers(result_df, is_training)
        
        # Extract datetime features if enabled
        if self.preprocessing_config.get('datetime_features', True):
            result_df = self.extract_datetime_features(result_df)
        
        # Clean text if enabled
        if self.preprocessing_config.get('text_cleaning', True):
            result_df = self.clean_text_columns(result_df)
        
        # Encode categorical variables if enabled
        if self.preprocessing_config.get('categorical_encoding', 'auto') != 'none':
            result_df = self.encode_categorical(result_df, is_training)
        
        # Feature selection
        if self.preprocessing_config.get('feature_selection', {}).get('method') == 'variance_threshold':
            result_df = self._apply_feature_selection(result_df)
        
        # Scale numerical features if enabled
        if self.preprocessing_config.get('scaling', True):
            result_df = self._apply_scaling(result_df)
        
        # Dimensionality reduction
        if self.preprocessing_config.get('dimensionality_reduction', {}).get('method') == 'pca':
            result_df = self._apply_pca(result_df)
        
        # Restore target column if it exists
        if target_col and target_col in df.columns:
            # Convert target to binary (0/1) if it's not already
            if target_values.nunique() == 2:
                target_values = (target_values == target_values.max()).astype(int)
            result_df[target_col] = target_values
        
        # Optimize memory usage
        result_df = get_optimal_dtypes(result_df)
        
        logger.info(f"Data processing completed. Output shape: {result_df.shape}")
        
        return result_df
    
    def detect_column_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Automatically detect and categorize column types.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with detected column types
        """
        logger.info("Detecting column types...")
        
        # Initialize column type dictionaries
        self.column_types = {
            'numeric': [],
            'categorical': [],
            'datetime': [],
            'text': [],
            'binary': [],
            'id': [],
            'target': []
        }
        
        for col in df.columns:
            # Check if column name contains target-related keywords
            if any(keyword in col.lower() for keyword in ['target', 'label', 'class', 'y_']):
                self.column_types['target'].append(col)
                continue
                
            # Check if column name contains ID-related keywords
            if any(keyword in col.lower() for keyword in ['id', 'key', 'uuid', 'index']):
                self.column_types['id'].append(col)
                continue
            
            # Check data type
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if binary
                if set(df[col].dropna().unique()).issubset({0, 1}):
                    self.column_types['binary'].append(col)
                else:
                    self.column_types['numeric'].append(col)
            
            elif pd.api.types.is_datetime64_dtype(df[col]):
                self.column_types['datetime'].append(col)
            
            elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                # Try to convert to datetime
                try:
                    pd.to_datetime(df[col])
                    self.column_types['datetime'].append(col)
                    df[col] = pd.to_datetime(df[col])
                    continue
                except:
                    pass
                
                # Check if text or categorical
                if df[col].dropna().apply(lambda x: isinstance(x, str) and len(x) > 100).any():
                    self.column_types['text'].append(col)
                else:
                    # Check cardinality to determine if categorical
                    cardinality = df[col].nunique() / len(df)
                    if cardinality < 0.05:  # Less than 5% unique values
                        self.column_types['categorical'].append(col)
                    else:
                        self.column_types['text'].append(col)
        
        logger.info(f"Column types detected: {json.dumps({k: len(v) for k, v in self.column_types.items()})}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.
        
        Args:
            df: Input DataFrame
            is_training: Whether this is training data or not
            
        Returns:
            DataFrame with handled missing values
        """
        logger.info("Handling missing values...")
        
        # Get missing value strategy from config
        strategy = self.preprocessing_config.get('missing_value_strategy', 'mean')
        
        # Process each column type separately
        result_df = df.copy()
        
        # Handle numeric columns
        numeric_cols = self.column_types.get('numeric', [])
        if numeric_cols and any(result_df[numeric_cols].isnull().sum()):
            if is_training:
                if strategy == 'knn':
                    imputer = KNNImputer(n_neighbors=5)
                    self.imputers['numeric'] = imputer
                else:
                    imputer = SimpleImputer(strategy=strategy)
                    self.imputers['numeric'] = imputer
                
                # Fit and transform
                result_df[numeric_cols] = self.imputers['numeric'].fit_transform(result_df[numeric_cols])
            else:
                # Use pre-fitted imputer
                if 'numeric' in self.imputers:
                    result_df[numeric_cols] = self.imputers['numeric'].transform(result_df[numeric_cols])
        
        # Handle categorical columns
        cat_cols = self.column_types.get('categorical', [])
        if cat_cols and any(result_df[cat_cols].isnull().sum()):
            if is_training:
                imputer = SimpleImputer(strategy='most_frequent')
                self.imputers['categorical'] = imputer
                
                # Fit and transform
                result_df[cat_cols] = self.imputers['categorical'].fit_transform(result_df[cat_cols])
            else:
                # Use pre-fitted imputer
                if 'categorical' in self.imputers:
                    result_df[cat_cols] = self.imputers['categorical'].transform(result_df[cat_cols])
        
        # Handle datetime columns
        datetime_cols = self.column_types.get('datetime', [])
        for col in datetime_cols:
            if result_df[col].isnull().any():
                # Fill with median date
                if is_training:
                    median_date = result_df[col].dropna().median()
                    self.imputers[col] = median_date
                else:
                    median_date = self.imputers.get(col)
                
                if median_date is not None:
                    result_df[col] = result_df[col].fillna(median_date)
        
        # Handle text columns (fill with empty string)
        text_cols = self.column_types.get('text', [])
        for col in text_cols:
            if result_df[col].isnull().any():
                result_df[col] = result_df[col].fillna('')
        
        # Handle binary columns (fill with most frequent)
        binary_cols = self.column_types.get('binary', [])
        if binary_cols and any(result_df[binary_cols].isnull().sum()):
            if is_training:
                imputer = SimpleImputer(strategy='most_frequent')
                self.imputers['binary'] = imputer
                
                # Fit and transform
                result_df[binary_cols] = self.imputers['binary'].fit_transform(result_df[binary_cols])
            else:
                # Use pre-fitted imputer
                if 'binary' in self.imputers:
                    result_df[binary_cols] = self.imputers['binary'].transform(result_df[binary_cols])
        
        logger.info(f"Missing values handled. Remaining missing: {result_df.isnull().sum().sum()}")
        
        return result_df
    
    def handle_outliers(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Detect and handle outliers in the DataFrame.
        
        Args:
            df: Input DataFrame
            is_training: Whether this is training data or not
            
        Returns:
            DataFrame with handled outliers
        """
        logger.info("Handling outliers...")
        
        # Get outlier detection method from config
        method = self.preprocessing_config.get('outlier_detection_method', 'isolation_forest')
        
        # Only process numeric columns
        numeric_cols = self.column_types.get('numeric', [])
        if not numeric_cols:
            logger.info("No numeric columns to handle outliers.")
            return df
        
        result_df = df.copy()
        
        if method == 'z-score':
            # Z-score method
            for col in numeric_cols:
                if is_training:
                    # Calculate z-scores
                    z_scores = np.abs(stats.zscore(result_df[col].dropna()))
                    threshold = 3
                    self.outlier_detectors[col] = {'mean': result_df[col].mean(), 'std': result_df[col].std(), 'threshold': threshold}
                
                    # Replace outliers with mean
                    outlier_indices = np.where(z_scores > threshold)[0]
                    result_df.loc[result_df.index[outlier_indices], col] = result_df[col].mean()
                else:
                    # Use pre-calculated parameters
                    if col in self.outlier_detectors:
                        params = self.outlier_detectors[col]
                        z_scores = np.abs((result_df[col] - params['mean']) / params['std'])
                        outlier_indices = np.where(z_scores > params['threshold'])[0]
                        result_df.loc[result_df.index[outlier_indices], col] = params['mean']
        
        elif method == 'iqr':
            # IQR method
            for col in numeric_cols:
                if is_training:
                    # Calculate IQR
                    Q1 = result_df[col].quantile(0.25)
                    Q3 = result_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    self.outlier_detectors[col] = {'lower': lower_bound, 'upper': upper_bound, 'median': result_df[col].median()}
                
                    # Replace outliers with median
                    result_df.loc[result_df[col] < lower_bound, col] = result_df[col].median()
                    result_df.loc[result_df[col] > upper_bound, col] = result_df[col].median()
                else:
                    # Use pre-calculated parameters
                    if col in self.outlier_detectors:
                        params = self.outlier_detectors[col]
                        result_df.loc[result_df[col] < params['lower'], col] = params['median']
                        result_df.loc[result_df[col] > params['upper'], col] = params['median']
        
        elif method == 'isolation_forest':
            # Isolation Forest method
            if is_training:
                # Fit Isolation Forest
                iso_forest = IsolationForest(contamination=0.05, random_state=42)
                self.outlier_detectors['isolation_forest'] = iso_forest
                
                # Fit on numeric columns
                iso_forest.fit(result_df[numeric_cols].fillna(0))
            
            # Use pre-fitted model
            if 'isolation_forest' in self.outlier_detectors:
                # Predict outliers
                outlier_predictions = self.outlier_detectors['isolation_forest'].predict(result_df[numeric_cols].fillna(0))
                outlier_indices = np.where(outlier_predictions == -1)[0]
                
                # Replace outliers with median for each column
                for col in numeric_cols:
                    median_value = result_df[col].median()
                    result_df.loc[result_df.index[outlier_indices], col] = median_value
        
        logger.info(f"Outliers handled using {method} method.")
        
        return result_df
    
    def extract_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from datetime columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with extracted datetime features
        """
        logger.info("Extracting datetime features...")
        
        datetime_cols = self.column_types.get('datetime', [])
        if not datetime_cols:
            logger.info("No datetime columns to extract features from.")
            return df
        
        result_df = df.copy()
        
        for col in datetime_cols:
            # Ensure column is datetime type
            if not pd.api.types.is_datetime64_dtype(result_df[col]):
                try:
                    result_df[col] = pd.to_datetime(result_df[col])
                except:
                    logger.warning(f"Could not convert column {col} to datetime. Skipping.")
                    continue
            
            # Extract features
            result_df[f'{col}_year'] = result_df[col].dt.year
            result_df[f'{col}_month'] = result_df[col].dt.month
            result_df[f'{col}_day'] = result_df[col].dt.day
            result_df[f'{col}_dayofweek'] = result_df[col].dt.dayofweek
            result_df[f'{col}_hour'] = result_df[col].dt.hour
            result_df[f'{col}_quarter'] = result_df[col].dt.quarter
            result_df[f'{col}_is_weekend'] = result_df[col].dt.dayofweek.isin([5, 6]).astype(int)
            
            # Add to numeric and binary columns
            self.column_types['numeric'].extend([
                f'{col}_year', f'{col}_month', f'{col}_day', 
                f'{col}_dayofweek', f'{col}_hour', f'{col}_quarter'
            ])
            self.column_types['binary'].append(f'{col}_is_weekend')
            
            # Track transformation
            self.transformations_applied[col] = 'datetime_features'
        
        logger.info(f"Extracted features from {len(datetime_cols)} datetime columns.")
        
        return result_df
    
    def clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean text columns by removing special characters, lowercasing, etc.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cleaned text columns
        """
        logger.info("Cleaning text columns...")
        
        text_cols = self.column_types.get('text', [])
        if not text_cols:
            logger.info("No text columns to clean.")
            return df
        
        result_df = df.copy()
        
        def clean_text(text):
            if not isinstance(text, str):
                return ""
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http\S+', '', text)
            
            # Remove special characters and numbers
            text = re.sub(r'[^\w\s]', '', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        
        for col in text_cols:
            # Apply cleaning function
            result_df[col] = result_df[col].apply(clean_text)
            
            # Track transformation
            self.transformations_applied[col] = 'text_cleaning'
        
        logger.info(f"Cleaned {len(text_cols)} text columns.")
        
        return result_df
    
    def encode_categorical(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df: Input DataFrame
            is_training: Whether this is training data or not
            
        Returns:
            DataFrame with encoded categorical variables
        """
        logger.info("Encoding categorical variables...")
        
        categorical_cols = self.column_types.get('categorical', [])
        if not categorical_cols:
            logger.info("No categorical columns to encode.")
            return df
        
        result_df = df.copy()
        
        # Get encoding method from config
        encoding_method = self.preprocessing_config.get('categorical_encoding', 'auto')
        
        # Auto-select encoding method based on cardinality
        if encoding_method == 'auto':
            high_cardinality_cols = []
            low_cardinality_cols = []
            
            for col in categorical_cols:
                cardinality = result_df[col].nunique()
                if cardinality > 10:  # Arbitrary threshold
                    high_cardinality_cols.append(col)
                else:
                    low_cardinality_cols.append(col)
            
            # Apply one-hot encoding to low cardinality columns
            if low_cardinality_cols:
                result_df = self._apply_onehot_encoding(result_df, low_cardinality_cols, is_training)
            
            # Apply label encoding to high cardinality columns
            if high_cardinality_cols:
                result_df = self._apply_label_encoding(result_df, high_cardinality_cols, is_training)
        
        elif encoding_method == 'onehot':
            # Apply one-hot encoding to all categorical columns
            result_df = self._apply_onehot_encoding(result_df, categorical_cols, is_training)
        
        elif encoding_method == 'label':
            # Apply label encoding to all categorical columns
            result_df = self._apply_label_encoding(result_df, categorical_cols, is_training)
        
        elif encoding_method == 'target':
            # Target encoding requires target column, not implemented here
            logger.warning("Target encoding not implemented in this version.")
        
        elif encoding_method == 'frequency':
            # Apply frequency encoding
            result_df = self._apply_frequency_encoding(result_df, categorical_cols, is_training)
        
        logger.info(f"Categorical encoding completed using {encoding_method} method.")
        
        return result_df
    
    def _apply_onehot_encoding(self, df: pd.DataFrame, columns: List[str], is_training: bool = True) -> pd.DataFrame:
        """Apply one-hot encoding to specified columns."""
        if not columns:
            return df
        
        result_df = df.copy()
        
        if is_training:
            # Fit one-hot encoder
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            self.categorical_encoders['onehot'] = encoder
            
            # Fit on data
            encoder.fit(result_df[columns])
        
        # Use pre-fitted encoder
        if 'onehot' in self.categorical_encoders:
            encoder = self.categorical_encoders['onehot']
            
            # Transform data
            encoded_array = encoder.transform(result_df[columns])
            
            # Create DataFrame with encoded values
            encoded_df = pd.DataFrame(
                encoded_array,
                columns=encoder.get_feature_names_out(columns),
                index=result_df.index
            )
            
            # Drop original columns and join encoded ones
            result_df = result_df.drop(columns, axis=1)
            result_df = pd.concat([result_df, encoded_df], axis=1)
            
            # Update column types
            for col in encoded_df.columns:
                self.column_types['binary'].append(col)
            
            # Track transformation
            for col in columns:
                self.transformations_applied[col] = 'onehot_encoding'
        
        return result_df
    
    def _apply_label_encoding(self, df: pd.DataFrame, columns: List[str], is_training: bool = True) -> pd.DataFrame:
        """Apply label encoding to specified columns."""
        if not columns:
            return df
        
        result_df = df.copy()
        
        for col in columns:
            if is_training:
                # Fit label encoder
                encoder = LabelEncoder()
                self.categorical_encoders[col] = encoder
                
                # Fit on data
                encoder.fit(result_df[col].astype(str).fillna('MISSING'))
            
            # Use pre-fitted encoder
            if col in self.categorical_encoders:
                encoder = self.categorical_encoders[col]
                
                # Transform data
                result_df[col] = encoder.transform(result_df[col].astype(str).fillna('MISSING'))
                
                # Update column types
                self.column_types['numeric'].append(col)
                
                # Track transformation
                self.transformations_applied[col] = 'label_encoding'
        
        return result_df
    
    def _apply_frequency_encoding(self, df: pd.DataFrame, columns: List[str], is_training: bool = True) -> pd.DataFrame:
        """Apply frequency encoding to specified columns."""
        if not columns:
            return df
        
        result_df = df.copy()
        
        for col in columns:
            if is_training:
                # Calculate frequency map
                frequency_map = result_df[col].value_counts(normalize=True).to_dict()
                self.categorical_encoders[f'{col}_freq'] = frequency_map
            
            # Use pre-calculated frequency map
            if f'{col}_freq' in self.categorical_encoders:
                frequency_map = self.categorical_encoders[f'{col}_freq']
                
                # Transform data
                result_df[col] = result_df[col].map(frequency_map).fillna(0)
                
                # Update column types
                self.column_types['numeric'].append(col)
                
                # Track transformation
                self.transformations_applied[col] = 'frequency_encoding'
        
        return result_df
    
    def _apply_feature_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection using variance threshold."""
        threshold = float(self.preprocessing_config['feature_selection'].get('threshold', 0.01))
        self.feature_selector = VarianceThreshold(threshold=threshold)
        self.feature_selector.fit(df)
        
        # Get selected features
        selected_features = df.columns[self.feature_selector.get_support()].tolist()
        return df[selected_features]
    
    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature scaling."""
        scaling_method = self.preprocessing_config.get('scaling_method', 'standard')
        
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = RobustScaler()
        
        scaled_data = self.scaler.fit_transform(df)
        return pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    
    def _apply_pca(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply PCA dimensionality reduction."""
        logger.info("Applying PCA...")
        
        # Store target column if it exists
        target_col = self.data_config.get('target_column')
        if target_col and target_col in df.columns:
            target_values = df[target_col].copy()
        
        # Select numeric columns excluding target
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if target_col in numeric_cols:
            numeric_cols = numeric_cols.drop(target_col)
        
        # Check for remaining missing values in numeric columns
        if df[numeric_cols].isnull().any().any():
            logger.warning("Found remaining missing values in numeric columns, filling with mean")
            imputer = SimpleImputer(strategy='mean')
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        # Get PCA configuration
        pca_config = self.preprocessing_config.get('dimensionality_reduction', {}).get('pca', {})
        n_components = pca_config.get('n_components', 3)
        
        # Ensure n_components doesn't exceed number of features
        n_components = min(n_components, len(numeric_cols))
        
        # Apply PCA
        self.pca = PCA(n_components=n_components)
        pca_result = self.pca.fit_transform(df[numeric_cols])
        
        # Create new DataFrame with PCA results
        pca_df = pd.DataFrame(
            pca_result,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=df.index
        )
        
        # Add target column back if it exists
        if target_col and target_col in df.columns:
            pca_df[target_col] = target_values
        
        logger.info(f"PCA applied. Explained variance ratio: {self.pca.explained_variance_ratio_}")
        
        return pca_df
    
    def save_preprocessor(self, path: str):
        """
        Save preprocessor state to file.
        
        Args:
            path: Path to save the preprocessor
        """
        import joblib
        
        preprocessor_state = {
            'categorical_encoders': self.categorical_encoders,
            'scalers': self.scalers,
            'imputers': self.imputers,
            'outlier_detectors': self.outlier_detectors,
            'column_types': self.column_types,
            'transformations_applied': self.transformations_applied,
            'config': self.config
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(preprocessor_state, path)
        logger.info(f"Preprocessor saved to {path}")
    
    @classmethod
    def load_preprocessor(cls, path: str):
        """
        Load preprocessor state from file.
        
        Args:
            path: Path to load the preprocessor from
            
        Returns:
            DataProcessor instance
        """
        import joblib
        
        preprocessor_state = joblib.load(path)
        
        # Create instance
        instance = cls(preprocessor_state['config'])
        
        # Restore state
        instance.categorical_encoders = preprocessor_state['categorical_encoders']
        instance.scalers = preprocessor_state['scalers']
        instance.imputers = preprocessor_state['imputers']
        instance.outlier_detectors = preprocessor_state['outlier_detectors']
        instance.column_types = preprocessor_state['column_types']
        instance.transformations_applied = preprocessor_state['transformations_applied']
        
        logger.info(f"Preprocessor loaded from {path}")
        
        return instance 