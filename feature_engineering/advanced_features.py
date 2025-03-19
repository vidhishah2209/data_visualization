import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineering:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the feature engineering class with configuration."""
        self.config = config
        self.poly_features = None
        self.pca = None
        self.variance_threshold = None
        self.feature_names = None
    
    def create_polynomial_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features up to specified degree."""
        logger.info("Creating polynomial features")
        poly_config = self.config['feature_engineering']['polynomial_features']
        
        poly = PolynomialFeatures(
            degree=poly_config['degree'],
            include_bias=poly_config['include_bias']
        )
        
        poly_features = poly.fit_transform(X)
        self.poly_features = poly
        
        # Create feature names
        feature_names = poly.get_feature_names_out(X.columns)
        return pd.DataFrame(poly_features, columns=feature_names, index=X.index)
    
    def create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between numerical columns."""
        logger.info("Creating interaction features")
        
        numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
        interaction_features = pd.DataFrame(index=X.index)
        
        for i, col1 in enumerate(numerical_cols):
            for col2 in numerical_cols[i+1:]:
                interaction_features[f"{col1}_{col2}_interaction"] = X[col1] * X[col2]
        
        return pd.concat([X, interaction_features], axis=1)
    
    def create_time_features(self, X: pd.DataFrame, time_column: str) -> pd.DataFrame:
        """Create time-based features from datetime column."""
        logger.info("Creating time-based features")
        
        time_features = pd.DataFrame(index=X.index)
        df_time = pd.to_datetime(X[time_column])
        
        time_features['hour'] = df_time.dt.hour
        time_features['day'] = df_time.dt.day
        time_features['month'] = df_time.dt.month
        time_features['year'] = df_time.dt.year
        time_features['dayofweek'] = df_time.dt.dayofweek
        time_features['quarter'] = df_time.dt.quarter
        time_features['is_weekend'] = time_features['dayofweek'].isin([5, 6]).astype(int)
        
        return pd.concat([X, time_features], axis=1)
    
    def apply_dimensionality_reduction(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply dimensionality reduction techniques."""
        logger.info("Applying dimensionality reduction")
        
        dim_reduction_config = self.config['preprocessing']['dimensionality_reduction']
        
        if dim_reduction_config['method'] == 'pca':
            self.pca = PCA(n_components=dim_reduction_config['n_components'])
            reduced_features = self.pca.fit_transform(X)
            
            # Create feature names
            feature_names = [f"PC{i+1}" for i in range(dim_reduction_config['n_components'])]
            return pd.DataFrame(reduced_features, columns=feature_names, index=X.index)
        
        return X
    
    def select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection techniques."""
        logger.info("Applying feature selection")
        
        feature_selection_config = self.config['preprocessing']['feature_selection']
        
        if feature_selection_config['method'] == 'variance_threshold':
            self.variance_threshold = VarianceThreshold(
                threshold=feature_selection_config['threshold']
            )
            selected_features = self.variance_threshold.fit_transform(X)
            
            # Get selected feature names
            feature_names = X.columns[self.variance_threshold.get_support()].tolist()
            return pd.DataFrame(selected_features, columns=feature_names, index=X.index)
        
        return X
    
    def create_clustering_features(self, X: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
        """Create features based on clustering results."""
        logger.info("Creating clustering-based features")
        
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # Add cluster labels as features
        X['cluster'] = cluster_labels
        
        # Add distance to cluster centers
        for i in range(n_clusters):
            X[f'distance_to_cluster_{i}'] = np.linalg.norm(
                X - kmeans.cluster_centers_[i], axis=1
            )
        
        return X
    
    def create_aggregation_features(self, X: pd.DataFrame, group_by: str) -> pd.DataFrame:
        """Create aggregation features based on grouping."""
        logger.info("Creating aggregation features")
        
        numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
        
        agg_features = X.groupby(group_by)[numerical_cols].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).reset_index()
        
        # Flatten column names
        agg_features.columns = [
            f"{group_by}_{col[0]}_{col[1]}" if col[1] else col[0]
            for col in agg_features.columns
        ]
        
        return pd.merge(X, agg_features, on=group_by, how='left')
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering transformations."""
        logger.info("Starting feature engineering pipeline")
        
        # Store original feature names
        self.feature_names = X.columns.tolist()
        
        # Apply transformations based on configuration
        if self.config['feature_engineering']['polynomial_features']['degree'] > 1:
            X = self.create_polynomial_features(X)
        
        if self.config['feature_engineering']['interaction_features']:
            X = self.create_interaction_features(X)
        
        if self.config['feature_engineering']['time_features']:
            time_column = self.config['data']['time_column']
            X = self.create_time_features(X, time_column)
        
        # Apply dimensionality reduction
        X = self.apply_dimensionality_reduction(X)
        
        # Apply feature selection
        X = self.select_features(X)
        
        logger.info(f"Feature engineering completed. Final shape: {X.shape}")
        return X 