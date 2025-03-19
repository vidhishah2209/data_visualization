"""Feature generator module."""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import openai
import google.generativeai as genai
import warnings
from tqdm import tqdm

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import timer_decorator, logger

class FeatureGenerator:
    """Class for automated feature engineering."""
    
    def __init__(self, config: Dict):
        """Initialize with configuration."""
        self.config = config
        self.feature_config = config.get('feature_engineering', {})
        self.transformers = {}
        self.feature_names = {}
        
        # Set up LLM if enabled
        if self.feature_config.get('use_llm', True):
            self._setup_llm()
        
        # Set up embedding model if needed
        if self.feature_config.get('text_embeddings', True):
            self._setup_embedding_model()
        
        # Suppress warnings
        warnings.filterwarnings('ignore')
    
    def _setup_llm(self):
        """Set up LLM for feature engineering suggestions."""
        llm_provider = self.feature_config.get('llm_provider', 'openai')
        
        if llm_provider == 'openai':
            openai.api_key = os.getenv('OPENAI_API_KEY')
            self.llm_model = self.feature_config.get('llm_model', 'gpt-4')
            self.llm_provider = 'openai'
        elif llm_provider == 'gemini':
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            model_name = self.feature_config.get('llm_model', 'gemini-pro')
            self.llm_model = genai.GenerativeModel(model_name)
            self.llm_provider = 'gemini'
        else:
            logger.warning(f"LLM provider {llm_provider} not supported yet.")
    
    def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM."""
        try:
            if self.llm_provider == 'openai':
                response = openai.ChatCompletion.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            elif self.llm_provider == 'gemini':
                response = self.llm_model.generate_content(prompt)
                return response.text
            else:
                logger.warning("No valid LLM provider configured")
                return ""
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}")
            return ""
    
    def _setup_embedding_model(self):
        """Set up text embedding model."""
        model_name = self.feature_config.get('embedding_model', 'text-embedding-ada-002')
        if model_name != 'text-embedding-ada-002':
            try:
                self.embedding_model = SentenceTransformer(model_name)
            except Exception as e:
                logger.error(f"Error loading embedding model: {str(e)}")
                self.embedding_model = None
    
    @timer_decorator
    def generate_features(self, df: pd.DataFrame, column_types: Dict, is_training: bool = True) -> pd.DataFrame:
        """Generate features."""
        result_df = df.copy()
        
        # Generate text embeddings if enabled
        if self.feature_config.get('text_embeddings', True):
            result_df = self.generate_text_embeddings(result_df, column_types.get('text', []))
        
        # Generate polynomial features if enabled
        if self.feature_config.get('polynomial_features', True):
            result_df = self.generate_polynomial_features(result_df, column_types.get('numeric', []), is_training)
        
        # Generate clustering features if enabled
        if self.feature_config.get('clustering_features', True):
            result_df = self.generate_clustering_features(result_df, column_types.get('numeric', []), is_training)
        
        return result_df
    
    def generate_text_embeddings(self, df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
        """Generate embeddings for text columns."""
        if not text_columns:
            return df
        
        result_df = df.copy()
        
        for col in text_columns:
            if col not in result_df.columns:
                continue
            
            # Generate embeddings using OpenAI API
            if self.feature_config.get('embedding_model') == 'text-embedding-ada-002':
                try:
                    embeddings = []
                    for text in tqdm(result_df[col].fillna(''), desc=f"Generating embeddings for {col}"):
                        response = openai.Embedding.create(
                            input=text,
                            model="text-embedding-ada-002"
                        )
                        embeddings.append(response['data'][0]['embedding'])
                    
                    # Convert embeddings to DataFrame
                    embedding_df = pd.DataFrame(
                        embeddings,
                        columns=[f"{col}_emb_{i}" for i in range(len(embeddings[0]))],
                        index=result_df.index
                    )
                    
                    # Add embeddings to result DataFrame
                    result_df = pd.concat([result_df, embedding_df], axis=1)
                except Exception as e:
                    logger.error(f"Error generating embeddings for {col}: {str(e)}")
            
            # Generate embeddings using local model
            elif self.embedding_model is not None:
                try:
                    embeddings = self.embedding_model.encode(
                        result_df[col].fillna('').tolist(),
                        show_progress_bar=True,
                        batch_size=32
                    )
                    
                    # Convert embeddings to DataFrame
                    embedding_df = pd.DataFrame(
                        embeddings,
                        columns=[f"{col}_emb_{i}" for i in range(embeddings.shape[1])],
                        index=result_df.index
                    )
                    
                    # Add embeddings to result DataFrame
                    result_df = pd.concat([result_df, embedding_df], axis=1)
                except Exception as e:
                    logger.error(f"Error generating embeddings for {col}: {str(e)}")
        
        return result_df
    
    def generate_polynomial_features(self, df: pd.DataFrame, numeric_columns: List[str], is_training: bool = True) -> pd.DataFrame:
        """Generate polynomial features."""
        if not numeric_columns:
            return df
        
        result_df = df.copy()
        
        if is_training:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            self.transformers['polynomial'] = poly
            
            # Fit and transform
            poly_features = poly.fit_transform(result_df[numeric_columns])
            
            # Get feature names
            feature_names = poly.get_feature_names_out(numeric_columns)
            self.feature_names['polynomial'] = feature_names
            
            # Create DataFrame with polynomial features
            poly_df = pd.DataFrame(
                poly_features[:, len(numeric_columns):],  # Skip original features
                columns=feature_names[len(numeric_columns):],
                index=result_df.index
            )
            
            # Add polynomial features to result DataFrame
            result_df = pd.concat([result_df, poly_df], axis=1)
        else:
            if 'polynomial' in self.transformers:
                poly = self.transformers['polynomial']
                poly_features = poly.transform(result_df[numeric_columns])
                
                # Create DataFrame with polynomial features
                poly_df = pd.DataFrame(
                    poly_features[:, len(numeric_columns):],  # Skip original features
                    columns=self.feature_names['polynomial'][len(numeric_columns):],
                    index=result_df.index
                )
                
                # Add polynomial features to result DataFrame
                result_df = pd.concat([result_df, poly_df], axis=1)
        
        return result_df
    
    def generate_clustering_features(self, df: pd.DataFrame, numeric_columns: List[str], is_training: bool = True) -> pd.DataFrame:
        """Generate clustering-based features."""
        if not numeric_columns:
            return df
        
        result_df = df.copy()
        
        if is_training:
            kmeans = KMeans(n_clusters=min(5, len(df)), random_state=42)
            self.transformers['kmeans'] = kmeans
            cluster_labels = kmeans.fit_predict(result_df[numeric_columns])
            distances = kmeans.transform(result_df[numeric_columns])
        else:
            if 'kmeans' in self.transformers:
                kmeans = self.transformers['kmeans']
                cluster_labels = kmeans.predict(result_df[numeric_columns])
                distances = kmeans.transform(result_df[numeric_columns])
        
        if 'kmeans' in self.transformers:
            result_df['cluster_label'] = cluster_labels
            for i in range(distances.shape[1]):
                result_df[f'cluster_distance_{i}'] = distances[:, i]
        
        return result_df 