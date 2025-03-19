# Feature Engineering Module

This module provides automated feature engineering capabilities for machine learning workflows, including:

- Text embeddings generation using OpenAI's API or local models
- Polynomial feature generation
- Clustering-based feature generation
- Automated feature selection
- GPU acceleration support
- Support for multiple LLM providers (OpenAI, Google Gemini)

## Usage Example

```python
import pandas as pd
from feature_engineering.feature_generator import FeatureGenerator

# Configuration for OpenAI
config_openai = {
    'feature_engineering': {
        'use_llm': True,
        'text_embeddings': True,
        'polynomial_features': True,
        'clustering_features': True,
        'embedding_model': 'text-embedding-ada-002',
        'llm_provider': 'openai',
        'llm_model': 'gpt-4'
    }
}

# Configuration for Gemini
config_gemini = {
    'feature_engineering': {
        'use_llm': True,
        'text_embeddings': True,
        'polynomial_features': True,
        'clustering_features': True,
        'embedding_model': 'text-embedding-ada-002',
        'llm_provider': 'gemini',
        'llm_model': 'gemini-pro'
    }
}

# Initialize feature generator (choose your config)
generator = FeatureGenerator(config_gemini)  # Using Gemini

# Prepare your data
data = pd.DataFrame({
    'numeric1': [1, 2, 3],
    'numeric2': [4, 5, 6],
    'text': ['sample text 1', 'sample text 2', 'sample text 3']
})

# Define column types
column_types = {
    'numeric': ['numeric1', 'numeric2'],
    'text': ['text']
}

# Generate features
result_df = generator.generate_features(
    df=data,
    column_types=column_types,
    is_training=True
)

print(f"Generated features shape: {result_df.shape}")
print("\nGenerated feature columns:")
print(result_df.columns.tolist())
```

## Configuration Options

The module supports the following configuration options:

```python
config = {
    'feature_engineering': {
        # LLM Settings
        'use_llm': True,  # Enable/disable LLM features
        'llm_provider': 'gemini',  # LLM provider ('openai' or 'gemini')
        'llm_model': 'gemini-pro',  # LLM model to use
        
        # Text Embedding Settings
        'text_embeddings': True,  # Enable/disable text embeddings
        'embedding_model': 'text-embedding-ada-002',  # Model for text embeddings
        
        # Feature Generation Settings
        'polynomial_features': True,  # Enable/disable polynomial features
        'clustering_features': True,  # Enable/disable clustering features
    }
}
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- sentence-transformers
- openai
- google-generativeai
- tqdm

## Environment Variables

Set up your API keys based on the LLM provider you're using:

```bash
# For OpenAI
export OPENAI_API_KEY=your_openai_api_key_here

# For Google Gemini
export GOOGLE_API_KEY=your_google_api_key_here
```

## LLM Providers

1. **OpenAI**
   - Default provider
   - Supports GPT-3.5 and GPT-4 models
   - Requires OpenAI API key

2. **Google Gemini**
   - Alternative provider
   - Supports Gemini Pro model
   - Requires Google API key
   - Generally faster response times
   - More cost-effective for high-volume usage

## Feature Types

1. **Text Embeddings**
   - Generates embeddings for text columns using OpenAI's API or local models
   - Supports both API-based and local embedding models
   - Handles missing values automatically

2. **Polynomial Features**
   - Generates polynomial and interaction features for numeric columns
   - Degree 2 polynomials by default
   - Excludes bias terms to avoid multicollinearity

3. **Clustering Features**
   - Generates cluster assignments and distances to cluster centers
   - Automatically determines optimal number of clusters
   - Uses KMeans clustering with smart initialization

## State Management

The module maintains state for:
- Fitted transformers
- Feature names
- Applied transformations

This allows for consistent feature generation between training and inference. 