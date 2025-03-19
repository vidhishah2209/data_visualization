import os
import yaml
import logging
import numpy as np
import pandas as pd
import joblib
import time
import json
from typing import Dict, List, Union, Tuple, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
import traceback
import shutil
import sys
import re
import random
from tqdm import tqdm

# Configure logging
def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('hackathon_project.log')
        ]
    )
    # Suppress warnings
    warnings.filterwarnings('ignore')
    return logging.getLogger('hackathon_project')

logger = setup_logging()

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        # Replace environment variables
        config_str = json.dumps(config)
        env_pattern = r'\${([^}]*)}'
        
        def replace_env_var(match):
            env_var = match.group(1)
            return os.environ.get(env_var, f"${{{env_var}}}")
        
        config_str = re.sub(env_pattern, replace_env_var, config_str)
        config = json.loads(config_str)
        
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

def check_gpu_availability():
    """Check if GPU is available for computation."""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU is available: {gpu_count} device(s). Using {gpu_name}")
        else:
            logger.warning("No GPU available, using CPU.")
        return gpu_available
    except ImportError:
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"GPU is available: {len(gpus)} device(s)")
                return True
            else:
                logger.warning("No GPU available, using CPU.")
                return False
        except ImportError:
            logger.warning("Neither PyTorch nor TensorFlow is installed. Cannot check GPU availability.")
            return False

def timer_decorator(func):
    """Decorator to measure execution time of functions."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to execute.")
        return result
    return wrapper

def save_dataframe(df, path, format='csv'):
    """Save DataFrame to file in specified format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if format.lower() == 'csv':
        df.to_csv(path, index=False)
    elif format.lower() == 'parquet':
        df.to_parquet(path, index=False)
    elif format.lower() == 'pickle':
        df.to_pickle(path)
    elif format.lower() == 'excel':
        df.to_excel(path, index=False)
    elif format.lower() == 'json':
        df.to_json(path, orient='records')
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"DataFrame saved to {path}")

def load_dataframe(path):
    """Load DataFrame from file based on extension."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    file_extension = os.path.splitext(path)[1].lower()
    
    if file_extension == '.csv':
        return pd.read_csv(path)
    elif file_extension == '.parquet':
        return pd.read_parquet(path)
    elif file_extension in ['.xls', '.xlsx']:
        return pd.read_excel(path)
    elif file_extension == '.json':
        return pd.read_json(path)
    elif file_extension == '.pkl':
        return pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

def save_model(model, path):
    """Save model to file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")

def load_model(path):
    """Load model from file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)

def create_timestamp():
    """Create a timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def memory_usage(df):
    """Calculate memory usage of a DataFrame."""
    memory_usage_bytes = df.memory_usage(deep=True).sum()
    memory_usage_mb = memory_usage_bytes / (1024 * 1024)
    return f"{memory_usage_mb:.2f} MB"

def parallelize_dataframe(df, func, n_cores=None):
    """Apply a function to a DataFrame in parallel."""
    try:
        from joblib import Parallel, delayed
        
        if n_cores is None:
            n_cores = os.cpu_count()
        
        df_split = np.array_split(df, n_cores)
        df = pd.concat(Parallel(n_jobs=n_cores)(delayed(func)(df_part) for df_part in df_split))
        return df
    except Exception as e:
        logger.error(f"Error in parallel processing: {str(e)}")
        return func(df)

def create_directory_structure():
    """Create the project directory structure."""
    directories = [
        'data/raw',
        'data/processed',
        'preprocessing',
        'feature_engineering',
        'models',
        'explainability',
        'results/plots',
        'results/reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def get_file_size(file_path):
    """Get file size in MB."""
    if os.path.isfile(file_path):
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        return f"{file_size:.2f} MB"
    return "File not found"

def detect_file_format(file_path):
    """Detect file format based on extension and content."""
    extension = os.path.splitext(file_path)[1].lower()
    
    # Map extensions to formats
    format_map = {
        '.csv': 'csv',
        '.parquet': 'parquet',
        '.json': 'json',
        '.xls': 'excel',
        '.xlsx': 'excel',
        '.pkl': 'pickle',
        '.txt': 'text'
    }
    
    return format_map.get(extension, 'unknown')

def generate_summary_stats(df):
    """Generate summary statistics for a DataFrame."""
    summary = {
        'shape': df.shape,
        'memory_usage': memory_usage(df),
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_stats': df.describe().to_dict() if not df.select_dtypes(include=['number']).empty else {},
        'categorical_stats': {col: df[col].value_counts().to_dict() for col in df.select_dtypes(include=['object', 'category']).columns}
    }
    return summary

def plot_correlation_matrix(df, method='pearson', figsize=(12, 10), save_path=None):
    """Plot correlation matrix for numeric columns."""
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        logger.warning("No numeric columns to plot correlation matrix.")
        return None
    
    corr = numeric_df.corr(method=method)
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5)
    plt.title('Correlation Matrix')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Correlation matrix saved to {save_path}")
    
    return plt.gcf()

def plot_missing_values(df, figsize=(12, 6), save_path=None):
    """Plot missing values heatmap."""
    plt.figure(figsize=figsize)
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Missing Values')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Missing values plot saved to {save_path}")
    
    return plt.gcf()

def plot_feature_importance(importance, names, model_type, figsize=(12, 8), save_path=None):
    """Plot feature importance."""
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
    
    plt.figure(figsize=figsize)
    sns.barplot(x='feature_importance', y='feature_names', data=fi_df)
    plt.title(f'{model_type} Feature Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    return plt.gcf()

def create_plotly_figure(fig_type, data, **kwargs):
    """Create a Plotly figure based on type."""
    if fig_type == 'scatter':
        fig = px.scatter(data, **kwargs)
    elif fig_type == 'line':
        fig = px.line(data, **kwargs)
    elif fig_type == 'bar':
        fig = px.bar(data, **kwargs)
    elif fig_type == 'histogram':
        fig = px.histogram(data, **kwargs)
    elif fig_type == 'box':
        fig = px.box(data, **kwargs)
    elif fig_type == 'heatmap':
        fig = px.imshow(data, **kwargs)
    else:
        raise ValueError(f"Unsupported figure type: {fig_type}")
    
    return fig

def save_plotly_figure(fig, path, format='html'):
    """Save a Plotly figure to file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if format.lower() == 'html':
        fig.write_html(path)
    elif format.lower() == 'json':
        fig.write_json(path)
    elif format.lower() in ['png', 'jpg', 'jpeg', 'svg', 'pdf']:
        fig.write_image(path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Plotly figure saved to {path}")

def get_optimal_dtypes(df):
    """Optimize DataFrame dtypes to reduce memory usage."""
    result_df = df.copy()
    
    for col in result_df.columns:
        col_dtype = result_df[col].dtype
        
        # Optimize integers
        if col_dtype == 'int64':
            c_min, c_max = result_df[col].min(), result_df[col].max()
            
            if c_min >= 0:
                if c_max < 2**8:
                    result_df[col] = result_df[col].astype('uint8')
                elif c_max < 2**16:
                    result_df[col] = result_df[col].astype('uint16')
                elif c_max < 2**32:
                    result_df[col] = result_df[col].astype('uint32')
            else:
                if c_min > -2**7 and c_max < 2**7:
                    result_df[col] = result_df[col].astype('int8')
                elif c_min > -2**15 and c_max < 2**15:
                    result_df[col] = result_df[col].astype('int16')
                elif c_min > -2**31 and c_max < 2**31:
                    result_df[col] = result_df[col].astype('int32')
        
        # Optimize floats
        elif col_dtype == 'float64':
            result_df[col] = result_df[col].astype('float32')
    
    return result_df

def check_system_resources():
    """Check available system resources."""
    import psutil
    
    resources = {
        'cpu_count': os.cpu_count(),
        'memory_total': f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
        'memory_available': f"{psutil.virtual_memory().available / (1024**3):.2f} GB",
        'disk_total': f"{psutil.disk_usage('/').total / (1024**3):.2f} GB",
        'disk_free': f"{psutil.disk_usage('/').free / (1024**3):.2f} GB"
    }
    
    # Check for GPU if available
    gpu_info = {}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['gpu_count'] = torch.cuda.device_count()
            gpu_info['gpu_name'] = torch.cuda.get_device_name(0)
            gpu_info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
    except ImportError:
        pass
    
    if gpu_info:
        resources['gpu'] = gpu_info
    
    return resources 