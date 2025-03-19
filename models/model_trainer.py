import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error, r2_score
)
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import logging
import importlib

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model trainer with configuration."""
        self.config = config
        self.models = {}
        self.best_models = {}
        self.feature_importance = {}
        self.metrics = {}
        
    def _get_model_instance(self, model_name: str, model_config: Dict[str, Any]) -> Any:
        """Get model instance based on configuration."""
        logger.info(f"Creating model instance for {model_name}")
        
        # Split the model type into module and class
        module_path, class_name = model_config['type'].rsplit('.', 1)
        
        # Import the module
        module = importlib.import_module(module_path)
        
        # Get the model class
        model_class = getattr(module, class_name)
        
        # Create model instance with parameters
        return model_class(**model_config['params'])
    
    def _evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series,
                       X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, float]:
        """Evaluate model performance using multiple metrics."""
        metrics = {}
        
        # Make predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics['accuracy'] = accuracy_score(y, y_pred)
        metrics['precision'] = precision_score(y, y_pred, average='weighted')
        metrics['recall'] = recall_score(y, y_pred, average='weighted')
        metrics['f1'] = f1_score(y, y_pred, average='weighted')
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
        
        # Cross-validation scores
        if self.config['evaluation']['cross_validation']['enabled']:
            cv_scores = cross_val_score(
                model, X, y,
                cv=self.config['evaluation']['cross_validation']['folds'],
                scoring='accuracy'
            )
            metrics['cv_mean'] = cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()
        
        # Validation set metrics if provided
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            metrics['val_accuracy'] = accuracy_score(y_val, y_val_pred)
            metrics['val_f1'] = f1_score(y_val, y_val_pred, average='weighted')
        
        return metrics
    
    def _tune_hyperparameters(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Dict[str, Any]]:
        """Perform hyperparameter tuning using specified method."""
        logger.info("Starting hyperparameter tuning")
        
        tuning_config = self.config['models'][model.__class__.__name__]['hyperparameter_tuning']
        
        if tuning_config['method'] == 'grid_search':
            grid_search = GridSearchCV(
                model,
                param_grid=tuning_config['params'],
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X, y)
            return grid_search.best_estimator_, grid_search.best_params_
        
        return model, {}
    
    def _plot_feature_importance(self, model: Any, feature_names: List[str],
                               model_name: str) -> None:
        """Plot and save feature importance visualization."""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            logger.warning(f"Model {model_name} does not support feature importance")
            return
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            title=f'Feature Importance - {model_name}',
            orientation='h'
        )
        
        # Save plot
        output_dir = Path(self.config['output']['visualization_dir'])
        fig.write_html(output_dir / f'feature_importance_{model_name}.html')
        
        # Save feature importance data
        importance_df.to_csv(
            output_dir / f'feature_importance_{model_name}.csv',
            index=False
        )
    
    def _plot_confusion_matrix(self, model: Any, X: pd.DataFrame, y: pd.Series,
                             model_name: str) -> None:
        """Plot and save confusion matrix visualization."""
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Negative", "Positive"],
            y=["Negative", "Positive"],
            title=f'Confusion Matrix - {model_name}'
        )
        
        # Save plot
        output_dir = Path(self.config['output']['visualization_dir'])
        fig.write_html(output_dir / f'confusion_matrix_{model_name}.html')
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame = None, y_val: pd.Series = None) -> None:
        """Train multiple models and store results."""
        logger.info("Starting model training pipeline")
        
        for model_name, model_config in self.config['models'].items():
            logger.info(f"Training {model_name}")
            
            # Get model instance
            model = self._get_model_instance(model_name, model_config)
            
            # Perform hyperparameter tuning if enabled
            if model_config['hyperparameter_tuning']['enabled']:
                model, best_params = self._tune_hyperparameters(model, X_train, y_train)
                logger.info(f"Best parameters for {model_name}: {best_params}")
            
            # Train model
            model.fit(X_train, y_train)
            self.models[model_name] = model
            
            # Evaluate model
            metrics = self._evaluate_model(model, X_train, y_train, X_val, y_val)
            self.metrics[model_name] = metrics
            
            # Generate visualizations
            self._plot_feature_importance(model, X_train.columns.tolist(), model_name)
            self._plot_confusion_matrix(model, X_train, y_train, model_name)
            
            # Save model if configured
            if self.config['output']['save_models']:
                model_dir = Path(self.config['output']['model_dir'])
                model_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, model_dir / f'{model_name}.joblib')
        
        # Save metrics
        if self.config['output']['save_metrics']:
            metrics_dir = Path(self.config['output']['metrics_dir'])
            metrics_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self.metrics).to_csv(metrics_dir / 'model_metrics.csv')
    
    def predict(self, X: pd.DataFrame, model_name: str = None) -> pd.Series:
        """Make predictions using trained model(s)."""
        if model_name is not None:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            return self.models[model_name].predict(X)
        
        # If no model specified, use the best model based on validation accuracy
        best_model = max(
            self.models.items(),
            key=lambda x: self.metrics[x[0]].get('val_accuracy', 0)
        )[1]
        return best_model.predict(X) 