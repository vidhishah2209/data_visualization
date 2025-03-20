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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        self.model_config = config.get('model', {})
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
    
    def train_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                    y_train: pd.Series, y_test: pd.Series) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Train and evaluate multiple models."""
        logger.info("Starting model training")
        
        # Handle NaN values
        if y_train.isnull().any():
            logger.warning("Found NaN values in y_train, dropping those rows")
            mask = ~y_train.isnull()
            X_train = X_train[mask]
            y_train = y_train[mask]
        
        if y_test.isnull().any():
            logger.warning("Found NaN values in y_test, dropping those rows")
            mask = ~y_test.isnull()
            X_test = X_test[mask]
            y_test = y_test[mask]
        
        if X_train.isnull().any().any():
            logger.warning("Found NaN values in X_train, filling with mean")
            X_train = X_train.fillna(X_train.mean())
        
        if X_test.isnull().any().any():
            logger.warning("Found NaN values in X_test, filling with mean")
            X_test = X_test.fillna(X_test.mean())
        
        # Get list of models to try
        models_to_try = self.model_config.get('models_to_try', ['random_forest', 'gradient_boosting', 'logistic_regression'])
        
        # Train each model
        for model_name in models_to_try:
            try:
                logger.info(f"Training {model_name}")
                model = self._get_model(model_name)
                model.fit(X_train, y_train)
                self.models[model_name] = model
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred)
                self.metrics[model_name] = metrics
                
                logger.info(f"Model {model_name} trained successfully")
                logger.info(f"Metrics: {metrics}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        # Calculate ensemble metrics if enabled and if we have trained models
        if self.model_config.get('ensemble', True) and self.models:
            ensemble_metrics = self._calculate_ensemble_metrics(X_test, y_test)
            logger.info(f"Ensemble metrics: {ensemble_metrics}")
            self.metrics['ensemble'] = ensemble_metrics
        
        # Return best model's metrics if we have any models
        if self.models:
            best_model_name = max(self.metrics.items(), key=lambda x: x[1].get('accuracy', 0))[0]
            return self.models, self.metrics[best_model_name]
        else:
            logger.error("No models were successfully trained")
            return {}, {}
    
    def _get_model(self, model_name: str) -> Any:
        """Get model instance based on name."""
        if model_name == 'random_forest':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_name == 'gradient_boosting':
            return GradientBoostingClassifier(random_state=42)
        elif model_name == 'logistic_regression':
            return LogisticRegression(random_state=42)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred)
        }
    
    def _calculate_ensemble_metrics(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Calculate metrics for ensemble predictions."""
        if not self.models:
            return {}
        
        # Get predictions from all models
        predictions = []
        for model in self.models.values():
            predictions.append(model.predict_proba(X_test)[:, 1])
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)
        
        return self._calculate_metrics(y_test, ensemble_pred_binary)
    
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