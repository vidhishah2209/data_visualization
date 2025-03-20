import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from lime import lime_tabular

logger = logging.getLogger(__name__)

class ModelExplainer:
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        self.explanation_config = config.get('explanation', {})
        self.output_dir = Path(config.get('output_dir', 'results/explanations'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_explanations(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Generate explanations for all models."""
        logger.info("Starting model explanation generation")
        explanations = {}
        
        for model_name, model in models.items():
            try:
                # Generate feature importance
                feature_importance = self._calculate_feature_importance(model, X, y)
                
                # Generate LIME explanations
                lime_explanation = self._generate_lime_explanation(model, X)
                
                # Store explanations
                explanations[model_name] = {
                    'feature_importance': feature_importance,
                    'lime_explanation': lime_explanation
                }
                
                # Save visualizations
                self._save_explanation_plots(model_name, feature_importance)
                
                logger.info(f"Generated explanations for {model_name}")
                
            except Exception as e:
                logger.error(f"Error generating explanations for {model_name}: {str(e)}")
        
        return explanations
    
    def _calculate_feature_importance(self, model: Any, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Calculate feature importance using permutation importance."""
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': result.importances_mean,
            'std': result.importances_std,
            'p_values': result.importances_pvalues
        })
        
        return importance_df.sort_values('importance', ascending=False)
    
    def _generate_lime_explanation(self, model: Any, X: pd.DataFrame) -> Any:
        """Generate LIME explanations for model predictions."""
        explainer = lime_tabular.LimeTabularExplainer(
            X.values,
            feature_names=X.columns,
            class_names=['0', '1'],
            mode='classification'
        )
        
        # Generate explanation for first instance
        exp = explainer.explain_instance(X.iloc[0].values, model.predict_proba)
        return exp
    
    def _save_explanation_plots(self, model_name: str, feature_importance: pd.DataFrame) -> None:
        """Save explanation visualizations."""
        # Feature importance plot
        fig = px.bar(
            feature_importance,
            x='importance',
            y='feature',
            title=f'Feature Importance - {model_name}',
            error_x='std'
        )
        fig.write_html(self.output_dir / f'feature_importance_{model_name}.html')
    
    def save_explanations(self, explanations: Dict[str, Any]) -> None:
        """Save explanations to files."""
        for model_name, explanation in explanations.items():
            # Save feature importance
            explanation['feature_importance'].to_csv(
                self.output_dir / f'feature_importance_{model_name}.csv'
            ) 