import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
from typing import Dict, Any, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import logging
import joblib

logger = logging.getLogger(__name__)

class ModelExplainer:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model explainer with configuration."""
        self.config = config
        self.shap_values = {}
        self.lime_explainer = None
        self.feature_importance = {}
        self.partial_dependence = {}
        
    def explain_model(self, model: Any, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Generate comprehensive model explanations."""
        logger.info("Starting model explanation pipeline")
        
        # Generate SHAP values
        if self.config['explainability']['generate_shap']:
            self._generate_shap_values(model, X)
        
        # Generate LIME explanations
        if self.config['explainability']['generate_lime']:
            self._generate_lime_explanations(model, X)
        
        # Generate feature importance
        if self.config['explainability']['feature_importance']:
            self._generate_feature_importance(model, X)
        
        # Generate partial dependence plots
        if self.config['explainability']['partial_dependence']:
            self._generate_partial_dependence(model, X)
        
        # Generate LLM-based explanations if configured
        if self.config['explainability']['llm_explanations']:
            self._generate_llm_explanations(model, X, y)
        
        logger.info("Model explanation pipeline completed")
    
    def _generate_shap_values(self, model: Any, X: pd.DataFrame) -> None:
        """Generate SHAP values for model interpretability."""
        logger.info("Generating SHAP values")
        
        # Create SHAP explainer based on model type
        if hasattr(model, 'predict_proba'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, X)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X)
        
        # Handle both binary and multiclass cases
        if isinstance(shap_values, list):
            for i, values in enumerate(shap_values):
                self.shap_values[f'class_{i}'] = values
        else:
            self.shap_values['all'] = shap_values
        
        # Create and save SHAP visualizations
        self._create_shap_visualizations(X)
    
    def _create_shap_visualizations(self, X: pd.DataFrame) -> None:
        """Create and save SHAP visualizations."""
        output_dir = Path(self.config['output']['visualization_dir'])
        
        for class_name, shap_values in self.shap_values.items():
            # Summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X, show=False)
            plt.savefig(output_dir / f'shap_summary_{class_name}.png')
            plt.close()
            
            # Bar plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X, plot_type='bar', show=False)
            plt.savefig(output_dir / f'shap_bar_{class_name}.png')
            plt.close()
            
            # Dependence plots for top features
            feature_importance = np.abs(shap_values).mean(0)
            top_features = X.columns[np.argsort(-feature_importance)[:5]]
            
            for feature in top_features:
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(feature, shap_values, X, show=False)
                plt.savefig(output_dir / f'shap_dependence_{class_name}_{feature}.png')
                plt.close()
    
    def _generate_lime_explanations(self, model: Any, X: pd.DataFrame) -> None:
        """Generate LIME explanations for model interpretability."""
        logger.info("Generating LIME explanations")
        
        # Create LIME explainer
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X.values,
            feature_names=X.columns,
            class_names=['Negative', 'Positive'] if hasattr(model, 'predict_proba') else None,
            mode='classification' if hasattr(model, 'predict_proba') else 'regression'
        )
        
        # Generate explanations for a few examples
        output_dir = Path(self.config['output']['visualization_dir'])
        for i in range(min(5, len(X))):
            exp = self.lime_explainer.explain_instance(
                X.iloc[i].values,
                model.predict_proba if hasattr(model, 'predict_proba') else model.predict
            )
            
            # Save explanation plot
            fig = exp.as_pyplot_figure()
            fig.savefig(output_dir / f'lime_explanation_{i}.png')
            fig.close()
            
            # Save explanation as HTML
            exp.save_to_file(output_dir / f'lime_explanation_{i}.html')
    
    def _generate_feature_importance(self, model: Any, X: pd.DataFrame) -> None:
        """Generate feature importance metrics."""
        logger.info("Generating feature importance")
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            logger.warning("Model does not support feature importance")
            return
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Save feature importance
        output_dir = Path(self.config['output']['visualization_dir'])
        importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
        
        # Create feature importance plot
        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            title='Feature Importance',
            orientation='h'
        )
        fig.write_html(output_dir / 'feature_importance.html')
    
    def _generate_partial_dependence(self, model: Any, X: pd.DataFrame) -> None:
        """Generate partial dependence plots."""
        logger.info("Generating partial dependence plots")
        
        from sklearn.inspection import partial_dependence
        
        output_dir = Path(self.config['output']['visualization_dir'])
        
        # Generate partial dependence for top features
        feature_importance = np.abs(model.coef_[0]) if hasattr(model, 'coef_') else model.feature_importances_
        top_features = X.columns[np.argsort(-feature_importance)[:5]]
        
        for feature in top_features:
            pdp = partial_dependence(
                model,
                X,
                [feature],
                kind='average'
            )
            
            # Create partial dependence plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pdp[1][0],
                y=pdp[0][0],
                mode='lines+markers',
                name=feature
            ))
            
            fig.update_layout(
                title=f'Partial Dependence Plot - {feature}',
                xaxis_title=feature,
                yaxis_title='Partial Dependence'
            )
            
            fig.write_html(output_dir / f'partial_dependence_{feature}.html')
    
    def _generate_llm_explanations(self, model: Any, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Generate LLM-based explanations for model behavior."""
        logger.info("Generating LLM-based explanations")
        
        if not self.config['explainability']['llm_explanations']:
            return
        
        # Implementation will depend on the specific LLM integration
        # This is a placeholder for future implementation
        pass
    
    def save_explanations(self, output_dir: str) -> None:
        """Save explanation objects for later use."""
        logger.info("Saving explanation objects")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.shap_values:
            joblib.dump(self.shap_values, output_path / 'shap_values.joblib')
        if self.lime_explainer:
            joblib.dump(self.lime_explainer, output_path / 'lime_explainer.joblib')
        if self.feature_importance:
            joblib.dump(self.feature_importance, output_path / 'feature_importance.joblib')
        if self.partial_dependence:
            joblib.dump(self.partial_dependence, output_path / 'partial_dependence.joblib')
    
    def load_explanations(self, input_dir: str) -> None:
        """Load explanation objects from disk."""
        logger.info("Loading explanation objects")
        
        input_path = Path(input_dir)
        
        if (input_path / 'shap_values.joblib').exists():
            self.shap_values = joblib.load(input_path / 'shap_values.joblib')
        if (input_path / 'lime_explainer.joblib').exists():
            self.lime_explainer = joblib.load(input_path / 'lime_explainer.joblib')
        if (input_path / 'feature_importance.joblib').exists():
            self.feature_importance = joblib.load(input_path / 'feature_importance.joblib')
        if (input_path / 'partial_dependence.joblib').exists():
            self.partial_dependence = joblib.load(input_path / 'partial_dependence.joblib') 