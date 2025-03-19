import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class AdvancedVisualizer:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the visualizer with configuration."""
        self.config = config
        self.output_dir = Path(config['output']['visualization_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set visualization style
        plt.style.use(config['visualization']['style'])
        sns.set_palette(config['visualization']['color_palette'])
    
    def create_distribution_plots(self, data: pd.DataFrame) -> None:
        """Create distribution plots for numerical features."""
        logger.info("Creating distribution plots")
        
        numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
        
        for col in numerical_cols:
            # Create histogram with KDE
            fig = px.histogram(
                data,
                x=col,
                title=f'Distribution of {col}',
                marginal='box',
                nbins=30
            )
            
            # Add KDE curve
            kde = sns.kdeplot(data[col], color='red', ax=plt.gca())
            fig.add_trace(
                go.Scatter(
                    x=kde.get_lines()[0].get_xdata(),
                    y=kde.get_lines()[0].get_ydata(),
                    mode='lines',
                    name='KDE',
                    line=dict(color='red')
                )
            )
            
            fig.write_html(self.output_dir / f'distribution_{col}.html')
    
    def create_correlation_heatmap(self, data: pd.DataFrame) -> None:
        """Create an interactive correlation heatmap."""
        logger.info("Creating correlation heatmap")
        
        corr_matrix = data.corr()
        
        fig = px.imshow(
            corr_matrix,
            title='Feature Correlation Heatmap',
            aspect='auto',
            color_continuous_scale='RdBu'
        )
        
        fig.update_layout(
            width=1000,
            height=800,
            xaxis_title="Features",
            yaxis_title="Features"
        )
        
        fig.write_html(self.output_dir / 'correlation_heatmap.html')
    
    def create_3d_scatter_plot(self, data: pd.DataFrame, x: str, y: str, z: str,
                             color: Optional[str] = None) -> None:
        """Create an interactive 3D scatter plot."""
        logger.info(f"Creating 3D scatter plot for {x}, {y}, {z}")
        
        fig = px.scatter_3d(
            data,
            x=x,
            y=y,
            z=z,
            color=color,
            title=f'3D Scatter Plot: {x} vs {y} vs {z}'
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title=x,
                yaxis_title=y,
                zaxis_title=z
            )
        )
        
        fig.write_html(self.output_dir / f'3d_scatter_{x}_{y}_{z}.html')
    
    def create_3d_pca_visualization(self, data: pd.DataFrame, target: Optional[str] = None) -> None:
        """Create 3D visualization of PCA results."""
        logger.info("Creating 3D PCA visualization")
        
        # Prepare data
        X = data.select_dtypes(include=['float64', 'int64'])
        X_scaled = StandardScaler().fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(X_scaled)
        
        # Create DataFrame with PCA results
        pca_df = pd.DataFrame(
            pca_result,
            columns=['PC1', 'PC2', 'PC3'],
            index=data.index
        )
        
        if target is not None:
            pca_df['target'] = data[target]
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            pca_df,
            x='PC1',
            y='PC2',
            z='PC3',
            color='target' if target is not None else None,
            title='3D PCA Visualization'
        )
        
        # Add explained variance ratio information
        explained_var = pca.explained_variance_ratio_
        fig.update_layout(
            scene=dict(
                xaxis_title=f'PC1 ({explained_var[0]:.2%} variance)',
                yaxis_title=f'PC2 ({explained_var[1]:.2%} variance)',
                zaxis_title=f'PC3 ({explained_var[2]:.2%} variance)'
            )
        )
        
        fig.write_html(self.output_dir / '3d_pca.html')
    
    def create_feature_relationships(self, data: pd.DataFrame, target: str) -> None:
        """Create pairwise feature relationship plots."""
        logger.info("Creating feature relationship plots")
        
        numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
        n_cols = len(numerical_cols)
        
        # Create subplots
        fig = make_subplots(
            rows=n_cols,
            cols=n_cols,
            subplot_titles=[f'{col1} vs {col2}' for col1 in numerical_cols for col2 in numerical_cols]
        )
        
        for i, col1 in enumerate(numerical_cols):
            for j, col2 in enumerate(numerical_cols):
                fig.add_trace(
                    go.Scatter(
                        x=data[col1],
                        y=data[col2],
                        mode='markers',
                        marker=dict(
                            color=data[target],
                            colorscale='Viridis',
                            showscale=True
                        ),
                        name=f'{col1} vs {col2}',
                        showlegend=False
                    ),
                    row=i+1,
                    col=j+1
                )
        
        fig.update_layout(
            height=300*n_cols,
            width=300*n_cols,
            title_text="Feature Relationships",
            showlegend=False
        )
        
        fig.write_html(self.output_dir / 'feature_relationships.html')
    
    def create_time_series_plot(self, data: pd.DataFrame, time_col: str,
                              value_col: str, group_col: Optional[str] = None) -> None:
        """Create interactive time series plot."""
        logger.info(f"Creating time series plot for {value_col}")
        
        fig = px.line(
            data,
            x=time_col,
            y=value_col,
            color=group_col,
            title=f'Time Series: {value_col}'
        )
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title=value_col,
            hovermode='x unified'
        )
        
        fig.write_html(self.output_dir / f'time_series_{value_col}.html')
    
    def create_box_plots(self, data: pd.DataFrame, target: str) -> None:
        """Create box plots for numerical features grouped by target."""
        logger.info("Creating box plots")
        
        numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
        
        for col in numerical_cols:
            fig = px.box(
                data,
                x=target,
                y=col,
                title=f'Box Plot: {col} by {target}'
            )
            
            fig.write_html(self.output_dir / f'box_plot_{col}.html')
    
    def create_3d_surface_plot(self, data: pd.DataFrame, x: str, y: str, z: str) -> None:
        """Create 3D surface plot for continuous features."""
        logger.info(f"Creating 3D surface plot for {x}, {y}, {z}")
        
        # Create meshgrid
        x_range = np.linspace(data[x].min(), data[x].max(), 50)
        y_range = np.linspace(data[y].min(), data[y].max(), 50)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Fit polynomial surface
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(data[[x, y]])
        model = LinearRegression()
        model.fit(X_poly, data[z])
        
        # Create surface
        Z = np.zeros_like(X)
        for i in range(len(X)):
            for j in range(len(X[0])):
                point = poly.transform([[X[i,j], Y[i,j]]])
                Z[i,j] = model.predict(point)[0]
        
        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
        
        fig.update_layout(
            title=f'3D Surface Plot: {z} vs {x} and {y}',
            scene=dict(
                xaxis_title=x,
                yaxis_title=y,
                zaxis_title=z
            )
        )
        
        fig.write_html(self.output_dir / f'3d_surface_{x}_{y}_{z}.html')
    
    def create_all_visualizations(self, data: pd.DataFrame, target: Optional[str] = None) -> None:
        """Create all configured visualizations."""
        logger.info("Starting visualization pipeline")
        
        # Create basic visualizations
        self.create_distribution_plots(data)
        self.create_correlation_heatmap(data)
        
        if target is not None:
            self.create_feature_relationships(data, target)
            self.create_box_plots(data, target)
        
        # Create 3D visualizations
        numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_cols) >= 3:
            self.create_3d_scatter_plot(data, numerical_cols[0], numerical_cols[1], numerical_cols[2], target)
            self.create_3d_pca_visualization(data, target)
        
        # Create time series plots if datetime columns exist
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            for value_col in numerical_cols:
                self.create_time_series_plot(data, datetime_cols[0], value_col, target)
        
        logger.info("Visualization pipeline completed") 