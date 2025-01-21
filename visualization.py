import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import List, Tuple, Dict, Optional

def plot_points(points: np.ndarray, point_size: int, title: str) -> go.Figure:
    """Create a 3D scatter plot of points."""
    fig = go.Figure(data=[
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color='blue',
                opacity=0.8
            ),
            hovertemplate=(
                "X: %{x:.2f}<br>" +
                "Y: %{y:.2f}<br>" +
                "Z: %{z:.2f}<br>" +
                "<extra></extra>"
            )
        )
    ])
    
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode='data',
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)"
        ),
        showlegend=False
    )
    
    return fig

def plot_deviation_histogram(
    max_dev: float,
    mean_dev: float,
    std_dev: float,
    title: str = "Deviation Distribution"
) -> go.Figure:
    """Create a histogram of deviations with statistics."""
    # Generate sample data for visualization
    x = np.linspace(0, max_dev, 100)
    y = np.exp(-(x - mean_dev)**2 / (2 * std_dev**2))
    
    fig = go.Figure()
    
    # Add distribution curve
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='Distribution',
        line=dict(color='blue')
    ))
    
    # Add mean line
    fig.add_trace(go.Scatter(
        x=[mean_dev, mean_dev],
        y=[0, np.max(y)],
        mode='lines',
        name='Mean',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Deviation (mm)",
        yaxis_title="Density",
        showlegend=True
    )
    
    return fig
