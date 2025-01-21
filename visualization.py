import plotly.graph_objects as go
import numpy as np
import streamlit as st
from typing import List, Tuple, Optional

def plot_point_cloud_heatmap(
    points: np.ndarray,
    values: np.ndarray,
    point_size: int,
    color_scale: str,
    title: str,
    show_colorbar: bool = True
) -> go.Figure:
    """
    Enhanced 3D scatter plot with heatmap coloring.
    
    Args:
        points: Nx3 array of point coordinates
        values: N array of scalar values for coloring
        point_size: Size of points in visualization
        color_scale: Name of the colormap
        title: Plot title
        show_colorbar: Whether to show the colorbar
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure(data=[
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=values,
                colorscale=color_scale,
                showscale=show_colorbar,
                colorbar=dict(
                    title="Deviation (mm)",
                    titleside="right"
                )
            )
        )
    ])
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            aspectmode='data',
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)"
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig

def plot_multiple_point_clouds(
    pcd_data: List[Tuple[np.ndarray, str, str]],
    point_size: int
) -> go.Figure:
    """
    Enhanced visualization for multiple point clouds.
    
    Args:
        pcd_data: List of tuples (points, color, label)
        point_size: Size of points in visualization
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    for points, color, label in pcd_data:
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=color,
                    opacity=0.7
                ),
                name=label
            )
        )
    
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)"
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def plot_deviation_histogram(
    distances: np.ndarray,
    bins: int = 50,
    title: str = "Deviation Distribution"
) -> go.Figure:
    """
    Create a histogram of deviation values.
    
    Args:
        distances: Array of deviation values
        bins: Number of histogram bins
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure(data=[
        go.Histogram(
            x=distances,
            nbinsx=bins,
            name="Deviations"
        )
    ])
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Deviation (mm)",
        yaxis_title="Count",
        bargap=0.1
    )
    
    return fig
