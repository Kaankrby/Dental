import plotly.graph_objects as go
import numpy as np
import streamlit as st
from typing import List, Tuple, Optional

def plot_cavity_deviation_map(
    points: np.ndarray,
    values: np.ndarray,
    point_size: int,
    color_scale: str,
    title: str,
    show_colorbar: bool = True,
    tolerance_range: float = 0.5
) -> go.Figure:
    """
    Create a 3D deviation map specifically for cavity analysis.
    
    Args:
        points: Nx3 array of point coordinates
        values: N array of deviation values
        point_size: Size of points in visualization
        color_scale: Name of the colormap
        title: Plot title
        show_colorbar: Whether to show the colorbar
        tolerance_range: Acceptable deviation range in mm
        
    Returns:
        Plotly figure object
    """
    # Create custom color ranges for under/over preparation
    colors = np.where(values > tolerance_range, '#FF0000',  # Red for underprepared
             np.where(values < -tolerance_range, '#0000FF',  # Blue for overprepared
             '#00FF00'))  # Green for within tolerance
    
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
                ),
                cmin=-tolerance_range,
                cmax=tolerance_range
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

def plot_cavity_analysis_summary(
    metrics: dict,
    title: str = "Cavity Preparation Analysis"
) -> go.Figure:
    """
    Create a summary visualization of cavity preparation metrics.
    
    Args:
        metrics: Dictionary containing cavity analysis metrics
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    # Calculate percentages
    total_points = metrics['points_in_cavity']
    under_prep_pct = (metrics['underprepared_points'] / total_points) * 100
    over_prep_pct = (metrics['overprepared_points'] / total_points) * 100
    good_prep_pct = 100 - under_prep_pct - over_prep_pct
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            name='Underprepared',
            x=['Preparation Quality'],
            y=[under_prep_pct],
            marker_color='red'
        ),
        go.Bar(
            name='Good',
            x=['Preparation Quality'],
            y=[good_prep_pct],
            marker_color='green'
        ),
        go.Bar(
            name='Overprepared',
            x=['Preparation Quality'],
            y=[over_prep_pct],
            marker_color='blue'
        )
    ])
    
    fig.update_layout(
        title=title,
        yaxis_title="Percentage of Points",
        barmode='stack',
        showlegend=True,
        yaxis_range=[0, 100]
    )
    
    return fig

def plot_deviation_histogram(
    distances: np.ndarray,
    bins: int = 50,
    title: str = "Deviation Distribution",
    tolerance_range: float = 0.5
) -> go.Figure:
    """
    Create a histogram of cavity deviation values with tolerance ranges.
    
    Args:
        distances: Array of deviation values
        bins: Number of histogram bins
        title: Plot title
        tolerance_range: Acceptable deviation range in mm
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add main histogram
    fig.add_trace(go.Histogram(
        x=distances,
        nbinsx=bins,
        name="Deviations"
    ))
    
    # Add vertical lines for tolerance ranges
    fig.add_vline(x=-tolerance_range, line_dash="dash", line_color="red",
                 annotation_text="Under-preparation limit")
    fig.add_vline(x=tolerance_range, line_dash="dash", line_color="red",
                 annotation_text="Over-preparation limit")
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Deviation (mm)",
        yaxis_title="Count",
        bargap=0.1,
        showlegend=False
    )
    
    return fig

def plot_reference_points(
    points: np.ndarray,
    point_size: int,
    title: str = "Reference Cavity Region"
) -> go.Figure:
    """
    Plot reference cavity region points.
    
    Args:
        points: Nx3 array of point coordinates
        point_size: Size of points in visualization
        title: Plot title
        
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
                color='#00CC96',
                opacity=0.7
            ),
            name="Reference Region"
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
