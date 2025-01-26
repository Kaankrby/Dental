import os
os.environ['OPEN3D_CPU_RENDERING'] = 'true'

import plotly.graph_objects as go
import numpy as np
from typing import List, Tuple

def plot_point_cloud_heatmap(
    points: np.ndarray,
    values: np.ndarray,
    point_size: int,
    color_scale: str,
    title: str
) -> go.Figure:
    """3D scatter plot with heatmap coloring for deviations."""
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
                showscale=True,
                colorbar=dict(
                    title="Deviation (mm)",
                    titleside="right"
                ),
                opacity=0.8
            ),
            hovertemplate=(
                "X: %{x:.2f} mm<br>"
                "Y: %{y:.2f} mm<br>"
                "Z: %{z:.2f} mm<br>"
                "Deviation: %{marker.color:.3f} mm<extra></extra>"
            )
        )
    ])
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        scene=dict(
            aspectmode='data',
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                up=dict(x=0, y=0, z=1)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=600
    )
    return fig

def plot_multiple_point_clouds(
    pcd_data: list[tuple[np.ndarray, str, str]],
    point_size: int
) -> go.Figure:
    """
    Visualize multiple point clouds in different colors
    
    Parameters:
    pcd_data: List of tuples (points_array, color_hex, label)
    point_size: Size of points in visualization
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
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig

def plot_deviation_histogram(
    data: np.ndarray,
    bins: int = 50,
    title: str = "Deviation Distribution"
) -> go.Figure:
    """Histogram of deviation values with statistical markers."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=bins,
        marker_color='#636EFA',
        opacity=0.75,
        name="Deviations"
    ))
    
    mean_val = np.mean(data)
    std_val = np.std(data)
    
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                 annotation_text=f"Mean: {mean_val:.3f}")
    fig.add_vline(x=mean_val + std_val, line_dash="dot", line_color="orange")
    fig.add_vline(x=mean_val - std_val, line_dash="dot", line_color="orange")
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Deviation (mm)",
        yaxis_title="Count",
        bargap=0.05,
        height=400
    )
    return fig

def plot_normal_angle_distribution(
    points: np.ndarray,
    angles: np.ndarray,
    point_size: int,
    color_scale: str
) -> go.Figure:
    """3D visualization of normal angle deviations."""
    fig = go.Figure(data=[
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=angles,
                colorscale=color_scale,
                showscale=True,
                colorbar=dict(
                    title="Normal Angle (°)",
                    titleside="right"
                ),
                opacity=0.8
            ),
            hovertemplate=(
                "X: %{x:.2f} mm<br>"
                "Y: %{y:.2f} mm<br>"
                "Z: %{z:.2f} mm<br>"
                "Angle: %{marker.color:.1f}°<extra></extra>"
            )
        )
    ])
    
    fig.update_layout(
        title=dict(text="Normal Angle Distribution", x=0.5, xanchor='center'),
        scene=dict(
            aspectmode='data',
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                up=dict(x=0, y=0, z=1)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=600
    )
    return fig