import plotly.graph_objects as go
import numpy as np
from typing import List, Tuple, Optional

def plot_point_cloud_heatmap(
    points: np.ndarray,
    values: np.ndarray,
    point_size: int,
    color_scale: str,
    title: str
) -> go.Figure:
    """Enhanced 3D heatmap plot with improved camera settings."""
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
                    titleside="right",
                    xpad=20
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

def plot_normal_angle_distribution(
    points: np.ndarray,
    angles: np.ndarray,
    point_size: int,
    color_scale: str
) -> go.Figure:
    """Visualize normal angle deviations."""
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
                    titleside="right",
                    xpad=20
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

def plot_deviation_histogram(
    data: np.ndarray,
    bins: int = 50,
    title: str = "Distribution",
    xaxis_title: str = "Deviation (mm)"
) -> go.Figure:
    """Enhanced histogram with statistical annotations."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=bins,
        marker_color='#636EFA',
        opacity=0.75,
        name="Distribution",
        hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>"
    ))
    
    # Add statistical indicators
    mean_val = np.mean(data)
    std_val = np.std(data)
    
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                 annotation_text=f"Mean: {mean_val:.3f}")
    fig.add_vline(x=mean_val+std_val, line_dash="dot", line_color="orange")
    fig.add_vline(x=mean_val-std_val, line_dash="dot", line_color="orange")
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title=xaxis_title,
        yaxis_title="Count",
        bargap=0.05,
        showlegend=False,
        height=400
    )
    return fig


def plot_multiple_point_clouds(  # Make sure this exact name exists
    pcd_data: List[Tuple[np.ndarray, str, str]],
    point_size: int
) -> go.Figure:
    """Enhanced visualization for multiple point clouds."""
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