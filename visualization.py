import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import List, Tuple, Dict

def plot_cavity_deviation_map(
    points: np.ndarray,
    deviations: np.ndarray,
    point_size: int,
    colorscale: str,
    title: str,
    tolerance_range: float = 0.5
) -> go.Figure:
    """Create an interactive 3D cavity deviation map."""
    # Create custom color scale centered at 0
    max_abs_dev = max(abs(deviations.min()), abs(deviations.max()))
    
    fig = go.Figure(data=[
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=deviations,
                colorscale=colorscale,
                colorbar=dict(
                    title="Deviation (mm)",
                    ticksuffix=" mm"
                ),
                cmin=-max_abs_dev,
                cmax=max_abs_dev,
                showscale=True
            ),
            hovertemplate=(
                "X: %{x:.2f}<br>" +
                "Y: %{y:.2f}<br>" +
                "Z: %{z:.2f}<br>" +
                "Deviation: %{marker.color:.2f} mm<br>" +
                "<extra></extra>"
            )
        )
    ])
    
    # Add tolerance range planes
    x_range = [points[:, 0].min(), points[:, 0].max()]
    y_range = [points[:, 1].min(), points[:, 1].max()]
    z_mid = points[:, 2].mean()
    
    # Add tolerance planes (semi-transparent)
    fig.add_trace(go.Surface(
        x=[[x_range[0], x_range[1]], [x_range[0], x_range[1]]],
        y=[[y_range[0], y_range[0]], [y_range[1], y_range[1]]],
        z=[[z_mid + tolerance_range, z_mid + tolerance_range],
           [z_mid + tolerance_range, z_mid + tolerance_range]],
        opacity=0.2,
        showscale=False,
        name="Upper Tolerance"
    ))
    
    fig.add_trace(go.Surface(
        x=[[x_range[0], x_range[1]], [x_range[0], x_range[1]]],
        y=[[y_range[0], y_range[0]], [y_range[1], y_range[1]]],
        z=[[z_mid - tolerance_range, z_mid - tolerance_range],
           [z_mid - tolerance_range, z_mid - tolerance_range]],
        opacity=0.2,
        showscale=False,
        name="Lower Tolerance"
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)"
        ),
        showlegend=True
    )
    
    return fig

def plot_cavity_analysis_summary(metrics: Dict, title: str) -> go.Figure:
    """Create a summary visualization of cavity preparation quality."""
    total_points = metrics['points_in_cavity']
    within_tolerance = total_points - (metrics['underprepared_points'] + metrics['overprepared_points'])
    
    labels = ['Within Tolerance', 'Underprepared', 'Overprepared']
    values = [
        within_tolerance,
        metrics['underprepared_points'],
        metrics['overprepared_points']
    ]
    colors = ['#00CC96', '#EF553B', '#AB63FA']
    
    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker=dict(colors=colors),
            textinfo='percent+label',
            hovertemplate=(
                "%{label}<br>" +
                "Points: %{value}<br>" +
                "Percentage: %{percent}<br>" +
                "<extra></extra>"
            )
        )
    ])
    
    fig.update_layout(
        title=title,
        annotations=[
            dict(
                text=f"Total Points<br>{total_points}",
                x=0.5, y=0.5,
                font_size=14,
                showarrow=False
            )
        ]
    )
    
    return fig

def plot_deviation_histogram(
    deviations: np.ndarray,
    title: str,
    tolerance_range: float = 0.5,
    bins: int = 50
) -> go.Figure:
    """Create a histogram of deviations with tolerance ranges."""
    fig = go.Figure()
    
    # Add main histogram
    fig.add_trace(go.Histogram(
        x=deviations,
        nbinsx=bins,
        name="Deviations",
        hovertemplate=(
            "Deviation: %{x:.2f} mm<br>" +
            "Count: %{y}<br>" +
            "<extra></extra>"
        )
    ))
    
    # Add tolerance range indicators
    fig.add_vline(
        x=-tolerance_range,
        line_dash="dash",
        line_color="red",
        annotation_text="Lower Tolerance",
        annotation_position="top"
    )
    fig.add_vline(
        x=tolerance_range,
        line_dash="dash",
        line_color="red",
        annotation_text="Upper Tolerance",
        annotation_position="top"
    )
    
    # Add mean and std deviation indicators
    mean_dev = np.mean(deviations)
    std_dev = np.std(deviations)
    
    fig.add_vline(
        x=mean_dev,
        line_dash="solid",
        line_color="green",
        annotation_text=f"Mean: {mean_dev:.2f}mm",
        annotation_position="bottom"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Deviation (mm)",
        yaxis_title="Count",
        showlegend=False,
        annotations=[
            dict(
                x=0.95,
                y=0.95,
                xref="paper",
                yref="paper",
                text=f"Standard Deviation: {std_dev:.2f}mm",
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
        ]
    )
    
    return fig

def plot_reference_points(
    points: np.ndarray,
    point_size: int,
    title: str
) -> go.Figure:
    """Create an interactive 3D visualization of reference cavity points."""
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
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)"
        )
    )
    
    return fig

def plot_preparation_zones(
    points: np.ndarray,
    labels: np.ndarray,
    point_size: int,
    title: str
) -> go.Figure:
    """Create an interactive 3D scatter plot of cavity regions."""
    # Generate colors for each region
    unique_labels = np.unique(labels[labels >= 0])
    colors = px.colors.qualitative.Set3[:len(unique_labels)]
    
    fig = go.Figure()
    
    # Plot each region with a different color
    for i, label in enumerate(unique_labels):
        mask = labels == label
        fig.add_trace(go.Scatter3d(
            x=points[mask, 0],
            y=points[mask, 1],
            z=points[mask, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=colors[i],
                opacity=0.8
            ),
            name=f'Region {label + 1}'
        ))
    
    # Plot noise points if any
    noise_mask = labels == -1
    if np.any(noise_mask):
        fig.add_trace(go.Scatter3d(
            x=points[noise_mask, 0],
            y=points[noise_mask, 1],
            z=points[noise_mask, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color='gray',
                opacity=0.3
            ),
            name='Noise'
        ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data'
        ),
        showlegend=True
    )
    
    return fig

def plot_region(
    points: np.ndarray,
    labels: np.ndarray,
    point_size: int,
    title: str
) -> go.Figure:
    """Create an interactive 3D scatter plot of cavity regions."""
    # Generate colors for each region
    unique_labels = np.unique(labels[labels >= 0])
    colors = px.colors.qualitative.Set3[:len(unique_labels)]
    
    fig = go.Figure()
    
    # Plot each region with a different color
    for i, label in enumerate(unique_labels):
        mask = labels == label
        fig.add_trace(go.Scatter3d(
            x=points[mask, 0],
            y=points[mask, 1],
            z=points[mask, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=colors[i],
                opacity=0.8
            ),
            name=f'Region {label + 1}'
        ))
    
    # Plot noise points if any
    noise_mask = labels == -1
    if np.any(noise_mask):
        fig.add_trace(go.Scatter3d(
            x=points[noise_mask, 0],
            y=points[noise_mask, 1],
            z=points[noise_mask, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color='gray',
                opacity=0.3
            ),
            name='Noise'
        ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data'
        ),
        showlegend=True
    )
    
    return fig
