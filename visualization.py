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
    zones: np.ndarray,
    point_size: int,
    title: str
) -> go.Figure:
    """Create a 3D visualization of cavity preparation zones."""
    zone_colors = {
        0: '#EF553B',  # Margin
        1: '#00CC96',  # Walls
        2: '#AB63FA',  # Floor
    }
    
    fig = go.Figure()
    
    for zone_id in np.unique(zones):
        zone_points = points[zones == zone_id]
        zone_name = ['Margin', 'Walls', 'Floor'][zone_id]
        
        fig.add_trace(go.Scatter3d(
            x=zone_points[:, 0],
            y=zone_points[:, 1],
            z=zone_points[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=zone_colors[zone_id],
                opacity=0.7
            ),
            name=zone_name,
            hovertemplate=(
                f"{zone_name}<br>" +
                "X: %{x:.2f}<br>" +
                "Y: %{y:.2f}<br>" +
                "Z: %{z:.2f}<br>" +
                "<extra></extra>"
            )
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
