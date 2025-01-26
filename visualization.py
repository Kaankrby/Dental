import os
os.environ['OPEN3D_CPU_RENDERING'] = 'true'

import plotly.graph_objects as go
import numpy as np
from typing import List, Tuple
import rhino3dm as rh
import plotly.express as px
import open3d as o3d
import copy

def plot_point_cloud_heatmap(pcd: o3d.geometry.PointCloud) -> go.Figure:
    """Create a heatmap visualization of point cloud"""
    points = np.asarray(pcd.points)
    
    # Create figure
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=points[:, 2],  # Color by Z coordinate
            colorscale='Viridis',
            showscale=True
        )
    )])
    
    # Update layout
    fig.update_layout(
        scene=dict(
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    
    return fig

def plot_multiple_point_clouds(
    point_clouds: List[o3d.geometry.PointCloud],
    labels: List[str],
    colors: List[str] = None
) -> go.Figure:
    """Plot multiple point clouds with different colors"""
    if colors is None:
        colors = ['blue', 'red', 'green', 'yellow']
    
    fig = go.Figure()
    
    for pcd, label, color in zip(point_clouds, labels, colors):
        points = np.asarray(pcd.points)
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            name=label,
            marker=dict(
                size=2,
                color=color,
                opacity=0.7
            )
        ))
    
    fig.update_layout(
        scene=dict(
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    
    return fig

def plot_deviation_histogram(
    distances: np.ndarray,
    title: str = "Point Cloud Distances"
) -> go.Figure:
    """Create histogram of point cloud distances"""
    fig = go.Figure(data=[go.Histogram(
        x=distances,
        nbinsx=50,
        name="Distance Distribution"
    )])
    
    fig.update_layout(
        title=title,
        xaxis_title="Distance (mm)",
        yaxis_title="Count",
        showlegend=False
    )
    
    return fig

def plot_registration_result(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray
) -> go.Figure:
    """Visualize registration result"""
    # Transform source point cloud
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(transformation)
    
    # Plot both point clouds
    return plot_multiple_point_clouds(
        [source_transformed, target],
        ["Aligned Source", "Target"],
        ["blue", "red"]
    )

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

def plot_rhino_model(model: rh.File3dm) -> go.Figure:
    """Create 3D visualization of Rhino model layers"""
    fig = go.Figure()
    
    # Color mapping for layers
    colors = px.colors.qualitative.Plotly
    
    for i, layer in enumerate(model.Layers):
        # Get layer meshes
        meshes = [obj.Geometry for obj in model.Objects 
                if obj.Attributes.LayerIndex == i 
                and isinstance(obj.Geometry, rh.Mesh)]
        
        # Combine vertices
        vertices = []
        for mesh in meshes:
            vertices.extend([[v.X, v.Y, v.Z] for v in mesh.Vertices])
        
        if not vertices:
            continue
            
        # Add to plot
        vertices = np.array(vertices)
        fig.add_trace(go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=colors[i % len(colors)],
                opacity=0.7
            ),
            name=f"{layer.Name} Layer"
        ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data'
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig