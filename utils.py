import open3d as o3d
import numpy as np
import streamlit as st
import time
from typing import Tuple, List, Dict, Any
from functools import wraps

def performance_monitor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        st.sidebar.text(f"{func.__name__} took {execution_time:.2f}s")
        return result
    return wrapper

@st.cache_resource(show_spinner=False)
def load_mesh(stl_path: str) -> o3d.geometry.TriangleMesh:
    """
    Load and clean an STL mesh with improved error handling and validation.
    
    Args:
        stl_path: Path to the STL file
        
    Returns:
        Cleaned mesh object
        
    Raises:
        ValueError: If mesh is invalid or empty
    """
    try:
        # Validate file size
        import os
        file_size = os.path.getsize(stl_path)
        max_size = 100 * 1024 * 1024  # 100MB limit
        if file_size > max_size:
            raise ValueError(f"File too large: {file_size/1024/1024:.1f}MB (max {max_size/1024/1024}MB)")
        
        # Load mesh with explicit options
        mesh = o3d.io.read_triangle_mesh(
            stl_path,
            enable_post_processing=True,
            print_progress=False
        )
        
        if not mesh.has_triangles():
            raise ValueError("Mesh has no triangles")
        
        # Ensure mesh has vertex normals
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        
        # Clean mesh with explicit parameters
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        # Ensure mesh is properly oriented
        mesh.orient_triangles()
        
        # Validate cleaned mesh
        if len(mesh.triangles) == 0:
            raise ValueError("Mesh has no valid triangles after cleaning")
            
        return mesh
        
    except Exception as e:
        st.error(f"Error loading mesh: {str(e)}")
        raise

@st.cache_data
def sample_point_cloud(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int,
    nb_neighbors: int,
    std_ratio: float
) -> o3d.geometry.PointCloud:
    """
    Convert mesh to point cloud with improved sampling and validation.
    
    Args:
        mesh: Input mesh
        num_points: Number of points to sample
        nb_neighbors: Number of neighbors for outlier removal
        std_ratio: Standard deviation ratio for outlier removal
        
    Returns:
        Cleaned point cloud
    """
    try:
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        
        # Remove outliers
        pcd_clean, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        
        if len(pcd_clean.points) < num_points * 0.5:
            st.warning(f"Warning: More than 50% of points removed as outliers")
            
        return pcd_clean
        
    except Exception as e:
        st.error(f"Error in point cloud sampling: {str(e)}")
        raise

def compute_advanced_metrics(
    source_aligned: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud
) -> Dict[str, float]:
    """
    Compute advanced comparison metrics between point clouds.
    
    Args:
        source_aligned: Aligned source point cloud
        target: Target point cloud
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    try:
        # Hausdorff distance
        distances = compute_deviations(source_aligned, target)
        metrics["hausdorff_distance"] = float(np.max(distances))
        
        # Point cloud statistics
        source_points = np.asarray(source_aligned.points)
        target_points = np.asarray(target.points)
        
        # Bounding box volume difference
        source_bbox = source_aligned.get_axis_aligned_bounding_box()
        target_bbox = target.get_axis_aligned_bounding_box()
        source_vol = np.prod(source_bbox.get_extent())
        target_vol = np.prod(target_bbox.get_extent())
        metrics["volume_difference"] = abs(source_vol - target_vol)
        
        # Center of mass difference
        source_com = np.mean(source_points, axis=0)
        target_com = np.mean(target_points, axis=0)
        metrics["center_of_mass_distance"] = float(np.linalg.norm(source_com - target_com))
        
        return metrics
        
    except Exception as e:
        st.error(f"Error computing advanced metrics: {str(e)}")
        return {"error": str(e)}

def validate_file_name(filename: str) -> bool:
    """
    Validate file name for security.
    
    Args:
        filename: Name of the file to validate
        
    Returns:
        True if filename is valid, False otherwise
    """
    import re
    # Check for valid characters
    if not re.match("^[a-zA-Z0-9_.-]+$", filename):
        return False
    # Check for valid extension
    if not filename.lower().endswith('.stl'):
        return False
    return True
