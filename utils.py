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

@st.cache_resource
def load_mesh(stl_path: str) -> o3d.geometry.TriangleMesh:
    """
    Load and clean an STL mesh with improved error handling and validation.
    Specifically designed for dental models.
    
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
            
        mesh = o3d.io.read_triangle_mesh(stl_path)
        
        # Validate mesh
        if not mesh.has_triangles():
            raise ValueError("Invalid dental model: No triangles found")
            
        # Clean mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_non_manifold_edges()
        
        # Validate cleaned mesh
        if len(mesh.triangles) == 0:
            raise ValueError("Invalid dental model: No valid triangles after cleaning")
            
        return mesh
        
    except Exception as e:
        st.error(f"Error loading dental model: {str(e)}")
        raise

@st.cache_data
def sample_point_cloud(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int,
    nb_neighbors: int,
    std_ratio: float
) -> o3d.geometry.PointCloud:
    """
    Convert mesh to point cloud with dental-specific sampling parameters.
    
    Args:
        mesh: Input dental mesh
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
            st.warning("Warning: Significant point loss during cleaning. Check model quality.")
            
        return pcd_clean
        
    except Exception as e:
        st.error(f"Error in point cloud processing: {str(e)}")
        raise

def compute_cavity_metrics(
    source_aligned: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    ref_bbox: o3d.geometry.AxisAlignedBoundingBox
) -> Dict[str, float]:
    """
    Compute cavity preparation specific metrics.
    
    Args:
        source_aligned: Aligned student model point cloud
        target: Reference model point cloud
        ref_bbox: Bounding box of the cavity region
        
    Returns:
        Dictionary of cavity-specific metrics
    """
    metrics = {}
    
    try:
        # Get points within cavity region
        source_points = np.asarray(source_aligned.points)
        target_points = np.asarray(target.points)
        
        # Compute cavity region metrics
        idx_in_bbox = ref_bbox.get_point_indices_within_bounding_box(source_aligned.points)
        if len(idx_in_bbox) == 0:
            raise ValueError("No points found in cavity region. Check alignment.")
            
        cavity_points = source_points[idx_in_bbox]
        
        # Compute distances only for points in cavity region
        cavity_pcd = source_aligned.select_by_index(idx_in_bbox)
        distances = np.asarray(cavity_pcd.compute_point_cloud_distance(target))
        
        metrics.update({
            "max_deviation": float(np.max(distances)),
            "mean_deviation": float(np.mean(distances)),
            "std_deviation": float(np.std(distances)),
            "points_in_cavity": len(idx_in_bbox),
            "cavity_volume": float(np.prod(ref_bbox.get_extent())),
            "distances": distances,
            "cavity_points": cavity_points
        })
        
        # Compute preparation assessment metrics
        metrics["underprepared_points"] = np.sum(distances > 0.5)  # Points more than 0.5mm from reference
        metrics["overprepared_points"] = np.sum(distances < -0.5)  # Points less than -0.5mm from reference
        
        return metrics
        
    except Exception as e:
        st.error(f"Error computing cavity metrics: {str(e)}")
        return {"error": str(e)}

def validate_file_name(filename: str) -> bool:
    """
    Validate dental model file name.
    
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
