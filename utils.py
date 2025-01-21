import open3d as o3d
import numpy as np
import streamlit as st
import time
import re
from typing import Tuple, List, Optional, Dict, Any
from functools import wraps

def validate_file_name(filename: str) -> bool:
    """Validate file name for security."""
    return bool(re.match(r'^[\w\-. ]+$', filename))

def load_mesh(file_path: str) -> o3d.geometry.TriangleMesh:
    """Load an STL mesh file using Open3D."""
    try:
        mesh = o3d.io.read_triangle_mesh(file_path)
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        return mesh
    except Exception as e:
        st.error(f"Error loading mesh: {str(e)}")
        raise

def performance_monitor(func):
    """Decorator to monitor function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        st.sidebar.text(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@st.cache_data
def process_mesh_to_points(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int,
    nb_neighbors: int,
    std_ratio: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Process mesh to point cloud and return numpy arrays."""
    # Sample points
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    
    # Remove outliers
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    
    # Convert to numpy arrays for better serialization
    points = np.asarray(cl.points)
    normals = np.asarray(cl.normals)
    
    return points, normals

def sample_point_cloud(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int,
    nb_neighbors: int,
    std_ratio: float
) -> o3d.geometry.PointCloud:
    """Sample and clean point cloud from mesh."""
    points, normals = process_mesh_to_points(mesh, num_points, nb_neighbors, std_ratio)
    
    # Create new point cloud from numpy arrays
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    return pcd

def compute_cavity_metrics(
    test_pcd: o3d.geometry.PointCloud,
    reference_pcd: o3d.geometry.PointCloud,
    reference_bbox: o3d.geometry.AxisAlignedBoundingBox,
    tolerance: float = 0.5
) -> Dict[str, float]:
    """Compute cavity-specific metrics between test and reference models."""
    # Convert to numpy for calculations
    test_points = np.asarray(test_pcd.points)
    ref_points = np.asarray(reference_pcd.points)
    
    # Get points within cavity region
    indices = reference_bbox.get_point_indices_within_bounding_box(test_points)
    cavity_points = test_points[indices]
    
    # Compute point-to-point distances
    distances = []
    for point in cavity_points:
        dist = np.linalg.norm(ref_points - point, axis=1)
        distances.append(np.min(dist))
    distances = np.array(distances)
    
    # Compute metrics
    metrics = {
        'cavity_points': cavity_points,
        'distances': distances,
        'points_in_cavity': len(cavity_points),
        'max_deviation': float(np.max(distances)),
        'mean_deviation': float(np.mean(distances)),
        'std_deviation': float(np.std(distances)),
        'underprepared_points': int(np.sum(distances < -tolerance)),
        'overprepared_points': int(np.sum(distances > tolerance))
    }
    
    return metrics
