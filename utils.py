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
        st.sidebar.metric(f"{func.__name__} Time", f"{execution_time:.2f}s")
        return result
    return wrapper

@st.cache_resource(show_spinner="Loading mesh...")
def load_mesh(stl_path: str) -> o3d.geometry.TriangleMesh:
    """Load mesh without watertight validation"""
    try:
        mesh = o3d.io.read_triangle_mesh(stl_path)
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        return mesh
    except Exception as e:
        st.error(f"Mesh loading failed: {str(e)}")
        raise

@st.cache_data
def sample_point_cloud(mesh: o3d.geometry.TriangleMesh, num_points: int, 
                      nb_neighbors: int, std_ratio: float) -> o3d.geometry.PointCloud:
    """Convert mesh to cleaned point cloud"""
    try:
        pcd = mesh.sample_points_uniformly(num_points)
        pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
        pcd_clean.estimate_normals()
        return pcd_clean
    except Exception as e:
        st.error(f"Point cloud sampling failed: {str(e)}")
        raise

def compute_advanced_metrics(source_pcd: o3d.geometry.PointCloud, 
                            target_pcd: o3d.geometry.PointCloud) -> Dict[str, float]:
    """Point cloud-based metrics calculation"""
    metrics = {}
    
    # Compute point distances
    distances = source_pcd.compute_point_cloud_distance(target_pcd)
    distances = np.asarray(distances)
    
    metrics.update({
        "mean_deviation": float(np.mean(distances)),
        "max_deviation": float(np.max(distances)),
        "median_deviation": float(np.median(distances)),
        "std_deviation": float(np.std(distances)),
        "hausdorff_distance": float(np.max(distances))
    })
    
    # Convex hull volume comparison
    source_hull = source_pcd.compute_convex_hull()[0]
    target_hull = target_pcd.compute_convex_hull()[0]
    metrics["volume_difference"] = abs(source_hull.volume - target_hull.volume)
    
    return metrics

def validate_file_name(filename: str) -> bool:
    """Enhanced filename validation"""
    import re
    return bool(re.match(r"^[\w\-. ]+\.stl$", filename, re.IGNORECASE))