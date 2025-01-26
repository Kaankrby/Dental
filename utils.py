import open3d as o3d
import numpy as np
import streamlit as st
import time
import re
import os
from typing import Tuple, List, Dict, Any, Optional
from functools import wraps
import tempfile

def performance_monitor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        st.sidebar.metric(f"{func.__name__} Time", f"{execution_time:.2f}s")
        return result
    return wrapper

def load_mesh(stl_path: str) -> o3d.geometry.PointCloud:
    try:
        # Try both point cloud and mesh formats
        pcd = o3d.io.read_point_cloud(stl_path)
        if pcd.has_points():
            return pcd
            
        # Explicitly read as STL mesh
        mesh = o3d.io.read_triangle_mesh(stl_path, enable_post_processing=True)
        
        if not mesh.has_vertices():
            raise ValueError("STL file contains no readable vertices")
            
        # Create point cloud from vertices if no faces
        if not mesh.has_triangles():
            pcd = o3d.geometry.PointCloud()
            pcd.points = mesh.vertices
            return pcd
            
        # Smart sampling based on mesh density
        num_points = max(1000, len(mesh.vertices) // 10)
        return mesh.sample_points_uniformly(num_points)
        
    except Exception as e:
        st.error(f"STL Load Error: {str(e)}")
        raise

def sample_point_cloud(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int,
    nb_neighbors: int,
    std_ratio: float
) -> o3d.geometry.PointCloud:
    """Sample point cloud from mesh."""
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    pcd.estimate_normals()
    pcd = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)[0]
    return pcd

def _safe_mesh_volume(mesh: Optional[o3d.geometry.TriangleMesh]) -> Optional[float]:
    """Best-effort attempt to retrieve a mesh volume."""
    if mesh is None:
        return None

    try:
        volume = mesh.get_volume()
        if not np.isfinite(volume):
            return None
        return abs(float(volume))
    except Exception:
        return None


def compute_advanced_metrics(
    source_aligned: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    source_mesh: Optional[o3d.geometry.TriangleMesh] = None,
    target_mesh: Optional[o3d.geometry.TriangleMesh] = None
) -> Dict[str, Any]:
    """Compute comparison metrics."""
    # Calculate distances
    forward_dists = source_aligned.compute_point_cloud_distance(target)
    distances = np.asarray(forward_dists)
    backward_dists = target.compute_point_cloud_distance(source_aligned)
    backward_distances = np.asarray(backward_dists)
    
    # Basic statistics
    mean_dev = float(np.mean(distances)) if distances.size else 0.0
    max_dev = float(np.max(distances)) if distances.size else 0.0
    max_dev_backward = float(np.max(backward_distances)) if backward_distances.size else 0.0
    symmetric_hausdorff = max(max_dev, max_dev_backward)
    
    # Calculate volume difference
    source_volume = _safe_mesh_volume(source_mesh)
    target_volume = _safe_mesh_volume(target_mesh)

    if source_volume is None or target_volume is None or max(source_volume, target_volume) == 0:
        source_bbox = source_aligned.get_oriented_bounding_box()
        target_bbox = target.get_oriented_bounding_box()
        source_volume = np.prod(source_bbox.extent)
        target_volume = np.prod(target_bbox.extent)

    volume_diff = abs(source_volume - target_volume)
    volume_sim = min(source_volume, target_volume) / max(source_volume, target_volume) if max(source_volume, target_volume) else 0.0
    
    # Calculate center of mass distance
    source_com = np.mean(np.asarray(source_aligned.points), axis=0)
    target_com = np.mean(np.asarray(target.points), axis=0)
    com_dist = np.linalg.norm(source_com - target_com)
    
    # Calculate normal angle differences
    source_normals = np.asarray(source_aligned.normals)
    target_normals = np.asarray(target.normals)
    normal_angles = []

    if len(source_normals) and len(target_normals):
        target_kdtree = o3d.geometry.KDTreeFlann(target)
        for i, point in enumerate(source_aligned.points):
            if i >= len(source_normals):
                break
            [_, idx, _] = target_kdtree.search_knn_vector_3d(point, 1)
            dot_product = np.clip(np.abs(np.dot(source_normals[i], target_normals[idx[0]])), -1.0, 1.0)
            angle = np.arccos(dot_product)
            normal_angles.append(angle)

    normal_angles = np.array(normal_angles, dtype=float) if normal_angles else np.array([], dtype=float)
    mean_normal_angle = float(np.rad2deg(np.mean(normal_angles))) if normal_angles.size else 0.0
    
    return {
        'distances': distances,
        'mean_deviation': float(mean_dev),
        'max_deviation': float(max_dev),
        'hausdorff_distance': float(symmetric_hausdorff),
        'hausdorff_forward': float(max_dev),
        'hausdorff_backward': float(max_dev_backward),
        'volume_difference': float(volume_diff),
        'volume_similarity': float(volume_sim),
        'center_of_mass_distance': float(com_dist),
        'mean_normal_angle': mean_normal_angle,
        'normal_angles': np.rad2deg(normal_angles) if normal_angles.size else np.array([])
    }

def validate_file_name(filename: str) -> bool:
    """Validate filename."""
    pattern = r'^[\w\-. ]+\.stl$'
    return bool(re.match(pattern, filename, re.IGNORECASE))

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as f:
        f.write(uploaded_file.getbuffer())
        return f.name

def validate_3dm_file(file_path: str):
    """Check if file is valid Rhino .3dm"""
    try:
        model = rh.File3dm.Read(file_path)
        if model.Objects.Count == 0:
            raise ValueError("No meshes found in .3dm file")
        return True
    except Exception as e:
        st.error(f"Invalid .3dm file: {str(e)}")
        return False

def validate_stl_file(file_path: str):
    """Check STL file validity"""
    try:
        mesh = o3d.io.read_triangle_mesh(file_path)
        if not mesh.has_triangles():
            raise ValueError("Invalid STL - no triangles found")
        return True
    except Exception as e:
        st.error(f"STL Error: {str(e)}")
        return False
