import open3d as o3d
import numpy as np
import streamlit as st
import time
import re
import os
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

def validate_mesh_watertight(mesh: o3d.geometry.TriangleMesh) -> None:
    """Basic mesh validation without strict watertight requirement."""
    if not mesh.has_triangles():
        raise ValueError("Invalid mesh: No triangles found")
    if len(mesh.triangles) < 100:
        raise ValueError("Mesh has too few triangles (<100)")

@st.cache_resource(show_spinner="Loading mesh...")
def load_mesh(stl_path: str) -> o3d.geometry.TriangleMesh:
    """Enhanced mesh loading with lenient validation."""
    try:
        # Security checks
        if not os.path.exists(stl_path):
            raise FileNotFoundError(f"File not found: {stl_path}")
            
        # File size validation
        file_size = os.path.getsize(stl_path)
        max_size = 250 * 1024 * 1024  # 250MB
        if file_size > max_size:
            raise ValueError(f"File exceeds size limit ({file_size/1e6:.1f}MB > {max_size/1e6}MB)")
        
        # Load mesh with validation
        mesh = o3d.io.read_triangle_mesh(
            stl_path,
            enable_post_processing=True,
            print_progress=False
        )
        
        if not mesh.has_triangles():
            raise ValueError("Invalid mesh: No triangles found")
            
        if len(mesh.triangles) < 100:
            raise ValueError("Mesh has too few triangles (<100)")
            
        # Clean mesh with explicit parameters
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        # Ensure mesh is properly oriented
        mesh.orient_triangles()
        
        # Ensure mesh has vertex normals
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
            
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
    """Optimized point cloud sampling with density estimation."""
    try:
        # Adaptive sampling for large meshes
        if num_points > 50000:
            mesh = mesh.simplify_vertex_clustering(
                voxel_size=0.1*mesh.get_max_bound().max()
            )
            
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        
        # Statistical outlier removal with validation
        pcd_clean, inliers = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        
        if len(pcd_clean.points) < 1000:
            raise ValueError("Too few points remaining after outlier removal")
            
        # Estimate normals for cleaned point cloud
        pcd_clean.estimate_normals()
        
        return pcd_clean
        
    except Exception as e:
        st.error(f"Point cloud sampling failed: {str(e)}")
        raise

def compute_advanced_metrics(
    source_aligned: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    target_kdtree: o3d.geometry.KDTreeFlann,
    target_normals: np.ndarray
) -> Dict[str, float]:
    """Enhanced metrics calculation with normal comparison."""
    metrics = {}
    
    try:
        # Compute point distances
        source_points = np.asarray(source_aligned.points)
        target_points = np.asarray(target.points)
        
        # For each source point, find nearest target point
        distances = []
        normal_angles = []
        for src_point, src_normal in zip(source_points, 
                                       np.asarray(source_aligned.normals)):
            _, idx, _ = target_kdtree.search_knn_vector_3d(src_point, 1)
            distance = np.linalg.norm(src_point - target_points[idx[0]])
            angle = np.degrees(np.arccos(
                np.dot(src_normal, target_normals[idx[0]])
            ))
            
            distances.append(distance)
            normal_angles.append(angle)
            
        distances = np.array(distances)
        normal_angles = np.array(normal_angles)
        
        # Distance metrics
        metrics["max_deviation"] = float(distances.max())
        metrics["mean_deviation"] = float(distances.mean())
        metrics["median_deviation"] = float(np.median(distances))
        metrics["std_deviation"] = float(distances.std())
        
        # Normal alignment metrics
        metrics["mean_normal_angle"] = float(normal_angles.mean())
        metrics["max_normal_angle"] = float(normal_angles.max())
        
        # Volume comparison
        source_vol = source_aligned.get_axis_aligned_bounding_box().volume()
        target_vol = target.get_axis_aligned_bounding_box().volume()
        metrics["volume_difference"] = abs(source_vol - target_vol)
        metrics["volume_similarity"] = min(source_vol, target_vol) / max(source_vol, target_vol)
        
        # Surface area comparison
        source_area = np.linalg.norm(np.asarray(source_aligned.normals), axis=1).sum()
        target_area = np.linalg.norm(target_normals, axis=1).sum()
        metrics["area_difference"] = abs(source_area - target_area)
        
        return metrics
        
    except Exception as e:
        st.error(f"Metrics calculation failed: {str(e)}")
        return {"error": str(e)}

def validate_file_name(filename: str) -> bool:
    """Enhanced filename validation with pattern checks."""
    if not re.match(r"^[\w\-. ]+\.stl$", filename, re.IGNORECASE):
        return False
    if ".." in filename or "/" in filename or "\\" in filename:
        return False
    return True
