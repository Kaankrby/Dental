import sys
if sys.version_info >= (3, 10):
    import collections.abc
    collections = sys.modules['collections']
    collections.Mapping = collections.abc.Mapping

import open3d as o3d
import numpy as np
import streamlit as st
import time
import re
import os
from typing import Tuple, List, Dict, Any
from functools import wraps
import tempfile
import hashlib
from stl import mesh as stl_mesh

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
        # First try Open3D's fast path
        mesh = o3d.io.read_triangle_mesh(stl_path)
        if mesh.has_vertices():
            pcd = o3d.geometry.PointCloud()
            pcd.points = mesh.vertices
            return pcd

        # Fallback: Direct binary parsing with safety limits
        with open(stl_path, 'rb') as f:
            header = f.read(80)
            raw_triangle_count = f.read(4)
            
            # Validate realistic file size
            file_size = os.path.getsize(stl_path)
            max_reasonable = 500 * 1024 * 1024  # 500MB
            if file_size > max_reasonable:
                raise ValueError(f"File size {file_size/1e6:.1f}MB exceeds safety limit")
                
            # Calculate actual triangle capacity
            bytes_per_triangle = 50  # 12 floats (4 bytes each) + 2 byte attr
            max_triangles = (file_size - 84) // bytes_per_triangle  # 84=header
            
            # Read triangles in chunks
            chunk_size = 100000
            vertices = []
            while True:
                chunk = f.read(bytes_per_triangle * chunk_size)
                if not chunk:
                    break
                    
                # Convert to numpy array
                data = np.frombuffer(chunk, dtype=np.float32)
                triangles = data.reshape(-1, 12)[:, :9]  # Ignore normals+attributes
                vertices.append(triangles.reshape(-1, 3))
                
            if not vertices:
                raise ValueError("No vertices found in binary STL")
                
            all_vertices = np.unique(np.vstack(vertices), axis=0)
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(all_vertices)
            return pcd

    except Exception as e:
        st.error(f"""
            STL Load Error: {str(e)}
            File: {os.path.basename(stl_path)}
            Size: {os.path.getsize(stl_path)/1e6:.1f}MB
            Header: {header[:20] if 'header' in locals() else 'N/A'}
        """)
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

def compute_advanced_metrics(
    source_aligned: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud
) -> Dict[str, Any]:
    """Compute comparison metrics."""
    # Calculate distances
    dists = source_aligned.compute_point_cloud_distance(target)
    distances = np.asarray(dists)
    
    # Basic statistics
    mean_dev = np.mean(distances)
    max_dev = np.max(distances)
    
    # Calculate volume difference
    source_bbox = source_aligned.get_axis_aligned_bounding_box()
    target_bbox = target.get_axis_aligned_bounding_box()
    source_vol = np.prod(source_bbox.get_extent())
    target_vol = np.prod(target_bbox.get_extent())
    volume_diff = abs(source_vol - target_vol)
    volume_sim = min(source_vol, target_vol) / max(source_vol, target_vol)
    
    # Calculate center of mass distance
    source_com = np.mean(np.asarray(source_aligned.points), axis=0)
    target_com = np.mean(np.asarray(target.points), axis=0)
    com_dist = np.linalg.norm(source_com - target_com)
    
    # Calculate normal angle differences
    source_normals = np.asarray(source_aligned.normals)
    target_normals = np.asarray(target.normals)
    target_kdtree = o3d.geometry.KDTreeFlann(target)
    
    normal_angles = []
    for i, point in enumerate(source_aligned.points):
        [_, idx, _] = target_kdtree.search_knn_vector_3d(point, 1)
        dot_product = np.clip(np.abs(np.dot(source_normals[i], target_normals[idx[0]])), -1.0, 1.0)
        angle = np.arccos(dot_product)
        normal_angles.append(angle)
    
    normal_angles = np.array(normal_angles)
    mean_normal_angle = float(np.rad2deg(np.mean(normal_angles)))
    
    return {
        'distances': distances,
        'mean_deviation': float(mean_dev),
        'max_deviation': float(max_dev),
        'hausdorff_distance': float(max_dev),
        'volume_difference': float(volume_diff),
        'volume_similarity': float(volume_sim),
        'center_of_mass_distance': float(com_dist),
        'mean_normal_angle': mean_normal_angle,
        'normal_angles': np.rad2deg(normal_angles)  # Convert to degrees for visualization
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
