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
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            end_time = time.time()
            st.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
    return wrapper

@performance_monitor
def load_mesh(stl_path: str) -> o3d.geometry.PointCloud:
    """Load mesh as single merged point cloud"""
    try:
        # Try Open3D first
        mesh = o3d.io.read_triangle_mesh(stl_path)
        if mesh.has_vertices():
            pcd = o3d.geometry.PointCloud()
            pcd.points = mesh.vertices
            return pcd
            
        # Fallback to raw point cloud reading
        with open(stl_path, 'rb') as f:
            data = f.read()
            
        # Skip header and process as raw points
        points = []
        pos = 80  # Skip header
        while pos + 12 <= len(data):  # Read 3 floats at a time
            try:
                point = np.frombuffer(data[pos:pos+12], dtype=np.float32)
                points.append(point)
                pos += 50  # Skip to next potential point
            except:
                pos += 1  # Try next byte
                
        if not points:
            raise ValueError("No valid points found in file")
            
        # Convert to array and remove duplicates
        points = np.unique(np.vstack(points), axis=0)
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        st.info(f"Recovered {len(points)} unique points")
        return pcd
        
    except Exception as e:
        st.error(f"Error loading mesh: {str(e)}")
        raise

def load_ascii_stl(path: str) -> o3d.geometry.PointCloud:
    """Handle malformed ASCII STLs"""
    vertices = []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if 'vertex' in line:
                try:
                    parts = line.strip().split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except:
                    continue
    if not vertices:
        raise ValueError("No vertices found in ASCII STL")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.unique(vertices, axis=0))
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
