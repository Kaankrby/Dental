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
import rhino3dm as rh
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

def estimate_point_spacing(
    pcd: o3d.geometry.PointCloud,
    sample_size: int = 2000,
    k: int = 2
) -> float:
    """Estimate characteristic point spacing (mm) via nearest-neighbor distance.

    Samples up to `sample_size` points and averages distance to the nearest neighbor.
    """
    pts = np.asarray(pcd.points)
    n = len(pts)
    if n < 2:
        return 1.0
    idxs = np.random.choice(n, size=min(sample_size, n), replace=False)
    tree = o3d.geometry.KDTreeFlann(pcd)
    dists = []
    for i in idxs:
        _, ind, _ = tree.search_knn_vector_3d(pcd.points[i], max(k, 2))
        if len(ind) >= 2:
            a = np.asarray(pcd.points[i])
            b = np.asarray(pcd.points[ind[1]])  # skip self at ind[0]
            dists.append(np.linalg.norm(a - b))
    if not dists:
        return 1.0
    return float(np.mean(dists))

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

def rhino_unit_scale_to_mm(unit) -> float:
    """Return scale to convert from given Rhino unit system to millimeters."""
    try:
        mapping = {
            rh.UnitSystem.Microns: 0.001,
            rh.UnitSystem.Millimeters: 1.0,
            rh.UnitSystem.Centimeters: 10.0,
            rh.UnitSystem.Meters: 1000.0,
            rh.UnitSystem.Kilometers: 1_000_000.0,
            rh.UnitSystem.Inches: 25.4,
            rh.UnitSystem.Feet: 304.8,
            rh.UnitSystem.Yards: 914.4,
            rh.UnitSystem.Miles: 1_609_344.0,
        }
        return float(mapping.get(unit, 1.0))
    except Exception:
        return 1.0

def rhino_unit_name(unit) -> str:
    """Human friendly name for Rhino unit system."""
    try:
        names = {
            rh.UnitSystem.Microns: "Microns",
            rh.UnitSystem.Millimeters: "Millimeters",
            rh.UnitSystem.Centimeters: "Centimeters",
            rh.UnitSystem.Meters: "Meters",
            rh.UnitSystem.Kilometers: "Kilometers",
            rh.UnitSystem.Inches: "Inches",
            rh.UnitSystem.Feet: "Feet",
            rh.UnitSystem.Yards: "Yards",
            rh.UnitSystem.Miles: "Miles",
        }
        return names.get(unit, "Millimeters")
    except Exception:
        return "Millimeters"

@performance_monitor
def compute_voxel_overlap_metrics(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float
) -> Dict[str, Any]:
    """Approximate volumetric overlap via voxelization for open meshes.

    Returns approximate volumes (mm^3) and overlap ratios.
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")

    src_pts = np.asarray(source.points)
    tgt_pts = np.asarray(target.points)
    if len(src_pts) == 0 or len(tgt_pts) == 0:
        return {
            'volume_ref_approx': 0.0,
            'volume_test_approx': 0.0,
            'volume_intersection': 0.0,
            'volume_union': 0.0,
            'volume_overlap_jaccard': 0.0,
            'coverage_ref_pct': 0.0,
            'coverage_test_pct': 0.0,
        }

    # Align voxel grid origins using global min bound so indices line up
    min_bound = np.minimum(src_pts.min(axis=0), tgt_pts.min(axis=0))

    src_shift = o3d.geometry.PointCloud(source)
    src_shift.translate(-min_bound)
    tgt_shift = o3d.geometry.PointCloud(target)
    tgt_shift.translate(-min_bound)

    vg_src = o3d.geometry.VoxelGrid.create_from_point_cloud(src_shift, voxel_size)
    vg_tgt = o3d.geometry.VoxelGrid.create_from_point_cloud(tgt_shift, voxel_size)

    vox_src = {tuple(v.grid_index) for v in vg_src.get_voxels()}
    vox_tgt = {tuple(v.grid_index) for v in vg_tgt.get_voxels()}

    inter = vox_src & vox_tgt
    union = vox_src | vox_tgt

    v_unit = voxel_size ** 3
    vol_src = len(vox_src) * v_unit
    vol_tgt = len(vox_tgt) * v_unit
    vol_inter = len(inter) * v_unit
    vol_union = len(union) * v_unit

    jacc = (len(inter) / len(union)) if len(union) else 0.0
    cov_src = (len(inter) / len(vox_src) * 100.0) if len(vox_src) else 0.0
    cov_tgt = (len(inter) / len(vox_tgt) * 100.0) if len(vox_tgt) else 0.0

    return {
        'volume_ref_approx': float(vol_tgt),
        'volume_test_approx': float(vol_src),
        'volume_intersection': float(vol_inter),
        'volume_union': float(vol_union),
        'volume_overlap_jaccard': float(jacc),
        'coverage_ref_pct': float(cov_tgt),
        'coverage_test_pct': float(cov_src),
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
    """Check if file is valid Rhino .3dm with at least one mesh object."""
    try:
        model = rh.File3dm.Read(file_path)
        # Ensure object table is not empty
        if len(model.Objects) == 0:
            raise ValueError("No objects found in .3dm file")
        # Ensure there is at least one mesh
        mesh_count = sum(1 for obj in model.Objects if isinstance(obj.Geometry, rh.Mesh))
        if mesh_count == 0:
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
