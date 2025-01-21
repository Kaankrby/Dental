import trimesh
import numpy as np
import time
import streamlit as st
import re
from typing import Tuple, List, Optional, Dict
from scipy.spatial import cKDTree

def validate_file_name(filename: str) -> bool:
    """Validate file name for security."""
    return bool(re.match(r'^[\w\-. ]+$', filename))

@st.cache_data
def load_mesh(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load an STL mesh file using trimesh and return vertices and faces."""
    try:
        mesh = trimesh.load_mesh(file_path)
        return mesh.vertices, mesh.faces
    except Exception as e:
        st.error(f"Error loading mesh: {str(e)}")
        raise

def performance_monitor(func):
    """Decorator to monitor function performance."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@st.cache_data
def sample_points_from_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_points: int
) -> np.ndarray:
    """Sample points from mesh using trimesh."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points

def remove_outliers(
    points: np.ndarray,
    nb_neighbors: int,
    std_ratio: float
) -> np.ndarray:
    """Remove outliers using statistical analysis."""
    # Build KD-tree for efficient nearest neighbor search
    tree = cKDTree(points)
    
    # Find distances to k-nearest neighbors
    distances, _ = tree.query(points, k=nb_neighbors)
    mean_distances = np.mean(distances, axis=1)
    
    # Calculate threshold for outlier removal
    threshold = np.mean(mean_distances) + std_ratio * np.std(mean_distances)
    
    # Filter points
    mask = mean_distances < threshold
    return points[mask]

def get_bounding_box(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get axis-aligned bounding box for points."""
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    return min_bound, max_bound

def points_in_box(
    points: np.ndarray,
    min_bound: np.ndarray,
    max_bound: np.ndarray
) -> np.ndarray:
    """Get boolean mask for points within bounding box."""
    mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)
    return mask

def process_mesh(
    file_path: str,
    num_points: int,
    nb_neighbors: int,
    std_ratio: float
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Process mesh file to points with outlier removal."""
    # Load and sample mesh
    vertices, faces = load_mesh(file_path)
    points = sample_points_from_mesh(vertices, faces, num_points)
    
    # Remove outliers
    clean_points = remove_outliers(points, nb_neighbors, std_ratio)
    
    # Get bounding box
    bbox = get_bounding_box(clean_points)
    
    return clean_points, bbox

def compute_cavity_metrics(
    test_points: np.ndarray,
    reference_points: np.ndarray,
    ref_bbox: Tuple[np.ndarray, np.ndarray],
    tolerance: float = 0.5
) -> dict:
    """Compute cavity-specific metrics between test and reference points."""
    # Get points within cavity region
    min_bound, max_bound = ref_bbox
    mask = points_in_box(test_points, min_bound, max_bound)
    cavity_points = test_points[mask]
    
    # Build KD-tree for efficient distance computation
    ref_tree = cKDTree(reference_points)
    
    # Compute point-to-point distances
    distances, _ = ref_tree.query(cavity_points)
    
    # Convert distances to signed distances (negative inside, positive outside)
    normals = compute_point_normals(cavity_points, reference_points)
    signed_distances = distances * np.sign(np.sum(normals * (cavity_points - reference_points), axis=1))
    
    # Compute metrics
    metrics = {
        'cavity_points': cavity_points,
        'distances': signed_distances,
        'points_in_cavity': len(cavity_points),
        'max_deviation': float(np.max(np.abs(signed_distances))),
        'mean_deviation': float(np.mean(signed_distances)),
        'std_deviation': float(np.std(signed_distances)),
        'underprepared_points': int(np.sum(signed_distances > tolerance)),
        'overprepared_points': int(np.sum(signed_distances < -tolerance))
    }
    
    return metrics

def compute_point_normals(
    points: np.ndarray,
    reference_points: np.ndarray,
    k: int = 20
) -> np.ndarray:
    """Compute approximate point normals using PCA on local neighborhoods."""
    tree = cKDTree(reference_points)
    _, indices = tree.query(points, k=k)
    
    normals = np.zeros_like(points)
    for i, idx in enumerate(indices):
        # Get local neighborhood
        neighbors = reference_points[idx]
        
        # Center points
        centered = neighbors - np.mean(neighbors, axis=0)
        
        # Compute normal using SVD
        u, _, _ = np.linalg.svd(centered)
        normals[i] = u[:, -1]
        
        # Orient normal towards point
        if np.dot(normals[i], points[i] - np.mean(neighbors, axis=0)) < 0:
            normals[i] *= -1
            
    return normals
