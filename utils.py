import trimesh
import numpy as np
import time
import streamlit as st
import re
from typing import Tuple, List, Optional, Dict
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

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
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=nb_neighbors)
    mean_distances = np.mean(distances, axis=1)
    threshold = np.mean(mean_distances) + std_ratio * np.std(mean_distances)
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

def identify_cavity_regions(
    points: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 10
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Identify distinct regions in the cavity using DBSCAN clustering.
    Returns cluster labels and region properties.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    
    regions = []
    unique_labels = np.unique(labels[labels >= 0])  # Exclude noise points (-1)
    
    for label in unique_labels:
        mask = labels == label
        region_points = points[mask]
        
        center = np.mean(region_points, axis=0)
        min_bound = np.min(region_points, axis=0)
        max_bound = np.max(region_points, axis=0)
        
        regions.append({
            'label': int(label),
            'center': center,
            'bbox': (min_bound, max_bound),
            'num_points': int(np.sum(mask))
        })
    
    return labels, regions

def process_mesh(
    file_path: str,
    num_points: int,
    nb_neighbors: int,
    std_ratio: float
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], List[Dict]]:
    """Process mesh file to points with outlier removal and region identification."""
    vertices, faces = load_mesh(file_path)
    points = sample_points_from_mesh(vertices, faces, num_points)
    
    clean_points = remove_outliers(points, nb_neighbors, std_ratio)
    
    bbox = get_bounding_box(clean_points)
    
    _, regions = identify_cavity_regions(clean_points)
    
    return clean_points, bbox, regions

def compute_region_weights(regions: List[Dict]) -> np.ndarray:
    """Compute weights for each region based on size and location."""
    weights = []
    total_points = sum(region['num_points'] for region in regions)
    
    for region in regions:
        size_weight = region['num_points'] / total_points
        
        depth = region['center'][2]
        depth_weight = 1.0 + abs(depth) / 10.0  # Deeper regions get higher weight
        
        weights.append(size_weight * depth_weight)
    
    weights = np.array(weights)
    return weights / np.sum(weights)

def compute_cavity_metrics(
    test_points: np.ndarray,
    reference_points: np.ndarray,
    ref_bbox: Tuple[np.ndarray, np.ndarray],
    regions: List[Dict] = None,
    weights: np.ndarray = None,
    tolerance: float = 0.5
) -> dict:
    """Compute cavity-specific metrics between test and reference points."""
    min_bound, max_bound = ref_bbox
    mask = points_in_box(test_points, min_bound, max_bound)
    cavity_points = test_points[mask]
    
    ref_tree = cKDTree(reference_points)
    distances, _ = ref_tree.query(cavity_points)
    
    normals = compute_point_normals(cavity_points, reference_points)
    signed_distances = distances * np.sign(np.sum(normals * (cavity_points - reference_points), axis=1))
    
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
    
    if regions and weights is not None:
        region_metrics = []
        region_scores = []
        
        for i, region in enumerate(regions):
            region_mask = points_in_box(cavity_points, region['bbox'][0], region['bbox'][1])
            region_distances = signed_distances[region_mask]
            
            if len(region_distances) > 0:
                region_metric = {
                    'label': region['label'],
                    'num_points': len(region_distances),
                    'max_deviation': float(np.max(np.abs(region_distances))),
                    'mean_deviation': float(np.mean(region_distances)),
                    'underprepared': int(np.sum(region_distances > tolerance)),
                    'overprepared': int(np.sum(region_distances < -tolerance))
                }
                
                within_tolerance = np.sum(np.abs(region_distances) <= tolerance)
                region_score = (within_tolerance / len(region_distances)) * 100
                
                region_metrics.append(region_metric)
                region_scores.append(region_score * weights[i])
        
        metrics['region_metrics'] = region_metrics
        metrics['weighted_score'] = float(np.sum(region_scores))
    
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
        neighbors = reference_points[idx]
        centered = neighbors - np.mean(neighbors, axis=0)
        u, _, _ = np.linalg.svd(centered)
        normals[i] = u[:, -1]
        
        if np.dot(normals[i], points[i] - np.mean(neighbors, axis=0)) < 0:
            normals[i] *= -1
            
    return normals
