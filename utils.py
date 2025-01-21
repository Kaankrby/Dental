import re
import numpy as np
import trimesh
from typing import Tuple

def validate_file_name(filename: str) -> bool:
    """Validate file name for security."""
    return bool(re.match(r'^[\w\-. ]+$', filename))

def load_stl_file(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load STL file and return vertices and faces."""
    mesh = trimesh.load_mesh(file_path)
    return mesh.vertices, mesh.faces

def sample_points(vertices: np.ndarray, faces: np.ndarray, num_points: int) -> np.ndarray:
    """Sample points from mesh surface."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points

def compute_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute distances between source and target points."""
    target_tree = trimesh.proximity.ProximityQuery(trimesh.PointCloud(target))
    distances = target_tree.signed_distance(source)
    return distances

def compute_point_normals(
    points: np.ndarray,
    reference_points: np.ndarray,
    k: int = 20
) -> np.ndarray:
    """Compute approximate point normals using PCA on local neighborhoods."""
    tree = trimesh.proximity.ProximityQuery(trimesh.PointCloud(reference_points))
    _, indices = tree.kdtree.query(points, k=k)
    
    normals = np.zeros_like(points)
    for i, idx in enumerate(indices):
        neighbors = reference_points[idx]
        centered = neighbors - np.mean(neighbors, axis=0)
        u, _, _ = np.linalg.svd(centered)
        normals[i] = u[:, -1]
        
        if np.dot(normals[i], points[i] - np.mean(neighbors, axis=0)) < 0:
            normals[i] *= -1
            
    return normals

def compute_cavity_metrics(
    test_points: np.ndarray,
    reference_points: np.ndarray,
    ref_bbox: Tuple[np.ndarray, np.ndarray],
    tolerance: float = 0.5
) -> dict:
    """Compute cavity-specific metrics between test and reference points."""
    min_bound, max_bound = ref_bbox
    mask = np.all((test_points >= min_bound) & (test_points <= max_bound), axis=1)
    cavity_points = test_points[mask]
    
    distances = compute_distances(cavity_points, reference_points)
    
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
    
    return metrics

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
