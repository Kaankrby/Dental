import numpy as np
import streamlit as st
from typing import Tuple, Optional, Dict
from utils import performance_monitor, process_mesh, compute_cavity_metrics
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

class STLAnalyzer:
    def __init__(self):
        self.reference_points = None
        self.reference_bbox = None
        self.test_points = {}
        self.results = {}
        
    @performance_monitor
    def load_reference(self, file_path: str, num_points: int, nb_neighbors: int, std_ratio: float):
        """Load and process pre-cropped reference cavity model."""
        self.reference_points, self.reference_bbox = process_mesh(
            file_path,
            num_points,
            nb_neighbors,
            std_ratio
        )
        
    @performance_monitor
    def add_test_file(self, file_path: str, num_points: int, nb_neighbors: int, std_ratio: float):
        """Load and process a student's test cavity model."""
        points, _ = process_mesh(
            file_path,
            num_points,
            nb_neighbors,
            std_ratio
        )
        self.test_points[file_path] = points
        
    def icp_objective(self, params: np.ndarray, source: np.ndarray, target: np.ndarray) -> float:
        """Objective function for ICP optimization."""
        rotation = Rotation.from_euler('xyz', params[:3])
        translation = params[3:]
        
        # Apply transformation
        transformed = rotation.apply(source) + translation
        
        # Compute distances to nearest neighbors
        distances = np.min(np.linalg.norm(transformed[:, np.newaxis] - target, axis=2), axis=1)
        return np.mean(distances)
    
    @performance_monitor
    def align_point_clouds(
        self,
        source: np.ndarray,
        target: np.ndarray,
        max_iterations: int = 100
    ) -> Tuple[np.ndarray, float]:
        """Align source points to target points using ICP."""
        # Initial guess
        params = np.zeros(6)  # [rx, ry, rz, tx, ty, tz]
        
        # Optimize transformation
        result = minimize(
            self.icp_objective,
            params,
            args=(source, target),
            method='Nelder-Mead',
            options={'maxiter': max_iterations}
        )
        
        # Apply final transformation
        rotation = Rotation.from_euler('xyz', result.x[:3])
        translation = result.x[3:]
        aligned_points = rotation.apply(source) + translation
        
        return aligned_points, result.fun
        
    @performance_monitor
    def process_test_file(
        self,
        file_path: str,
        icp_max_iter: int = 100
    ) -> Dict:
        """Process a student's test file and compute cavity metrics."""
        if self.reference_points is None:
            raise ValueError("Reference cavity model not loaded")
            
        test_points = self.test_points[file_path]
        
        # Align points
        aligned_points, rmse = self.align_point_clouds(
            test_points,
            self.reference_points,
            max_iterations=icp_max_iter
        )
        
        # Compute metrics
        metrics = compute_cavity_metrics(
            aligned_points,
            self.reference_points,
            self.reference_bbox
        )
        
        # Add alignment quality metrics
        metrics.update({
            'alignment_rmse': float(rmse)
        })
        
        self.results[file_path] = {
            'metrics': metrics,
            'aligned_points': aligned_points
        }
        
        return self.results[file_path]
