import numpy as np
import trimesh
from typing import Tuple, Dict, Optional

class STLAnalyzer:
    def __init__(self):
        self.reference_points = None
        self.test_points = None
        self.metrics = None

    def load_reference(self, file_path: str, num_points: int = 10000) -> None:
        """Load and process reference STL file."""
        try:
            reference_mesh = trimesh.load_mesh(file_path)
            self.reference_points, _ = trimesh.sample.sample_surface(reference_mesh, num_points)
        except Exception as e:
            raise Exception(f"Error loading reference file: {str(e)}")

    def process_test_file(self, file_path: str, num_points: int = 10000, tolerance: float = 0.1) -> Dict:
        """Process a test STL file and compare with reference."""
        try:
            test_mesh = trimesh.load_mesh(file_path)
            self.test_points, _ = trimesh.sample.sample_surface(test_mesh, num_points)
            self.metrics = self._calculate_metrics(tolerance)
            return {
                'metrics': self.metrics,
                'test_points': self.test_points
            }
        except Exception as e:
            raise Exception(f"Error processing test file: {str(e)}")

    def _calculate_metrics(self, tolerance: float) -> Dict:
        """Calculate comparison metrics between reference and test points."""
        if self.reference_points is None:
            raise Exception("Reference model not loaded")
        if self.test_points is None:
            raise Exception("Test model not loaded")
        
        try:
            reference_tree = trimesh.proximity.ProximityQuery(trimesh.PointCloud(self.reference_points))
            distances = reference_tree.signed_distance(self.test_points)
            
            metrics = {
                'max_deviation': float(np.max(np.abs(distances))),
                'mean_deviation': float(np.mean(np.abs(distances))),
                'std_deviation': float(np.std(distances)),
                'points_in_tolerance': int(np.sum(np.abs(distances) <= tolerance)),
                'total_points': len(self.test_points),
                'tolerance_percentage': float(np.sum(np.abs(distances) <= tolerance) / len(self.test_points) * 100)
            }
            return metrics
        except Exception as e:
            raise Exception(f"Error calculating metrics: {str(e)}")
