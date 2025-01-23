import open3d as o3d
import numpy as np
import streamlit as st
from typing import Tuple, Optional, Dict, Any
from utils import performance_monitor

class STLAnalyzer:
    def __init__(self):
        self.reference_pcd = None
        self.test_pcds = {}
        
    @performance_monitor
    def load_reference(self, file_path: str, num_points: int, 
                      nb_neighbors: int, std_ratio: float):
        """Process reference STL to point cloud"""
        from utils import load_mesh, sample_point_cloud
        
        mesh = load_mesh(file_path)
        self.reference_pcd = sample_point_cloud(mesh, num_points, nb_neighbors, std_ratio)
        
    @performance_monitor
    def add_test_file(self, file_path: str, num_points: int,
                     nb_neighbors: int, std_ratio: float):
        """Process test STL to point cloud"""
        from utils import load_mesh, sample_point_cloud
        
        mesh = load_mesh(file_path)
        pcd = sample_point_cloud(mesh, num_points, nb_neighbors, std_ratio)
        self.test_pcds[file_path] = pcd
        
    @performance_monitor
    def prepare_for_registration(self, pcd: o3d.geometry.PointCloud, 
                                voxel_size: float) -> o3d.geometry.PointCloud:
        """Downsample and extract features"""
        pcd_down = pcd.voxel_down_sample(voxel_size)
        pcd_down.estimate_normals()
        return pcd_down
        
    @performance_monitor
    def global_registration(self, source: o3d.geometry.PointCloud, 
                           target: o3d.geometry.PointCloud, voxel_size: float) -> np.ndarray:
        """RANSAC-based initial alignment"""
        source_down = self.prepare_for_registration(source, voxel_size)
        target_down = self.prepare_for_registration(target, voxel_size)
        
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down,
            o3d.pipelines.registration.Feature(),
            mutual_filter=True,
            max_correspondence_distance=voxel_size * 1.5,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            ransac_n=4
        )
        return result.transformation
        
    @performance_monitor
    def refine_registration(self, source: o3d.geometry.PointCloud, 
                           target: o3d.geometry.PointCloud, 
                           init_transform: np.ndarray, threshold: float) -> Dict:
        """ICP refinement"""
        result = o3d.pipelines.registration.registration_icp(
            source, target,
            threshold, init_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        return {
            'transformation': result.transformation,
            'fitness': result.fitness,
            'rmse': result.inlier_rmse
        }
        
    @performance_monitor
    def process_test_file(self, file_path: str, voxel_size: float, 
                         icp_threshold: float) -> Dict:
        """Full processing pipeline"""
        test_pcd = self.test_pcds[file_path]
        
        # Initial alignment
        init_transform = self.global_registration(test_pcd, self.reference_pcd, voxel_size)
        
        # ICP refinement
        icp_result = self.refine_registration(test_pcd, self.reference_pcd, 
                                             init_transform, icp_threshold)
        
        # Transform and compute metrics
        aligned_pcd = test_pcd.transform(icp_result['transformation'])
        metrics = compute_advanced_metrics(aligned_pcd, self.reference_pcd)
        metrics.update(icp_result)
        
        return {
            'metrics': metrics,
            'aligned_pcd': aligned_pcd
        }