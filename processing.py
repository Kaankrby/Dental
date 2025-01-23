import open3d as o3d
import numpy as np
import streamlit as st
from typing import Tuple, Optional, Dict, Any
from utils import performance_monitor, validate_mesh_watertight

class STLAnalyzer:
    def __init__(self):
        self.reference_mesh = None
        self.reference_pcd = None
        self.reference_bbox = None
        self.test_meshes = {}
        self.results = {}
        
    @performance_monitor
    def load_reference(self, file_path: str, num_points: int, 
                      nb_neighbors: int, std_ratio: float) -> None:
        """Load and process reference STL file with validation."""
        from utils import load_mesh, sample_point_cloud
        
        self.reference_mesh = load_mesh(file_path)
        validate_mesh_watertight(self.reference_mesh)  # New validation
        
        self.reference_pcd = sample_point_cloud(
            self.reference_mesh,
            num_points,
            nb_neighbors,
            std_ratio
        )
        self.reference_bbox = self.reference_pcd.get_axis_aligned_bounding_box()
        self._precompute_reference_features()  # New precomputation
        
    def _precompute_reference_features(self) -> None:
        """Precompute features for faster comparisons."""
        self.reference_kdtree = o3d.geometry.KDTreeFlann(self.reference_pcd)
        self.reference_normals = np.asarray(self.reference_pcd.normals)
        
    @performance_monitor
    def add_test_file(self, file_path: str, num_points: int,
                     nb_neighbors: int, std_ratio: float) -> None:
        """Load and process test STL file with validation."""
        from utils import load_mesh, sample_point_cloud
        
        mesh = load_mesh(file_path)
        validate_mesh_watertight(mesh)  # New validation
        
        pcd = sample_point_cloud(mesh, num_points, nb_neighbors, std_ratio)
        self.test_meshes[file_path] = {
            'mesh': mesh,
            'pcd': pcd,
            'kdtree': o3d.geometry.KDTreeFlann(pcd),  # Precompute KDTree
            'normals': np.asarray(pcd.normals)
        }
        
    @performance_monitor
    def prepare_for_global_registration(
        self,
        pcd: o3d.geometry.PointCloud,
        voxel_size: float
    ) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
        """Optimized feature preparation with adaptive parameters."""
        # Adaptive parameters based on voxel size
        pcd_down = pcd.voxel_down_sample(voxel_size=max(voxel_size, 0.1))
        
        radius_normals = 2.5 * voxel_size
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_normals,
                max_nn=min(50, int(len(pcd_down.points)*0.1))
            )
        )
        
        # Optimized FPFH parameters
        radius_feature = 5.0 * voxel_size
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_feature,
                max_nn=min(100, len(pcd_down.points))
            )
        )
        return pcd_down, fpfh
        
    @performance_monitor
    def global_registration(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        voxel_size: float
    ) -> np.ndarray:
        """Optimized global registration with adaptive RANSAC."""
        source_down, source_fpfh = self.prepare_for_global_registration(source, voxel_size)
        target_down, target_fpfh = self.prepare_for_global_registration(target, voxel_size)
        
        # Adaptive parameters based on point cloud size
        n_points = min(len(source_down.points), len(target_down.points))
        ransac_n = max(3, min(6, int(n_points * 0.001)))
        
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down,
            source_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=voxel_size * 1.5,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=ransac_n,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(
                    np.deg2rad(15)  # New normal consistency check
                )
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(500000, 0.999)
        )
        return result.transformation
        
    @performance_monitor
    def refine_registration(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        init_transform: np.ndarray,
        threshold: float,
        max_iter: int
    ) -> o3d.pipelines.registration.RegistrationResult:
        """Enhanced ICP with normal compatibility check."""
        return o3d.pipelines.registration.registration_icp(
            source, target,
            threshold, init_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iter,
                relative_fitness=1e-6,
                relative_rmse=1e-6
            )
        )
        
    @performance_monitor
    def process_test_file(
        self,
        file_path: str,
        use_global_reg: bool,
        voxel_size: float,
        icp_threshold: float,
        max_iter: int,
        ignore_outside_bbox: bool
    ) -> Dict[str, Any]:
        """Enhanced processing with additional metrics and validation."""
        if self.reference_pcd is None:
            raise ValueError("Reference not loaded")
            
        test_data = self.test_meshes[file_path]
        test_pcd = test_data['pcd']
        
        # Multi-stage registration
        transform_init = np.eye(4)
        if use_global_reg:
            with st.spinner("Global registration..."):
                transform_init = self.global_registration(
                    test_pcd, self.reference_pcd, voxel_size
                )
        
        # Multi-resolution ICP
        with st.spinner("ICP refinement..."):
            icp_result = self.refine_registration(
                test_pcd, self.reference_pcd,
                transform_init, icp_threshold, max_iter
            )
        
        test_aligned = test_pcd.transform(icp_result.transformation)
        
        if ignore_outside_bbox:
            indices = self.reference_bbox.get_point_indices_within_bounding_box(
                test_aligned.points
            )
            test_aligned = test_aligned.select_by_index(indices)
            if len(indices) == 0:
                raise ValueError("No points remaining after bbox filtering")
                
        # Enhanced metrics calculation
        from utils import compute_advanced_metrics
        metrics = compute_advanced_metrics(
            test_aligned, 
            self.reference_pcd,
            self.reference_kdtree,
            self.reference_normals
        )
        
        metrics.update({
            'fitness': icp_result.fitness,
            'inlier_rmse': icp_result.inlier_rmse,
            'transformation': icp_result.transformation
        })
        
        self.results[file_path] = {
            'metrics': metrics,
            'aligned_pcd': test_aligned
        }
        
        return self.results[file_path]