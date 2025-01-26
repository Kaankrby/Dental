import open3d as o3d
import numpy as np
import streamlit as st
from typing import Tuple, Optional, Dict, Any
from utils import performance_monitor
import rhino3dm as rh

class STLAnalyzer:
    def __init__(self):
        self.reference_mesh = None
        self.reference_pcd = None
        self.reference_bbox = None
        self.test_meshes = {}
        self.results = {}

    @performance_monitor
    def load_reference(self, file_path: str, num_points: int, nb_neighbors: int, std_ratio: float):
        """Load and process reference STL file."""
        from utils import load_mesh, sample_point_cloud
        
        self.reference_mesh = load_mesh(file_path)
        self.reference_pcd = sample_point_cloud(
            self.reference_mesh,
            num_points,
            nb_neighbors,
            std_ratio
        )
        self.reference_bbox = self.reference_pcd.get_axis_aligned_bounding_box()

    @performance_monitor
    def add_test_file(self, file_path: str, num_points: int, nb_neighbors: int, std_ratio: float):
        """Load and process a test STL file."""
        from utils import load_mesh, sample_point_cloud

        mesh = load_mesh(file_path)
        pcd = sample_point_cloud(mesh, num_points, nb_neighbors, std_ratio)
        self.test_meshes[file_path] = {
            'mesh': mesh,
            'pcd': pcd
        }

    @performance_monitor
    def prepare_for_global_registration(
        self,
        pcd: o3d.geometry.PointCloud,
        voxel_size: float
    ) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
        """Prepare point cloud for global registration."""
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=2.0*voxel_size,
                max_nn=30
            )
        )
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=5.0*voxel_size,
                max_nn=100
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
        """Perform global registration using RANSAC."""
        source_down, source_fpfh = self.prepare_for_global_registration(source, voxel_size)
        target_down, target_fpfh = self.prepare_for_global_registration(target, voxel_size)

        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down,
            source_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=voxel_size * 1.5,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
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
        """Refine registration using ICP."""
        result = o3d.pipelines.registration.registration_icp(
            source, target,
            threshold, init_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
        )
        return result

    @performance_monitor
    def process_test_file(
        self,
        file_path: str,
        use_global_reg: bool,
        voxel_size: float,
        icp_threshold: float,
        max_iter: int,
        ignore_outside_bbox: bool
    ) -> Dict:
        """Process a single test file and compute metrics."""
        if self.reference_pcd is None:
            raise ValueError("Reference not loaded")

        test_data = self.test_meshes[file_path]
        test_pcd = test_data['pcd']

        # Initial alignment
        transform_init = np.eye(4)
        if use_global_reg:
            transform_init = self.global_registration(
                test_pcd,
                self.reference_pcd,
                voxel_size
            )

        # ICP refinement
        icp_result = self.refine_registration(
            test_pcd,
            self.reference_pcd,
            transform_init,
            icp_threshold,
            max_iter
        )

        # Transform test point cloud
        test_aligned = test_pcd.transform(icp_result.transformation)

        # Filter points if needed
        if ignore_outside_bbox:
            indices = self.reference_bbox.get_point_indices_within_bounding_box(
                test_aligned.points
            )
            test_aligned = test_aligned.select_by_index(indices)

        # Compute metrics
        from utils import compute_advanced_metrics
        metrics = compute_advanced_metrics(test_aligned, self.reference_pcd)
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

class RhinoAnalyzer:
    def __init__(self):
        self.layer_weights = {}  # Will be set from UI
        self.reference = None
        
    def set_weights(self, weights: dict):
        """Update weights from UI"""
        self.layer_weights = weights
        
    def load_reference_3dm(self, file_path: str):
        """Load Rhino .3dm file with layered meshes"""
        model = rh.File3dm.Read(file_path)
        weighted_points = []
        
        for obj in model.Objects:
            if isinstance(obj.Geometry, rh.Mesh):
                mesh = obj.Geometry
                layer = model.Layers.FindIndex(obj.Attributes.LayerIndex)
                weight = self.layer_weights.get(layer.Name, 1.0)
                
                # Directly use vertices
                vertices = np.array([[v.X, v.Y, v.Z] for v in mesh.Vertices])
                weights = np.full((len(vertices), 1), weight)
                weighted_points.append(np.hstack((vertices, weights)))
        
        # Combine all layers
        all_points = np.vstack(weighted_points)
        self.reference = o3d.geometry.PointCloud()
        self.reference.points = o3d.utility.Vector3dVector(all_points[:, :3])
        self.reference.colors = o3d.utility.Vector3dVector(
            np.tile(all_points[:, 3:], (1, 3))  # Weight as RGB
        )
        self.kdtree = o3d.geometry.KDTreeFlann(self.reference)

    def calculate_weighted_deviation(self, test_points: np.ndarray) -> dict:
        """Calculate weighted deviations against reference"""
        deviations = []
        weighted_deviations = []
        
        for point in test_points:
            # Find nearest 3 reference points
            _, idxs, dists = self.kdtree.search_knn_vector_3d(point, 3)
            
            # Calculate weighted distance
            weights = np.array([self.reference.colors[id][0] for id in idxs])
            weighted_dists = np.array(dists) * weights
            weighted_dev = np.mean(weighted_dists)
            
            deviations.append(np.mean(dists))
            weighted_deviations.append(weighted_dev)
            
        return {
            'raw_deviations': np.array(deviations),
            'weighted_deviations': np.array(weighted_deviations),
            'mean_raw': np.mean(deviations),
            'mean_weighted': np.mean(weighted_deviations),
            'max_raw': np.max(deviations),
            'max_weighted': np.max(weighted_deviations)
        }