import open3d as o3d
import numpy as np
from typing import Tuple, Optional, Dict, Any
from backend.core.utils import performance_monitor, load_mesh, compute_advanced_metrics, default_layer_weight, rhino_unit_scale_to_mm, compute_voxel_overlap_metrics
import rhino3dm as rh
import copy
import logging

logger = logging.getLogger(__name__)

class STLAnalyzer:
    def __init__(self):
        self.reference_mesh = None
        self.reference_pcd = None
        self.reference_bbox = None
        self.test_meshes = {}
        self.results = {}

    @performance_monitor
    def load_reference(self, file_path: str, num_points: int, layers: int, voxel_size: float):
        """Load reference as merged point cloud"""
        self.voxel_size = voxel_size
        self.num_layers = layers
        
        raw_pcd = load_mesh(file_path)
        self.reference_full = raw_pcd
        
        points = np.asarray(raw_pcd.points)
        if len(points) > num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
            sampled_points = points[indices]
        else:
            sampled_points = points
            
        self.reference_pcd = o3d.geometry.PointCloud()
        self.reference_pcd.points = o3d.utility.Vector3dVector(sampled_points)
        
        logger.info(f"Loaded reference with {len(sampled_points)} points")
        return self.reference_pcd

    @performance_monitor
    def add_test_file(self, file_path: str, num_points: int, nb_neighbors: int, std_ratio: float):
        """Load and process a test STL file."""
        mesh = load_mesh(file_path)
        # Note: sample_point_cloud was not defined in original processing.py but called. 
        # Assuming it was meant to be implemented or imported. 
        # For now, we'll just use the mesh vertices as pcd.
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.points
        
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

        transform_init = np.eye(4)
        if use_global_reg:
            transform_init = self.global_registration(
                test_pcd,
                self.reference_pcd,
                voxel_size
            )

        icp_result = self.refine_registration(
            test_pcd,
            self.reference_pcd,
            transform_init,
            icp_threshold,
            max_iter
        )

        test_aligned = test_pcd.transform(icp_result.transformation)

        if ignore_outside_bbox:
            indices = self.reference_bbox.get_point_indices_within_bounding_box(
                test_aligned.points
            )
            test_aligned = test_aligned.select_by_index(indices)

        metrics = compute_advanced_metrics(test_aligned, self.reference_pcd)
        metrics.update({
            'fitness': icp_result.fitness,
            'inlier_rmse': icp_result.inlier_rmse,
            'transformation': icp_result.transformation.tolist() # Convert to list for JSON serialization
        })

        self.results[file_path] = {
            'metrics': metrics,
            # 'aligned_pcd': test_aligned # Cannot serialize PCD directly
        }

        return self.results[file_path]

class RhinoAnalyzer:
    def __init__(self):
        self.reference_pcd = None
        self.reference_full = None
        self.reference_bbox = None
        self.target_pcd = None
        self.voxel_size = None
        self.num_layers = None
        self.layer_weights = {}
        self.reference_layers = {}
        self._ref_point_layers = None
        self._ref_point_weights = None
        self._ref_kdtree = None

    @performance_monitor
    def load_reference_3dm(self, file_path: str, layer_weights: dict, max_points: int = 100000):
        """Load Rhino .3dm reference and build a combined point cloud with layer mapping."""
        self.layer_weights = dict(layer_weights or {})
        model = rh.File3dm.Read(file_path)

        layers = list(model.Layers)
        self.reference_layers = {}
        all_points = []
        all_layer_names = []

        for layer in layers:
            pts = []
            for obj in model.Objects:
                if obj.Attributes.LayerIndex == layer.Index and isinstance(obj.Geometry, rh.Mesh):
                    for v in obj.Geometry.Vertices:
                        pts.append([v.X, v.Y, v.Z])
            if not pts:
                continue
            pts = np.unique(np.asarray(pts, dtype=np.float64), axis=0)
            self.reference_layers[layer.Name] = pts

            all_points.append(pts)
            all_layer_names.extend([layer.Name] * len(pts))

            if layer.Name not in self.layer_weights:
                try:
                    self.layer_weights[layer.Name] = default_layer_weight(layer.Name)
                except Exception:
                    self.layer_weights[layer.Name] = 1.0

        if not all_points:
            raise ValueError("No mesh vertices found in .3dm reference")

        all_points = np.vstack(all_points)
        all_layer_names = np.array(all_layer_names, dtype=object)

        if len(all_points) > max_points:
            idx = np.random.choice(len(all_points), max_points, replace=False)
            sampled_points = all_points[idx]
            sampled_layers = all_layer_names[idx]
        else:
            sampled_points = all_points
            sampled_layers = all_layer_names

        try:
            scale_to_mm = rhino_unit_scale_to_mm(model.Settings.ModelUnitSystem)
        except Exception:
            scale_to_mm = 1.0
        if scale_to_mm != 1.0:
            sampled_points = sampled_points * float(scale_to_mm)

        self.reference_pcd = o3d.geometry.PointCloud()
        self.reference_pcd.points = o3d.utility.Vector3dVector(sampled_points)
        if len(sampled_points) > 3:
            self.reference_pcd.estimate_normals()
        self.reference_full = self.reference_pcd
        self.reference_bbox = self.reference_pcd.get_axis_aligned_bounding_box()

        self._ref_point_layers = sampled_layers
        self._ref_point_weights = np.array(
            [float(self.layer_weights.get(ln, 1.0)) for ln in sampled_layers], dtype=np.float64
        ) if len(sampled_layers) else np.array([], dtype=np.float64)
        self._ref_kdtree = o3d.geometry.KDTreeFlann(self.reference_pcd)

        logger.info(f"Loaded 3DM reference with {len(sampled_points)} points across {len(self.reference_layers)} layers")
        return self.reference_pcd

    @performance_monitor
    def load_target(self, file_path: str, estimate_normals: bool = True, stl_scale_to_mm: float = 1.0):
        """Load target STL/mesh into point cloud."""
        target_pcd = load_mesh(file_path)
        points = np.asarray(target_pcd.points, dtype=np.float64)
        if stl_scale_to_mm and stl_scale_to_mm != 1.0 and len(points):
            points = points * float(stl_scale_to_mm)

        self.target_pcd = o3d.geometry.PointCloud()
        self.target_pcd.points = o3d.utility.Vector3dVector(points)
        if estimate_normals and len(points) > 3:
            self.target_pcd.estimate_normals()

        logger.info(f"Loaded target with {len(points)} points")
        return self.target_pcd

    def _nearest_layer_weights(self, points: np.ndarray) -> np.ndarray:
        if self._ref_kdtree is None or self._ref_point_layers is None:
            return np.ones(len(points))
        weights = np.ones(len(points), dtype=np.float64)
        for i, p in enumerate(points):
            _, idx, _ = self._ref_kdtree.search_knn_vector_3d(p, 1)
            layer_name = self._ref_point_layers[idx[0]]
            weights[i] = float(self.layer_weights.get(layer_name, 1.0))
        return weights

    def apply_layer_weights(self, distances: np.ndarray, points: np.ndarray) -> np.ndarray:
        if len(distances) != len(points):
            raise ValueError("Distances and points length mismatch")
        lw = self._nearest_layer_weights(points)
        return distances * lw

    def _reference_point_weights(self) -> np.ndarray:
        if self._ref_point_weights is not None and len(self._ref_point_weights):
            return self._ref_point_weights
        if self.reference_pcd is None:
            return np.array([], dtype=np.float64)
        if self._ref_point_layers is None:
            self._ref_point_weights = np.ones(len(self.reference_pcd.points), dtype=np.float64)
        else:
            self._ref_point_weights = np.array(
                [float(self.layer_weights.get(ln, 1.0)) for ln in self._ref_point_layers], dtype=np.float64
            )
        return self._ref_point_weights

    def _nearest_layers(self, points: np.ndarray) -> np.ndarray:
        if self._ref_kdtree is None or self._ref_point_layers is None or len(points) == 0:
            return np.array([], dtype=object)
        layers = np.empty(len(points), dtype=object)
        for i, p in enumerate(points):
            _, idx, _ = self._ref_kdtree.search_knn_vector_3d(p, 1)
            layers[i] = self._ref_point_layers[idx[0]]
        return layers

    def _filter_reference_for_alignment(self) -> o3d.geometry.PointCloud:
        pts = np.asarray(self.reference_pcd.points)
        if self._ref_point_layers is None or len(pts) == 0:
            return self.reference_pcd
        mask = []
        for ln in self._ref_point_layers:
            w = float(self.layer_weights.get(ln, 1.0))
            mask.append((ln.lower() != 'notimportant') and (w > 0))
        mask = np.array(mask, dtype=bool)
        if not np.any(mask):
            return self.reference_pcd
        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(pts[mask])
        if len(out.points) > 3:
            out.estimate_normals()
        return out

    @performance_monitor
    def process_test_file(
            self,
            file_path: str,
            stl_scale_to_mm: float,
            use_global_reg: bool,
            voxel_size: float,
            icp_threshold: float,
            max_iter: int,
            ignore_outside_bbox: bool,
            include_notimportant_metrics: bool = False,
            use_full_ref_global: bool = False,
            icp_mode: str = 'auto',
            volume_voxel_size: float = 0.5
        ) -> Dict:
        """Process a single test file and compute metrics with layer-weighted deviations."""
        if self.reference_pcd is None:
            raise ValueError("Reference not loaded")

        self.load_target(file_path, estimate_normals=True, stl_scale_to_mm=stl_scale_to_mm)
        ref_for_align = self._filter_reference_for_alignment()

        transform_init = np.eye(4)
        if use_global_reg:
            source_down = self.target_pcd.voxel_down_sample(voxel_size=max(voxel_size, 1e-3))
            source_down.estimate_normals()
            ref_for_global = self.reference_pcd if use_full_ref_global else ref_for_align
            target_down = ref_for_global.voxel_down_sample(voxel_size=max(voxel_size, 1e-3))
            target_down.estimate_normals()
            source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                source_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=5.0*voxel_size, max_nn=100)
            )
            target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                target_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=5.0*voxel_size, max_nn=100)
            )
            result_init = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
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
            transform_init = result_init.transformation

        def _run_icp(estimation):
            return o3d.pipelines.registration.registration_icp(
                self.target_pcd,
                ref_for_align,
                icp_threshold,
                transform_init,
                estimation,
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
            )

        if icp_mode == 'point_to_point':
            icp_result = _run_icp(o3d.pipelines.registration.TransformationEstimationPointToPoint())
        elif icp_mode == 'point_to_plane':
            icp_result = _run_icp(o3d.pipelines.registration.TransformationEstimationPointToPlane())
        else:
            try:
                icp_result = _run_icp(o3d.pipelines.registration.TransformationEstimationPointToPlane())
            except Exception:
                icp_result = _run_icp(o3d.pipelines.registration.TransformationEstimationPointToPoint())

        test_aligned = copy.deepcopy(self.target_pcd)
        test_aligned.transform(icp_result.transformation)

        if ignore_outside_bbox and self.reference_bbox is not None:
            indices = self.reference_bbox.get_point_indices_within_bounding_box(test_aligned.points)
            test_aligned = test_aligned.select_by_index(indices)

        aligned_pts = np.asarray(test_aligned.points)
        nearest_layers = self._nearest_layers(aligned_pts) if len(aligned_pts) else np.array([], dtype=object)
        eval_layers = np.array([], dtype=object)
        if len(nearest_layers) and not include_notimportant_metrics:
            eval_mask = (np.char.lower(nearest_layers.astype(str)) != 'notimportant') & (
                np.array([float(self.layer_weights.get(ln, 1.0)) for ln in nearest_layers]) > 0
            )
            eval_idx = np.nonzero(eval_mask)[0]
            eval_pcd = test_aligned.select_by_index(eval_idx)
            eval_layers = nearest_layers[eval_idx]
        else:
            eval_pcd = test_aligned
            eval_layers = nearest_layers

        metrics = compute_advanced_metrics(eval_pcd, self.reference_pcd)
        metrics.update({
            'fitness': icp_result.fitness,
            'inlier_rmse': icp_result.inlier_rmse,
            'transformation': icp_result.transformation.tolist()
        })
        metrics['eval_layer_names'] = eval_layers.tolist() if len(eval_layers) else []

        aligned_points = np.asarray(eval_pcd.points)
        raw_dists = metrics['distances']
        weighted_dists = self.apply_layer_weights(raw_dists, aligned_points)
        metrics['mean_weighted_deviation'] = float(np.mean(weighted_dists))
        metrics['max_weighted_deviation'] = float(np.max(weighted_dists))
        metrics['weighted_distances'] = weighted_dists.tolist() # Serialize
        metrics['distances'] = raw_dists.tolist() # Serialize

        ref_metrics = compute_advanced_metrics(self.reference_pcd, eval_pcd)
        ref_distances = ref_metrics.get('distances', np.array([]))
        metrics['ref_distances'] = ref_distances.tolist()
        metrics['mean_ref_deviation'] = float(ref_metrics.get('mean_deviation', 0.0))
        metrics['max_ref_deviation'] = float(ref_metrics.get('max_deviation', 0.0))
        if len(ref_distances):
            ref_weights = self._reference_point_weights()
            ref_weighted = ref_distances * ref_weights
            metrics['ref_weighted_distances'] = ref_weighted.tolist()
            metrics['mean_ref_weighted_deviation'] = float(np.mean(ref_weighted))
            metrics['max_ref_weighted_deviation'] = float(np.max(ref_weighted))
        else:
            metrics['ref_weighted_distances'] = []
            metrics['mean_ref_weighted_deviation'] = 0.0
            metrics['max_ref_weighted_deviation'] = 0.0

        try:
            vox = compute_voxel_overlap_metrics(eval_pcd, self.reference_pcd, volume_voxel_size)
            metrics.update(vox)
            metrics['volume_similarity'] = float(metrics.get('volume_overlap_jaccard', 0.0))
        except Exception:
            pass

        return {
            'metrics': metrics,
            # 'aligned_pcd': test_aligned, # Cannot serialize
            # 'eval_pcd': eval_pcd, # Cannot serialize
        }
