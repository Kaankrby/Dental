import open3d as o3d
import numpy as np
import streamlit as st
from typing import Tuple, Optional, Dict, Any
from utils import performance_monitor, load_mesh
import rhino3dm as rh
import copy

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
        # Store parameters
        self.voxel_size = voxel_size
        self.num_layers = layers
        
        # Load as single merged cloud
        raw_pcd = load_mesh(file_path)
        self.reference_full = raw_pcd
        
        # Direct sampling
        points = np.asarray(raw_pcd.points)
        if len(points) > num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
            sampled_points = points[indices]
        else:
            sampled_points = points
            
        # Create sampled point cloud
        self.reference_pcd = o3d.geometry.PointCloud()
        self.reference_pcd.points = o3d.utility.Vector3dVector(sampled_points)
        
        st.info(f"Loaded reference with {len(sampled_points)} points")
        return self.reference_pcd

    @performance_monitor
    def add_test_file(self, file_path: str, num_points: int, nb_neighbors: int, std_ratio: float):
        """Load and process a test STL file."""
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
        self.reference_pcd = None
        self.reference_full = None
        self.reference_bbox = None
        self.target_pcd = None
        self.voxel_size = None
        self.num_layers = None
        self.layer_weights = {}
        # Per-layer and per-point bookkeeping for 3DM
        self.reference_layers = {}  # layer_name -> np.ndarray[N,3]
        self._ref_point_layers = None  # np.ndarray[str] per point in reference_pcd
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
            # Collect vertices from meshes on this layer
            pts = []
            for obj in model.Objects:
                if obj.Attributes.LayerIndex == layer.Index and isinstance(obj.Geometry, rh.Mesh):
                    for v in obj.Geometry.Vertices:
                        pts.append([v.X, v.Y, v.Z])
            if not pts:
                continue
            pts = np.unique(np.asarray(pts, dtype=np.float64), axis=0)
            self.reference_layers[layer.Name] = pts

            # Append and tag
            all_points.append(pts)
            all_layer_names.extend([layer.Name] * len(pts))

            # Ensure a default weight exists
            if layer.Name not in self.layer_weights:
                self.layer_weights[layer.Name] = 1.0

        if not all_points:
            raise ValueError("No mesh vertices found in .3dm reference")

        all_points = np.vstack(all_points)
        all_layer_names = np.array(all_layer_names, dtype=object)

        # Downsample to max_points uniformly
        if len(all_points) > max_points:
            idx = np.random.choice(len(all_points), max_points, replace=False)
            sampled_points = all_points[idx]
            sampled_layers = all_layer_names[idx]
        else:
            sampled_points = all_points
            sampled_layers = all_layer_names

        # Scale reference points to millimeters according to model units
        try:
            from utils import rhino_unit_scale_to_mm
            scale_to_mm = rhino_unit_scale_to_mm(model.Settings.ModelUnitSystem)
        except Exception:
            scale_to_mm = 1.0
        if scale_to_mm != 1.0:
            sampled_points = sampled_points * float(scale_to_mm)

        self.reference_pcd = o3d.geometry.PointCloud()
        self.reference_pcd.points = o3d.utility.Vector3dVector(sampled_points)
        # Estimate normals for metric computations
        if len(sampled_points) > 3:
            self.reference_pcd.estimate_normals()
        self.reference_full = self.reference_pcd
        self.reference_bbox = self.reference_pcd.get_axis_aligned_bounding_box()

        # Build KDTree and per-point layer mapping
        self._ref_point_layers = sampled_layers
        self._ref_kdtree = o3d.geometry.KDTreeFlann(self.reference_pcd)

        st.info(f"Loaded 3DM reference with {len(sampled_points)} points across {len(self.reference_layers)} layers")
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

        st.info(f"Loaded target with {len(points)} points")
        return self.target_pcd

    def analyze_with_layers(self, target_pcd: o3d.geometry.PointCloud, transformation: np.ndarray):
        """Second stage: Detailed analysis using layers"""
        # Apply initial transformation
        target_transformed = copy.deepcopy(target_pcd)
        target_transformed.transform(transformation)
        
        results = {}
        # Now use self.reference_full and separate into layers for detailed analysis
        for layer in range(self.num_layers):
            # Layer-specific analysis here
            layer_results = self._analyze_layer(target_transformed, layer)
            results[f'layer_{layer}'] = layer_results
            
        return results
        
    def _analyze_layer(self, target_pcd: o3d.geometry.PointCloud, layer: int):
        """Analyze specific layer"""
        # Layer-specific analysis implementation
        # This will use self.reference_full and extract relevant points
        # based on layer information
        return {
            'mean_distance': 0.0,  # Placeholder
            'max_deviation': 0.0,  # Placeholder
            'coverage': 0.0        # Placeholder
        }

    def get_reference_layers(self):
        return list(self.reference_layers.keys()) if self.reference_layers else []

    def get_target_layers(self):
        # Targets are STL without layers; return empty list for now
        return []

    def _nearest_layer_weights(self, points: np.ndarray) -> np.ndarray:
        """For each point, find nearest reference point's layer and return weights."""
        if self._ref_kdtree is None or self._ref_point_layers is None:
            return np.ones(len(points))
        weights = np.ones(len(points), dtype=np.float64)
        for i, p in enumerate(points):
            _, idx, _ = self._ref_kdtree.search_knn_vector_3d(p, 1)
            layer_name = self._ref_point_layers[idx[0]]
            weights[i] = float(self.layer_weights.get(layer_name, 1.0))
        return weights

    def apply_layer_weights(self, distances: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Apply layer weights to per-point distances based on nearest reference layer."""
        if len(distances) != len(points):
            raise ValueError("Distances and points length mismatch")
        lw = self._nearest_layer_weights(points)
        return distances * lw

    def _nearest_layers(self, points: np.ndarray) -> np.ndarray:
        """Return nearest reference layer name for each point."""
        if self._ref_kdtree is None or self._ref_point_layers is None or len(points) == 0:
            return np.array([], dtype=object)
        layers = np.empty(len(points), dtype=object)
        for i, p in enumerate(points):
            _, idx, _ = self._ref_kdtree.search_knn_vector_3d(p, 1)
            layers[i] = self._ref_point_layers[idx[0]]
        return layers

    def _filter_reference_for_alignment(self) -> o3d.geometry.PointCloud:
        """Reference subset for alignment: exclude NOTIMPORTANT and zero-weight layers."""
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

    # Do not filter target for alignment prior to an initial transform; mapping is unreliable pre-align.

    @performance_monitor
    def add_test_file(self, file_path: str, num_points: int, nb_neighbors: int, std_ratio: float):
        """For API compatibility. No internal storage needed for multi-file pipeline."""
        # This method can remain a no-op since process step will load per-file
        return True

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

        # Load target
            test_pcd = self.load_target(file_path, estimate_normals=True, stl_scale_to_mm=stl_scale_to_mm)

        # Build filtered reference for alignment; keep full target for robustness
        ref_for_align = self._filter_reference_for_alignment()

        # Initial alignment
        transform_init = np.eye(4)
        if use_global_reg:
            # Reuse STLAnalyzer's pipeline for feature-based init
            # Downsample + FPFH
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

        # ICP refinement
        # Choose estimation based on icp_mode
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
        else:  # auto
            try:
                icp_result = _run_icp(o3d.pipelines.registration.TransformationEstimationPointToPlane())
            except Exception:
                icp_result = _run_icp(o3d.pipelines.registration.TransformationEstimationPointToPoint())

        # Transform test point cloud
        test_aligned = copy.deepcopy(self.target_pcd)
        test_aligned.transform(icp_result.transformation)

        # Filter by bounding box if requested
        if ignore_outside_bbox and self.reference_bbox is not None:
            indices = self.reference_bbox.get_point_indices_within_bounding_box(test_aligned.points)
            test_aligned = test_aligned.select_by_index(indices)

        # Compute advanced metrics (raw)
        from utils import compute_advanced_metrics
        # Metrics: include/exclude NOTIMPORTANT independently
        aligned_pts = np.asarray(test_aligned.points)
        nearest_layers = self._nearest_layers(aligned_pts) if len(aligned_pts) else np.array([], dtype=object)
        if len(nearest_layers) and not include_notimportant_metrics:
            eval_mask = (np.char.lower(nearest_layers.astype(str)) != 'notimportant') & (
                np.array([float(self.layer_weights.get(ln, 1.0)) for ln in nearest_layers]) > 0
            )
            eval_pcd = test_aligned.select_by_index(np.nonzero(eval_mask)[0])
        else:
            eval_pcd = test_aligned

        metrics = compute_advanced_metrics(eval_pcd, self.reference_pcd)
        metrics.update({
            'fitness': icp_result.fitness,
            'inlier_rmse': icp_result.inlier_rmse,
            'transformation': icp_result.transformation
        })

        # Compute layer-weighted deviations
        aligned_points = np.asarray(eval_pcd.points)
        raw_dists = metrics['distances']
        weighted_dists = self.apply_layer_weights(raw_dists, aligned_points)
        metrics['mean_weighted_deviation'] = float(np.mean(weighted_dists))
        metrics['max_weighted_deviation'] = float(np.max(weighted_dists))
        metrics['weighted_distances'] = weighted_dists

        # Voxel-based volumetric overlap (approximate, works for open meshes)
        try:
            from utils import compute_voxel_overlap_metrics
            vox = compute_voxel_overlap_metrics(eval_pcd, self.reference_pcd, volume_voxel_size)
            metrics.update({
                'volume_intersection_vox': vox.get('volume_intersection', 0.0),
                'volume_union_vox': vox.get('volume_union', 0.0),
                'volume_ref_vox': vox.get('volume_ref_approx', 0.0),
                'volume_test_vox': vox.get('volume_test_approx', 0.0),
                'volume_overlap_jaccard': vox.get('volume_overlap_jaccard', 0.0),
                'coverage_ref_pct': vox.get('coverage_ref_pct', 0.0),
                'coverage_test_pct': vox.get('coverage_test_pct', 0.0),
            })
            # Backwards-compatible override for similarity using Jaccard
            metrics['volume_similarity'] = float(metrics.get('volume_overlap_jaccard', 0.0))
        except Exception:
            pass

        return {
            'metrics': metrics,
            'aligned_pcd': test_aligned,
            'eval_pcd': eval_pcd,
        }

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
