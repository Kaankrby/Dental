import sys
if sys.version_info >= (3, 10):
    import collections.abc
    sys.modules['collections'].Mapping = collections.abc.Mapping

import os
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import open3d as o3d
import rhino3dm as rh

from processing import RhinoAnalyzer
from visualization import (
    plot_multiple_point_clouds,
    plot_deviation_distribution,
    plot_deviation_histogram,
    plot_point_cloud_by_values,
    plot_rhino_model,
)
from utils import (
    save_uploaded_file,
    validate_3dm_file,
    validate_stl_file,
    rhino_unit_name,
    estimate_point_spacing,
    default_layer_weight,
)
from streamlit.runtime.scriptrunner import get_script_run_ctx

# -------------------------------------------------
# Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Dental STL Analyzer Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="Ã„Å¸Ã…Â¸Ã‚Â¦Ã‚Â·",
)

# Initialize analyzer once per session
ctx = get_script_run_ctx()
if ctx and 'analyzer' not in st.session_state:
    st.session_state['analyzer'] = RhinoAnalyzer()
if 'analyzer' not in st.session_state:
    st.session_state['analyzer'] = RhinoAnalyzer()
analyzer = st.session_state['analyzer']

st.markdown(
    """
    <style>
    .sticky-metrics {
        position: sticky;
        top: 0;
        z-index: 50;
        background-color: var(--background-color, #0e1117);
        padding: 0.5rem 0.75rem;
        border-bottom: 1px solid rgba(250, 250, 250, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _guidance(title: str, message: str) -> None:
    """Utility to render inline help via popover or caption."""
    pop = getattr(st, "popover", None)
    if callable(pop):
        with pop(title):
            st.write(message)
    else:
        st.caption(f"{title}: {message}")


LAYER_FOCUS_STATE_KEY = "layer_focus_selection"
COMPARATOR_MODES = ("Legacy (Test anchored)", "Dual Reference/Test")
DEFAULT_COMPARATOR_MODE = COMPARATOR_MODES[1]
ANALYSIS_RESULTS_KEY = "analysis_payloads"

if ANALYSIS_RESULTS_KEY not in st.session_state:
    st.session_state[ANALYSIS_RESULTS_KEY] = []


def _point_cloud_from_array(points: np.ndarray) -> o3d.geometry.PointCloud:
    """Rebuild an Open3D point cloud from a numpy array."""
    arr = np.asarray(points)
    pcd = o3d.geometry.PointCloud()
    if arr.size:
        pcd.points = o3d.utility.Vector3dVector(arr)
    return pcd


def _render_legacy_entry(
    entry: dict,
    idx: int,
    analyzer: RhinoAnalyzer,
    deviation_tolerance: float,
    point_size: int,
    color_scale: str,
) -> None:
    metrics = entry["metrics"]
    eval_points = np.asarray(entry["eval_points"])
    dist = np.asarray(entry["dist"])
    wdist = np.asarray(entry["wdist"])
    aligned_pcd = _point_cloud_from_array(entry.get("aligned_points", np.empty((0, 3))))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Metrics Summary")
        st.metric("Mean Deviation", f"{metrics['mean_deviation']:.3f} mm")
        st.metric("Max Deviation", f"{metrics['max_deviation']:.3f} mm")
        st.metric("Mean Weighted", f"{metrics['mean_weighted_deviation']:.3f} mm")
        st.metric("Max Weighted", f"{metrics['max_weighted_deviation']:.3f} mm")
        if 'volume_intersection_vox' in metrics:
            st.metric("Overlap (Jaccard)", f"{metrics['volume_overlap_jaccard']*100:.1f}%")
            st.metric("Intersect Volume", f"{metrics['volume_intersection_vox']:.3f} mm^3")
            ref_gap = metrics.get("volume_ref_gap_vox")
            if ref_gap is not None:
                st.metric("Reference Difference Volume", f"{ref_gap:.3f} mm^3")
            st.metric("Overlap vs Ref", f"{metrics['coverage_ref_pct']:.1f}%")
            st.metric("Overlap vs Test", f"{metrics['coverage_test_pct']:.1f}%")
        else:
            st.metric("Volume Similarity", f"{metrics['volume_similarity']*100:.1f}%")

    with c2:
        fig1 = plot_deviation_histogram(dist, title="Raw Deviation Distribution")
        fig2 = plot_deviation_histogram(wdist, title="Weighted Deviation Distribution")
        st.plotly_chart(fig1, width='stretch')
        st.plotly_chart(fig2, width='stretch')

    st.subheader("Deviation Analysis")
    rms = float(np.sqrt(np.mean(dist**2))) if len(dist) else 0.0
    rms_w = float(np.sqrt(np.mean(wdist**2))) if len(wdist) else 0.0
    p95 = float(np.percentile(dist, 95)) if len(dist) else 0.0
    p95_w = float(np.percentile(wdist, 95)) if len(wdist) else 0.0
    within = float(np.mean(dist <= deviation_tolerance) * 100.0) if len(dist) else 0.0
    within_w = float(np.mean(wdist <= deviation_tolerance) * 100.0) if len(wdist) else 0.0

    c3, c4, c5 = st.columns(3)
    with c3:
        st.metric("RMS Deviation", f"{rms:.3f} mm")
        st.metric("RMS Weighted", f"{rms_w:.3f} mm")
    with c4:
        st.metric("P95 Deviation", f"{p95:.3f} mm")
        st.metric("P95 Weighted", f"{p95_w:.3f} mm")
    with c5:
        st.metric(f"Within {deviation_tolerance:.2f} mm", f"{within:.1f}%")
        st.metric(f"Within {deviation_tolerance:.2f} mm (Weighted)", f"{within_w:.1f}%")

    if len(eval_points) and (len(dist) or len(wdist)):
        try:
            heat1 = plot_point_cloud_by_values(
                eval_points, dist, title="3D Heatmap: Raw Deviations",
                point_size=point_size, color_scale=color_scale, colorbar_title="Deviation (mm)"
            )
            heat2 = plot_point_cloud_by_values(
                eval_points, wdist, title="3D Heatmap: Weighted Deviations",
                point_size=point_size, color_scale=color_scale, colorbar_title="Weighted (mm)"
            )
            st.plotly_chart(heat1, width='stretch')
            st.plotly_chart(heat2, width='stretch')
        except Exception:
            st.warning("Unable to render 3D deviation heatmaps.")

    if analyzer.reference_pcd is not None and len(aligned_pcd.points):
        overlay = plot_multiple_point_clouds(
            [aligned_pcd, analyzer.reference_pcd],
            ["Aligned Test", "Reference"],
        )
        st.plotly_chart(overlay, width='stretch')

    run_params = entry["run_params"]
    if run_params.get("auto_voxels") or run_params.get("processing_mode") == "Adaptive":
        st.caption(
            (
                (
                    f"Auto voxel sizes: global {run_params.get('voxel_size_global_used', 0):.2f} mm, "
                    f"volume {run_params.get('volume_voxel_used', 0):.2f} mm. "
                    if run_params.get("auto_voxels") else ""
                )
                + (
                    f"Adaptive ICP threshold: {run_params.get('icp_threshold_used', 0):.2f} mm"
                    if run_params.get("processing_mode") == "Adaptive" else ""
                )
            )
        )

    export_df = pd.DataFrame(
        np.column_stack((eval_points, dist, wdist)),
        columns=["X", "Y", "Z", "Deviation", "WeightedDeviation"],
    )
    st.download_button(
        f"Download Results (CSV) - {entry['name']}",
        export_df.to_csv(index=False).encode("utf-8"),
        f"results_{idx}.csv",
        "text/csv",
        key=f"csv_legacy_{idx}",
    )
    combined_bytes = entry.get("combined_bytes")
    if combined_bytes:
        st.download_button(
            f"Download Combined 3DM - {entry['name']}",
            combined_bytes,
            f"comparison_{idx}.3dm",
            "application/octet-stream",
            key=f"download_combined_{idx}",
        )
    elif entry.get("combined_error"):
        st.warning(f"3DM export unavailable: {entry['combined_error']}")


def _render_dual_entry(
    entry: dict,
    idx: int,
    analyzer: RhinoAnalyzer,
    selected_layers: list,
    deviation_tolerance: float,
    point_size: int,
    color_scale: str,
) -> None:
    metrics = entry["metrics"]
    eval_points = np.asarray(entry["eval_points"])
    dist = np.asarray(entry["dist"])
    wdist = np.asarray(entry["wdist"])
    ref_dist = np.asarray(entry.get("ref_dist", []))
    ref_wdist = np.asarray(entry.get("ref_wdist", []))
    eval_layers = np.asarray(entry.get("eval_layers", []))
    aligned_pcd = _point_cloud_from_array(entry.get("aligned_points", np.empty((0, 3))))
    reference_pcd = getattr(analyzer, "reference_pcd", None)
    ref_points = np.asarray(reference_pcd.points) if reference_pcd is not None else np.empty((0, 3))
    ref_layers = analyzer.reference_point_layers() if reference_pcd is not None else np.array([], dtype=object)

    base_frames = {
        "Reference -> Test": {
            "points": ref_points,
            "dist": ref_dist,
            "wdist": ref_wdist,
            "layers": ref_layers,
        },
        "Test -> Reference": {
            "points": eval_points,
            "dist": dist,
            "wdist": wdist,
            "layers": eval_layers,
        },
    }

    focus_lower = {str(ln).lower() for ln in selected_layers}

    def _filter_array(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        arr = np.asarray(values)
        if not len(arr) or not len(mask) or len(arr) != len(mask):
            return arr
        return arr[mask]

    def _apply_focus(mode: str, frame: dict) -> dict:
        points = frame["points"]
        layers = frame.get("layers", np.array([], dtype=object))
        mask = np.ones(len(points), dtype=bool) if len(points) else np.array([], dtype=bool)
        if selected_layers and len(points):
            if mode == "Reference -> Test" and reference_pcd is not None:
                mask = analyzer.reference_layer_mask(selected_layers)
            elif len(layers) == len(points):
                layer_lower = np.char.lower(layers.astype(str))
                mask = np.isin(layer_lower, list(focus_lower))
        filtered_layers = layers[mask] if len(layers) == len(mask) and len(mask) else layers
        layer_counts = {}
        if len(filtered_layers):
            uniques, counts = np.unique(filtered_layers, return_counts=True)
            layer_counts = dict(sorted(zip(uniques, counts), key=lambda x: -x[1]))
        return {
            "points": points[mask] if len(points) and len(mask) == len(points) else points,
            "dist": _filter_array(frame["dist"], mask),
            "wdist": _filter_array(frame["wdist"], mask),
            "layers": filtered_layers,
            "layer_counts": layer_counts,
        }

    def _stats(values: np.ndarray, weighted: np.ndarray) -> dict:
        stats = {"rms": 0.0, "rms_w": 0.0, "p95": 0.0, "p95_w": 0.0, "within": 0.0, "within_w": 0.0}
        vals = np.asarray(values)
        wvals = np.asarray(weighted)
        if len(vals):
            stats["rms"] = float(np.sqrt(np.mean(vals**2)))
            stats["p95"] = float(np.percentile(vals, 95))
            stats["within"] = float(np.mean(vals <= deviation_tolerance) * 100.0)
        if len(wvals):
            stats["rms_w"] = float(np.sqrt(np.mean(wvals**2)))
            stats["p95_w"] = float(np.percentile(wvals, 95))
            stats["within_w"] = float(np.mean(wvals <= deviation_tolerance) * 100.0)
        return stats

    frames = {}
    for mode, frame in base_frames.items():
        filtered = _apply_focus(mode, frame)
        filtered["stats"] = _stats(filtered["dist"], filtered["wdist"])
        frames[mode] = filtered

    mode_key = f"deviation_mode_{entry['name']}_{idx}"
    if mode_key not in st.session_state:
        st.session_state[mode_key] = "Reference -> Test"
    active_mode = st.session_state.get(mode_key, "Reference -> Test")

    st.markdown('<div class="sticky-metrics">', unsafe_allow_html=True)
    col_rms, col_p95, col_tol = st.columns(3)
    col_rms.metric("RMS Ref->Test", f"{frames['Reference -> Test']['stats']['rms']:.3f} mm")
    col_rms.metric("RMS Test->Ref", f"{frames['Test -> Reference']['stats']['rms']:.3f} mm")
    col_p95.metric("P95 Ref->Test", f"{frames['Reference -> Test']['stats']['p95']:.3f} mm")
    col_p95.metric("P95 Test->Ref", f"{frames['Test -> Reference']['stats']['p95']:.3f} mm")
    col_tol.metric(
        f"Within {deviation_tolerance:.2f} mm (Ref->Test)", f"{frames['Reference -> Test']['stats']['within']:.1f}%"
    )
    col_tol.metric(
        f"Within {deviation_tolerance:.2f} mm (Test->Ref)", f"{frames['Test -> Reference']['stats']['within']:.1f}%"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    summary_tab, deviation_tab, volume_tab, export_tab = st.tabs(["Summary", "Deviation", "Volumes", "Exports"])

    run_params = entry["run_params"]

    with summary_tab:
        st.subheader("Metrics Summary")
        sum_c1, sum_c2 = st.columns(2)
        with sum_c1:
            st.metric("Mean Deviation (Test->Ref)", f"{metrics['mean_deviation']:.3f} mm")
            st.metric("Max Deviation (Test->Ref)", f"{metrics['max_deviation']:.3f} mm")
            st.metric("Mean Weighted (Test->Ref)", f"{metrics['mean_weighted_deviation']:.3f} mm")
            st.metric("Max Weighted (Test->Ref)", f"{metrics['max_weighted_deviation']:.3f} mm")
        with sum_c2:
            st.metric("Mean Deviation (Ref->Test)", f"{metrics.get('mean_ref_deviation', 0.0):.3f} mm")
            st.metric("Max Deviation (Ref->Test)", f"{metrics.get('max_ref_deviation', 0.0):.3f} mm")
            st.metric("Mean Weighted (Ref->Test)", f"{metrics.get('mean_ref_weighted_deviation', 0.0):.3f} mm")
            st.metric("Max Weighted (Ref->Test)", f"{metrics.get('max_ref_weighted_deviation', 0.0):.3f} mm")

        snapshot = {
            "Processing mode": run_params.get("processing_mode"),
            "Global registration": "On" if run_params.get("use_global_registration") else "Off",
            "ICP mode": run_params.get("icp_mode_label"),
            "ICP threshold used (mm)": f"{run_params.get('icp_threshold_used', 0.0):.3f}",
            "ICP iterations": run_params.get("icp_max_iter"),
            "Global voxel (mm)": f"{run_params.get('voxel_size_global_used', 0.0):.2f}",
            "Volume voxel (mm)": f"{run_params.get('volume_voxel_used', 0.0):.2f}",
            "Fitness": f"{metrics.get('fitness', 0.0):.3f}",
            "Inlier RMSE": f"{metrics.get('inlier_rmse', 0.0):.3f}",
        }
        st.markdown("**Last Run Snapshot**")
        snapshot_df = pd.DataFrame(list(snapshot.items()), columns=["Parameter", "Value"])
        st.table(snapshot_df)

    with deviation_tab:
        st.subheader("Deviation Analysis")
        active_mode = st.radio(
            "Visualization frame",
            list(frames.keys()),
            key=mode_key,
            horizontal=True,
            help="Switch between deviations anchored on the reference surface or the aligned test scan.",
        )
        active_frame = frames[active_mode]
        if len(active_frame["dist"]):
            dist_fig = plot_deviation_distribution(
                active_frame["dist"],
                active_frame["wdist"],
                title=f"{active_mode} Distribution",
                layer_counts=active_frame["layer_counts"],
            )
            st.plotly_chart(dist_fig, use_container_width=True)
            try:
                heat_raw = plot_point_cloud_by_values(
                    active_frame["points"],
                    active_frame["dist"],
                    title=f"3D Heatmap: {active_mode} (raw)",
                    point_size=point_size,
                    color_scale=color_scale,
                    colorbar_title="Deviation (mm)",
                )
                st.plotly_chart(heat_raw, use_container_width=True)
                if len(active_frame["wdist"]):
                    heat_weighted = plot_point_cloud_by_values(
                        active_frame["points"],
                        active_frame["wdist"],
                        title=f"3D Heatmap: {active_mode} (weighted)",
                        point_size=point_size,
                        color_scale=color_scale,
                        colorbar_title="Weighted (mm)",
                    )
                    st.plotly_chart(heat_weighted, use_container_width=True)
            except Exception:
                st.warning("Unable to render 3D deviation heatmaps.")
        else:
            st.warning("No points available with the current layer filter for this frame.")

        if analyzer.reference_pcd is not None and len(aligned_pcd.points):
            overlay = plot_multiple_point_clouds(
                [aligned_pcd, analyzer.reference_pcd],
                ["Aligned Test", "Reference"],
            )
            st.plotly_chart(overlay, use_container_width=True)

        if run_params.get("auto_voxels") or run_params.get("processing_mode") == "Adaptive":
            st.caption(
                (
                    (
                        f"Auto voxel sizes: global {run_params.get('voxel_size_global_used', 0):.2f} mm, "
                        f"volume {run_params.get('volume_voxel_used', 0):.2f} mm. "
                        if run_params.get("auto_voxels") else ""
                    )
                    + (
                        f"Adaptive ICP threshold: {run_params.get('icp_threshold_used', 0):.2f} mm"
                        if run_params.get("processing_mode") == "Adaptive" else ""
                    )
                )
            )

    with volume_tab:
        st.subheader("Volume & Coverage")
        if 'volume_intersection_vox' in metrics:
            v1, v2, v3 = st.columns(3)
            with v1:
                st.metric("Overlap (Jaccard)", f"{metrics['volume_overlap_jaccard']*100:.1f}%")
                st.metric("Intersect Volume", f"{metrics['volume_intersection_vox']:.3f} mm^3")
            with v2:
                st.metric("Overlap vs Ref", f"{metrics['coverage_ref_pct']:.1f}%")
                st.metric("Overlap vs Test", f"{metrics['coverage_test_pct']:.1f}%")
            with v3:
                ref_gap = metrics.get("volume_ref_gap_vox")
                if ref_gap is not None:
                    st.metric("Reference Difference Volume", f"{ref_gap:.3f} mm^3")
                test_gap = metrics.get("volume_test_gap_vox")
                if test_gap is not None:
                    st.metric("Test Difference Volume", f"{test_gap:.3f} mm^3")
        else:
            st.metric("Volume Similarity", f"{metrics['volume_similarity']*100:.1f}%")

    with export_tab:
        st.subheader("Exports")
        export_df = pd.DataFrame(
            np.column_stack((eval_points, dist, wdist)),
            columns=["X", "Y", "Z", "Deviation_TestToRef", "WeightedDeviation_TestToRef"],
        )
        st.download_button(
            f"Download Results (CSV) - {entry['name']}",
            export_df.to_csv(index=False).encode("utf-8"),
            f"results_{idx}.csv",
            "text/csv",
            key=f"csv_dual_{idx}",
        )
        combined_bytes = entry.get("combined_bytes")
        if combined_bytes:
            st.download_button(
                f"Download Combined 3DM - {entry['name']}",
                combined_bytes,
                f"comparison_{idx}.3dm",
                "application/octet-stream",
                key=f"download_combined_dual_{idx}",
            )
        elif entry.get("combined_error"):
            st.warning(f"3DM export unavailable: {entry['combined_error']}")


def render_analysis_entries(
    entries: list,
    analyzer: RhinoAnalyzer,
    selected_layers: list,
    use_legacy_view: bool,
    deviation_tolerance: float,
    point_size: int,
    color_scale: str,
) -> None:
    """Render stored analysis entries so UI interactions don't reset results."""
    if not entries:
        return

    for idx, entry in enumerate(entries, start=1):
        with st.expander(f"Results: {entry['name']}", expanded=True):
            if use_legacy_view:
                _render_legacy_entry(entry, idx, analyzer, deviation_tolerance, point_size, color_scale)
            else:
                _render_dual_entry(entry, idx, analyzer, selected_layers, deviation_tolerance, point_size, color_scale)

# ------------------
# -------------------------------
# Sidebar Controls
    else:
        try:
            analyzer = st.session_state['analyzer']
            with tempfile.TemporaryDirectory() as temp_dir:
                # Persist reference and validate
                ref_path = os.path.join(temp_dir, "reference.3dm")
                with open(ref_path, "wb") as f:
                    f.write(ref_file.getbuffer())
                if not validate_3dm_file(ref_path):
                    st.stop()

                # Initialize weights if empty
                if not st.session_state.get("layer_weights"):
                    _populate_layer_weights_from_3dm(ref_file)

                # Load reference with requested sampling
                reference_pcd = analyzer.load_reference_3dm(
                    ref_path,
                    st.session_state.get("layer_weights", {}),
                    max_points=num_points,
                )
                # Auto-guess voxel sizes from reference spacing
                if auto_voxels:
                    try:
                        spacing = estimate_point_spacing(reference_pcd, sample_size=2000, k=2)
                    except Exception:
                        spacing = 0.5
                    # Heuristics tuned for dental scale (mm)
                    def _quantize(x, step=0.05):
                        return round(x / step) * step
                    def _clamp(x, lo, hi):
                        return max(lo, min(hi, x))
                    voxel_size_global_used = _clamp(_quantize(spacing * 3.0), 0.1, 3.0)
                    volume_voxel_used = _clamp(_quantize(spacing * 2.0), 0.1, 1.5)
                else:
                    voxel_size_global_used = voxel_size_global
                    volume_voxel_used = volume_voxel

                # Adaptive ICP threshold from spacing if selected
                if processing_mode == "Adaptive":
                    try:
                        spacing_for_icp = spacing if 'spacing' in locals() else estimate_point_spacing(reference_pcd, sample_size=2000, k=2)
                    except Exception:
                        spacing_for_icp = 0.1
                    def _quantize_icp(x, step=0.01):
                        return round(x / step) * step
                    def _clamp_icp(x, lo, hi):
                        return max(lo, min(hi, x))
                    # Threshold around few-neighbor spacing; slightly generous
                    icp_threshold_used = _clamp_icp(_quantize_icp(spacing_for_icp * 3.0), 0.05, 1.5)
                else:
                    icp_threshold_used = icp_threshold

                session_entries = []
                progress_bar = st.progress(0)
                total_files = len(test_files)
                
                for i, tf in enumerate(test_files, start=1):
                    try:
                        st.write(f"Processing: {tf.name}")
                        test_path = os.path.join(temp_dir, f"test_{i}.stl")
                        with open(test_path, "wb") as f:
                            f.write(tf.getbuffer())
                        if not validate_stl_file(test_path):
                            continue

                        result = analyzer.process_test_file(
                            test_path,
                            stl_scale_to_mm,
                            use_global_registration,
                            voxel_size_global_used,
                            icp_threshold_used,
                            icp_max_iter,
                            True,
                            include_notimportant_metrics,
                            use_full_ref_global,
                            icp_mode,
                            volume_voxel_used,
                        )
                        metrics = result["metrics"]
                        eval_pcd_obj = result.get("eval_pcd", result["aligned_pcd"])
                        eval_points = np.asarray(eval_pcd_obj.points)
                        dist = np.asarray(metrics["distances"])
                        wdist = np.asarray(metrics["weighted_distances"])
                        ref_dist = np.asarray(metrics.get("ref_distances", []))
                        ref_wdist = np.asarray(metrics.get("ref_weighted_distances", []))
                        eval_layers = np.asarray(metrics.get("eval_layer_names", []))
                        aligned_points = np.asarray(result["aligned_pcd"].points)

                        combined_bytes = None
                        combined_error = None
                        try:
                            combined_bytes = export_combined_3dm(
                                ref_path,
                                test_path,
                                metrics.get("transformation"),
                                stl_scale_to_mm,
                                tf.name,
                            )
                        except Exception as export_err:
                            combined_error = str(export_err)

                        session_entries.append(
                            {
                                "name": tf.name,
                                "metrics": metrics,
                                "eval_points": eval_points,
                                "dist": dist,
                                "wdist": wdist,
                                "ref_dist": ref_dist,
                                "ref_wdist": ref_wdist,
                                "eval_layers": eval_layers,
                                "aligned_points": aligned_points,
                                "combined_bytes": combined_bytes,
                                "combined_error": combined_error,
                                "run_params": {
                                    "processing_mode": processing_mode,
                                    "use_global_registration": use_global_registration,
                                    "icp_mode_label": icp_mode_label,
                                    "icp_threshold_used": icp_threshold_used,
                                    "icp_max_iter": icp_max_iter,
                                    "voxel_size_global_used": voxel_size_global_used,
                                    "volume_voxel_used": volume_voxel_used,
                                    "auto_voxels": auto_voxels,
                                },
                            }
                        )
                    except Exception as file_err:
                        st.error(f"Error processing {tf.name}: {str(file_err)}")
                        continue
                    finally:
                        progress_bar.progress(i / total_files)

                if session_entries:
                    st.session_state[ANALYSIS_RESULTS_KEY] = session_entries
                    st.success("Analysis complete. Results remain interactive below.")
                else:
                    st.warning("No valid test files were processed.")
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.exception(e)

stored_entries = st.session_state.get(ANALYSIS_RESULTS_KEY, [])
if stored_entries:
    render_analysis_entries(
        stored_entries,
        analyzer,
        selected_layers,
        use_legacy_view,
        deviation_tolerance,
        point_size,
        color_scale,
    )
