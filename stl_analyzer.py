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
    plot_point_cloud_heatmap,
    plot_multiple_point_clouds,
    plot_deviation_histogram,
    plot_registration_result,
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
    page_icon="ğŸ¦·",
)

# Initialize analyzer once per session
ctx = get_script_run_ctx()
if ctx and 'analyzer' not in st.session_state:
    st.session_state['analyzer'] = RhinoAnalyzer()

# ------------------
# -------------------------------
# Sidebar Controls
# -------------------------------------------------
with st.sidebar:
    st.header("Analysis Parameters")

    # Processing modes
    processing_mode = st.radio(
        "Processing Mode",
        ["Balanced", "Precision", "Speed", "Adaptive"],
        help="Predefined parameter sets. Adaptive auto-tunes ICP threshold from model scale.",
    )

    # Point Cloud Generation
    st.subheader("Point Cloud")
    num_points = st.slider(
        "Sample Points",
        1000,
        100000,
        value=(
            20000 if processing_mode == "Adaptive" else 15000 if processing_mode == "Balanced" else 30000 if processing_mode == "Precision" else 5000
        ),
        help="Number of points to sample from the reference .3dm",
    )

    # Registration Parameters
    st.subheader("Registration")
    use_global_registration = st.checkbox(
        "Enable Global Registration",
        value=processing_mode != "Speed",
        help="Use RANSAC-based global registration for initial alignment",
    )

    auto_voxels = st.checkbox(
        "Auto Voxel Sizes (recommended)",
        value=True,
        help="Automatically choose voxel sizes based on reference scale",
    )

    voxel_size_global = st.slider(
        "Global Voxel Size (mm)",
        0.1,
        5.0,
        value=1.5 if processing_mode == "Balanced" else 0.5 if processing_mode == "Precision" else 3.0,
        disabled=(not use_global_registration) or auto_voxels,
    )

    use_full_ref_global = st.checkbox(
        "Use full reference for global registration",
        value=False,
        help="RANSAC uses the entire reference point cloud instead of filtered important layers.",
    )

    st.subheader("ICP Parameters")
    icp_threshold = st.slider(
        "ICP Threshold (mm)",
        0.01,
        2.0,
        value=(0.25 if processing_mode == "Adaptive" else 0.3 if processing_mode == "Balanced" else 0.1 if processing_mode == "Precision" else 0.5),
        disabled=(processing_mode == "Adaptive"),
    )

    icp_max_iter = st.slider(
        "ICP Max Iterations",
        10,
        2000,
        value=200 if processing_mode == "Balanced" else 500 if processing_mode == "Precision" else 100,
    )

    icp_mode_label = st.selectbox(
        "ICP Mode",
        [
            "Auto (plane→point fallback)",
            "Point-to-Plane",
            "Point-to-Point",
        ],
        help="Choose the ICP error metric. Point-to-plane is often better for scans with normals.",
    )
    icp_mode = {
        "Auto (plane→point fallback)": "auto",
        "Point-to-Plane": "point_to_plane",
        "Point-to-Point": "point_to_point",
    }[icp_mode_label]

    # Visualization
    st.subheader("Visualization")
    point_size = st.slider("Point Size", 1, 10, 3)
    color_scale = st.selectbox("Color Scale", ["viridis", "plasma", "turbo", "hot"])
    deviation_tolerance = st.slider(
        "Deviation Tolerance (mm)",
        0.01,
        2.0,
        value=0.2 if processing_mode == "Balanced" else 0.1 if processing_mode == "Precision" else 0.3,
        help="Threshold used to compute coverage within tolerance",
    )
    
    # Units
    st.subheader("Units")
    stl_units_label = st.selectbox(
        "Test STL Units",
        [
            "Millimeters (mm)",
            "Centimeters (cm)",
            "Meters (m)",
            "Inches (in)",
            "Feet (ft)",
            "Microns (µm)",
        ],
        help="STL is unitless; choose how to interpret and convert to mm",
    )
    stl_unit_scale_map = {
        "Millimeters (mm)": 1.0,
        "Centimeters (cm)": 10.0,
        "Meters (m)": 1000.0,
        "Inches (in)": 25.4,
        "Feet (ft)": 304.8,
        "Microns (µm)": 0.001,
    }
    stl_scale_to_mm = float(stl_unit_scale_map.get(stl_units_label, 1.0))
    volume_voxel = st.slider(
        "Volume Voxel Size (mm)",
        0.05,
        2.0,
        value=0.5 if processing_mode == "Balanced" else 0.25 if processing_mode == "Precision" else 1.0,
        help="Voxel size for approximate volume overlap on open meshes",
        disabled=auto_voxels,
    )

    # Metrics inclusion toggle for NOTIMPORTANT
    include_notimportant_metrics = st.checkbox(
        "Include NOTIMPORTANT in metrics",
        value=False,
        help="Controls only metrics; alignment always focuses on important layers."
    )

    # Layer weights editor
    if 'layer_weights' not in st.session_state:
        st.session_state.layer_weights = {}

    st.subheader("Layer Weights")
    if st.session_state.layer_weights:
        weight_df = pd.DataFrame(
            list(st.session_state.layer_weights.items()),
            columns=["Layer", "Weight"],
        )
        edited_df = st.data_editor(weight_df, width='stretch', num_rows="dynamic")
        st.session_state.layer_weights = edited_df.set_index("Layer")["Weight"].to_dict()
    else:
        st.info("Upload a reference .3dm to populate layers.")

    # Keep analyzer in sync
    analyzer = st.session_state['analyzer']
    analyzer.layer_weights = st.session_state.layer_weights

# -------------------------------------------------
# Main Interface
# -------------------------------------------------
st.title("Dental STL Analyzer Pro")
st.markdown(
    """
    Compare dental scan STL files against a layered Rhino (.3dm) reference with weighted deviations and visualizations.
    """
)

# File Upload Sections
with st.expander("Upload Files", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Reference 3DM")
        ref_file = st.file_uploader(
            "Upload Reference .3dm",
            type=["3dm"],
            help="Rhino file with layered meshes",
        )

    with col2:
        st.subheader("Test STL(s)")
        test_files = st.file_uploader(
            "Upload Test STL(s)",
            type=["stl"],
            accept_multiple_files=True,
            help="One or more STL scans to analyze against reference",
        )

# After file upload but before processing
if ref_file:
    st.subheader("Reference File Preview")
    # Save and load temporary file
    ref_path_preview = save_uploaded_file(ref_file)
    model_preview = rh.File3dm.Read(ref_path_preview)
    try:
        units_name = rhino_unit_name(model_preview.Settings.ModelUnitSystem)
        st.info(f"Reference units: {units_name} (converted to mm for analysis)")
    except Exception:
        st.info("Reference units: Millimeters (assumed)")

    # Layer overview
    with st.expander("Layer Summary"):
        layers = {layer.Name: layer for layer in model_preview.Layers}
        layer_table = pd.DataFrame.from_dict(
            {
                "Layer": [layer.Name for layer in model_preview.Layers],
                "Object Count": [
                    sum(1 for obj in model_preview.Objects if obj.Attributes.LayerIndex == layer.Index)
                    for layer in model_preview.Layers
                ],
                "Weight": [st.session_state.layer_weights.get(layer.Name, 1.0) for layer in model_preview.Layers],
            }
        )
        st.dataframe(layer_table, width='stretch')

    # Initialize/augment layer weights from the uploaded reference so the sidebar updates
    try:
        current = st.session_state.get("layer_weights", {})
        for layer in model_preview.Layers:
            if layer.Name not in current:
                current[layer.Name] = default_layer_weight(layer.Name)
        st.session_state.layer_weights = current
        st.session_state['analyzer'].layer_weights = current
        # Trigger a rerun once on new upload so sidebar reflects weights immediately
        last_name = st.session_state.get("_last_ref_name")
        if last_name != getattr(ref_file, "name", None):
            st.session_state["_last_ref_name"] = getattr(ref_file, "name", None)
            st.experimental_rerun()
    except Exception:
        pass

    # 3D Preview
    with st.expander("3D Preview"):
        col1, col2 = st.columns([3, 1])
        with col1:
            plot = plot_rhino_model(model_preview)
            st.plotly_chart(plot, width='stretch')
        with col2:
            st.metric("Total Layers", len(layers))
            st.metric(
                "Total Meshes",
                sum(1 for obj in model_preview.Objects if isinstance(obj.Geometry, rh.Mesh)),
            )
            st.metric(
                "Total Vertices",
                sum(
                    len(obj.Geometry.Vertices)
                    for obj in model_preview.Objects
                    if isinstance(obj.Geometry, rh.Mesh)
                ),
            )

# -------------------------------------------------
# New Analysis (3DM reference + multi STL)
# -------------------------------------------------
def _populate_layer_weights_from_3dm(uploaded_ref_file):
    try:
        tmp = save_uploaded_file(uploaded_ref_file)
        model = rh.File3dm.Read(tmp)
        layer_names = [layer.Name for layer in model.Layers]
        if layer_names:
            current = st.session_state.get("layer_weights", {})
            for ln in layer_names:
                if ln not in current:
                    current[ln] = default_layer_weight(ln)
            st.session_state.layer_weights = current
            st.session_state['analyzer'].layer_weights = current
    except Exception:
        pass

if st.button("Start Analysis", type="primary", key="start_analysis_v2"):
    if not ref_file:
        st.error("Please upload a reference .3dm file!")
    elif not test_files:
        st.error("Please upload at least one test STL file!")
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

                # Process each test file
                for i, tf in enumerate(test_files, start=1):
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

                    with st.expander(f"Results: {tf.name}", expanded=True):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.subheader("Metrics Summary")
                            st.metric("Mean Deviation", f"{metrics['mean_deviation']:.3f} mm")
                            st.metric("Max Deviation", f"{metrics['max_deviation']:.3f} mm")
                            st.metric("Mean Weighted", f"{metrics['mean_weighted_deviation']:.3f} mm")
                            st.metric("Max Weighted", f"{metrics['max_weighted_deviation']:.3f} mm")
                            if 'volume_intersection_vox' in metrics:
                                st.metric("Overlap (Jaccard)", f"{metrics['volume_overlap_jaccard']*100:.1f}%")
                                st.metric("Intersect Volume", f"{metrics['volume_intersection_vox']:.3f} mm³")
                                st.metric("Overlap vs Ref", f"{metrics['coverage_ref_pct']:.1f}%")
                                st.metric("Overlap vs Test", f"{metrics['coverage_test_pct']:.1f}%")
                            else:
                                st.metric("Volume Similarity", f"{metrics['volume_similarity']*100:.1f}%")

                        with c2:
                            dist = np.asarray(metrics["distances"]) 
                            wdist = np.asarray(metrics["weighted_distances"]) 
                            fig1 = plot_deviation_histogram(dist, title="Raw Deviation Distribution")
                            fig2 = plot_deviation_histogram(wdist, title="Weighted Deviation Distribution")
                            st.plotly_chart(fig1, width='stretch')
                            st.plotly_chart(fig2, width='stretch')

                        # Deviation analysis
                        st.subheader("Deviation Analysis")
                        eval_points = np.asarray(result.get("eval_pcd", result["aligned_pcd"]).points)
                        # Stats
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

                        # 3D heatmaps of deviations
                        try:
                            from visualization import plot_point_cloud_by_values
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
                        except Exception as _e:
                            st.warning("Unable to render 3D deviation heatmaps.")

                        overlay = plot_multiple_point_clouds(
                            [result["aligned_pcd"], analyzer.reference_pcd],
                            ["Aligned Test", "Reference"],
                        )
                        st.plotly_chart(overlay, width='stretch')

                        # Report auto voxel sizes used and adaptive ICP
                        if auto_voxels or processing_mode == "Adaptive":
                            st.caption(
                                (
                                    (f"Auto voxel sizes: global {voxel_size_global_used:.2f} mm, volume {volume_voxel_used:.2f} mm. " if auto_voxels else "") +
                                    (f"Adaptive ICP threshold: {icp_threshold_used:.2f} mm" if processing_mode == "Adaptive" else "")
                                )
                            )

                        # Use the exact points used to compute metrics to avoid length mismatch
                        # (already computed above as eval_points)
                        export_df = pd.DataFrame(
                            np.column_stack((eval_points, dist, wdist)),
                            columns=["X", "Y", "Z", "Deviation", "WeightedDeviation"],
                        )
                        st.download_button(
                            f"Download Results (CSV) - {tf.name}",
                            export_df.to_csv(index=False).encode("utf-8"),
                            f"results_{i}.csv",
                            "text/csv",
                        )
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.exception(e)
