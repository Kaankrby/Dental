Ã¯Â»Â¿import sys
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
    _guidance("Global registration", "Use RANSAC when scans start far apart; disable for already aligned meshes to save time.")

    auto_voxels = st.checkbox(
        "Auto Voxel Sizes (recommended)",
        value=True,
        help="Automatically choose voxel sizes based on reference scale",
    )
    _guidance("Auto voxel sizes", "Derive voxel scales from reference spacing to keep metrics consistent across jaw sizes.")

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
            "Auto (plane-to-point fallback)",
            "Point-to-Plane",
            "Point-to-Point",
        ],
        help="Choose the ICP error metric. Point-to-plane is often better for scans with normals.",
    )
    icp_mode = {
        "Auto (plane-to-point fallback)": "auto",
        "Point-to-Plane": "point_to_plane",
        "Point-to-Point": "point_to_point",
    }[icp_mode_label]
    _guidance("ICP mode", "Point-to-plane converges faster with reliable normals; point-to-point is safer for noisy scans.")

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
    st.subheader("Layer Focus")
    layer_options = analyzer.get_reference_layers()
    default_focus = [
        ln for ln in layer_options if analyzer.layer_weights.get(ln, 1.0) > 0 and ln.lower() != "notimportant"
    ]
    if layer_options:
        existing = st.session_state.get(LAYER_FOCUS_STATE_KEY, default_focus or layer_options)
        sanitized = [ln for ln in existing if ln in layer_options] or (default_focus or layer_options)
        st.session_state[LAYER_FOCUS_STATE_KEY] = sanitized
        st.multiselect(
            "Highlight Layers",
            options=layer_options,
            default=sanitized,
            key=LAYER_FOCUS_STATE_KEY,
            help="Filter deviation stats and visuals to the selected reference layers.",
        )
    else:
        st.info("Upload a reference to focus on inner fissure layers.")
        st.session_state[LAYER_FOCUS_STATE_KEY] = []
    
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
            "Microns (um)",
        ],
        help="STL is unitless; choose how to interpret and convert to mm",
    )
    stl_unit_scale_map = {
        "Millimeters (mm)": 1.0,
        "Centimeters (cm)": 10.0,
        "Meters (m)": 1000.0,
        "Inches (in)": 25.4,
        "Feet (ft)": 304.8,
        "Microns (um)": 0.001,
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
    analyzer.layer_weights = st.session_state.layer_weights

selected_layers = st.session_state.get(LAYER_FOCUS_STATE_KEY, [])

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

def _o3d_mesh_to_rhino(stl_path: str, transformation: np.ndarray, stl_scale_to_mm: float) -> rh.Mesh:
    mesh = o3d.io.read_triangle_mesh(stl_path)
    if not mesh.has_triangles() or not mesh.has_vertices():
        raise ValueError("Test STL has no mesh data to export")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Test STL has no valid triangles")

    scale = float(stl_scale_to_mm or 1.0)
    if scale != 1.0:
        vertices = vertices * scale

    transform = np.asarray(transformation, dtype=np.float64) if transformation is not None else np.eye(4)
    if transform.shape != (4, 4):
        raise ValueError("Invalid transformation matrix for export")
    hom = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
    vertices = (hom @ transform.T)[:, :3]

    rh_mesh = rh.Mesh()
    for v in vertices:
        rh_mesh.Vertices.Add(float(v[0]), float(v[1]), float(v[2]))
    for tri in faces:
        rh_mesh.Faces.AddFace(int(tri[0]), int(tri[1]), int(tri[2]))
    rh_mesh.Normals.ComputeNormals()
    rh_mesh.Compact()
    return rh_mesh

def export_combined_3dm(reference_path: str, test_stl_path: str, transformation: np.ndarray, stl_scale_to_mm: float, test_name: str) -> bytes:
    reference_model = rh.File3dm.Read(reference_path)
    if reference_model is None:
        raise ValueError("Unable to read reference .3dm for export")

    combined = rh.File3dm()
    combined.Settings.ModelUnitSystem = reference_model.Settings.ModelUnitSystem

    ref_layer = rh.Layer()
    ref_layer.Name = "Reference"
    ref_layer_index = combined.Layers.Add(ref_layer)

    test_layer = rh.Layer()
    safe_name = "".join(ch if ch.isalnum() or ch in (" ", "-", "_") else "_" for ch in str(test_name or "Test"))
    test_layer.Name = f"Aligned - {safe_name}".strip()
    test_layer_index = combined.Layers.Add(test_layer)

    ref_mesh_count = 0
    for obj in reference_model.Objects:
        geom = obj.Geometry
        if isinstance(geom, rh.Mesh):
            attrs = rh.ObjectAttributes()
            attrs.LayerIndex = ref_layer_index
            combined.Objects.AddMesh(geom, attrs)
            ref_mesh_count += 1
    if ref_mesh_count == 0:
        raise ValueError("Reference file does not contain mesh objects to export")

    aligned_mesh = _o3d_mesh_to_rhino(test_stl_path, transformation, stl_scale_to_mm)
    attrs = rh.ObjectAttributes()
    attrs.LayerIndex = test_layer_index
    combined.Objects.AddMesh(aligned_mesh, attrs)

    with tempfile.NamedTemporaryFile(suffix=".3dm", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        if not combined.Write(tmp_path, 7):
            raise ValueError("Failed to serialize combined .3dm")
        with open(tmp_path, "rb") as f:
            data = f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return data

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
                        eval_points = np.asarray(result.get("eval_pcd", result["aligned_pcd"]).points)
                        dist = np.asarray(metrics["distances"])
                        wdist = np.asarray(metrics["weighted_distances"])
                        ref_dist = np.asarray(metrics.get("ref_distances", []))
                        ref_wdist = np.asarray(metrics.get("ref_weighted_distances", []))
                        ref_points = np.asarray(analyzer.reference_pcd.points) if analyzer.reference_pcd else np.empty((0, 3))
                        ref_layers = analyzer.reference_point_layers() if analyzer.reference_pcd else np.array([], dtype=object)
                        eval_layers = np.asarray(metrics.get("eval_layer_names", []))

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
                                if mode == "Reference -> Test" and analyzer.reference_pcd is not None:
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
                            stats = {
                                "rms": 0.0,
                                "rms_w": 0.0,
                                "p95": 0.0,
                                "p95_w": 0.0,
                                "within": 0.0,
                                "within_w": 0.0,
                            }
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

                        mode_key = f"deviation_mode_{i}"
                        default_mode = "Reference -> Test"
                        if mode_key not in st.session_state:
                            st.session_state[mode_key] = default_mode
                        active_mode = st.session_state.get(mode_key, default_mode)

                        sticky = st.container()
                        with sticky:
                            st.markdown('<div class="sticky-metrics">', unsafe_allow_html=True)
                            col_rms, col_p95, col_tol = st.columns(3)
                            col_rms.metric("RMS Ref->Test", f"{frames['Reference -> Test']['stats']['rms']:.3f} mm")
                            col_rms.metric("RMS Test->Ref", f"{frames['Test -> Reference']['stats']['rms']:.3f} mm")
                            col_p95.metric("P95 Ref->Test", f"{frames['Reference -> Test']['stats']['p95']:.3f} mm")
                            col_p95.metric("P95 Test->Ref", f"{frames['Test -> Reference']['stats']['p95']:.3f} mm")
                            col_tol.metric(f"Within {deviation_tolerance:.2f} mm (Ref->Test)", f"{frames['Reference -> Test']['stats']['within']:.1f}%")
                            col_tol.metric(f"Within {deviation_tolerance:.2f} mm (Test->Ref)", f"{frames['Test -> Reference']['stats']['within']:.1f}%")
                            st.markdown("</div>", unsafe_allow_html=True)

                        summary_tab, deviation_tab, volume_tab, export_tab = st.tabs(
                            ["Summary", "Deviation", "Volumes", "Exports"]
                        )

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
                                "Processing mode": processing_mode,
                                "Global registration": "On" if use_global_registration else "Off",
                                "ICP mode": icp_mode_label,
                                "ICP threshold used (mm)": f"{icp_threshold_used:.3f}",
                                "ICP iterations": icp_max_iter,
                                "Global voxel (mm)": f"{voxel_size_global_used:.2f}",
                                "Volume voxel (mm)": f"{volume_voxel_used:.2f}",
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
                                help="Switch between seeing deviations anchored on the reference surface or the aligned test scan.",
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
                                    from visualization import plot_point_cloud_by_values

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

                            overlay = plot_multiple_point_clouds(
                                [result["aligned_pcd"], analyzer.reference_pcd],
                                ["Aligned Test", "Reference"],
                            )
                            st.plotly_chart(overlay, use_container_width=True)

                            if auto_voxels or processing_mode == "Adaptive":
                                st.caption(
                                    (
                                        (f"Auto voxel sizes: global {voxel_size_global_used:.2f} mm, volume {volume_voxel_used:.2f} mm. " if auto_voxels else "")
                                        + (f"Adaptive ICP threshold: {icp_threshold_used:.2f} mm" if processing_mode == "Adaptive" else "")
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
                                f"Download Results (CSV) - {tf.name}",
                                export_df.to_csv(index=False).encode("utf-8"),
                                f"results_{i}.csv",
                                "text/csv",
                            )
                            try:
                                combined_bytes = export_combined_3dm(
                                    ref_path,
                                    test_path,
                                    metrics.get("transformation"),
                                    stl_scale_to_mm,
                                    tf.name,
                                )
                                st.download_button(
                                    f"Download Combined 3DM - {tf.name}",
                                    combined_bytes,
                                    f"comparison_{i}.3dm",
                                    "application/octet-stream",
                                    key=f"download_combined_{i}",
                                )
                            except Exception as export_err:
                                st.warning(f"3DM export unavailable: {export_err}")
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.exception(e)
