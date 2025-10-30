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
from utils import save_uploaded_file, validate_3dm_file, validate_stl_file
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

# -------------------------------------------------
# Sidebar Controls
# -------------------------------------------------
with st.sidebar:
    st.header("Analysis Parameters")

    # Processing modes
    processing_mode = st.radio(
        "Processing Mode",
        ["Balanced", "Precision", "Speed"],
        help="Predefined parameter sets for different use cases",
    )

    # Point Cloud Generation
    st.subheader("Point Cloud")
    num_points = st.slider(
        "Sample Points",
        1000,
        100000,
        value=15000 if processing_mode == "Balanced" else 30000 if processing_mode == "Precision" else 5000,
        help="Number of points to sample from the reference .3dm",
    )

    # Registration Parameters
    st.subheader("Registration")
    use_global_registration = st.checkbox(
        "Enable Global Registration",
        value=processing_mode != "Speed",
        help="Use RANSAC-based global registration for initial alignment",
    )

    voxel_size_global = st.slider(
        "Global Voxel Size (mm)",
        0.1,
        5.0,
        value=1.5 if processing_mode == "Balanced" else 0.5 if processing_mode == "Precision" else 3.0,
        disabled=not use_global_registration,
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
        value=0.3 if processing_mode == "Balanced" else 0.1 if processing_mode == "Precision" else 0.5,
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
        edited_df = st.data_editor(weight_df, use_container_width=True, num_rows="dynamic")
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
        st.dataframe(layer_table, use_container_width=True)

    # 3D Preview
    with st.expander("3D Preview"):
        col1, col2 = st.columns([3, 1])
        with col1:
            plot = plot_rhino_model(model_preview)
            st.plotly_chart(plot, use_container_width=True)
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
                    current[ln] = 1.0
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
                analyzer.load_reference_3dm(
                    ref_path,
                    st.session_state.get("layer_weights", {}),
                    max_points=num_points,
                )

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
                        use_global_registration,
                        voxel_size_global,
                        icp_threshold,
                        icp_max_iter,
                        True,
                        include_notimportant_metrics,
                        use_full_ref_global,
                        icp_mode,
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
                            st.metric("Volume Similarity", f"{metrics['volume_similarity']*100:.1f}%")

                        with c2:
                            dist = np.asarray(metrics["distances"]) 
                            wdist = np.asarray(metrics["weighted_distances"]) 
                            fig1 = plot_deviation_histogram(dist, title="Raw Deviation Distribution")
                            fig2 = plot_deviation_histogram(wdist, title="Weighted Deviation Distribution")
                            st.plotly_chart(fig1, use_container_width=True)
                            st.plotly_chart(fig2, use_container_width=True)

                        overlay = plot_multiple_point_clouds(
                            [result["aligned_pcd"], analyzer.reference_pcd],
                            ["Aligned Test", "Reference"],
                        )
                        st.plotly_chart(overlay, use_container_width=True)

                        # Use the exact points used to compute metrics to avoid length mismatch
                        eval_points = np.asarray(result.get("eval_pcd", result["aligned_pcd"]).points)
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
