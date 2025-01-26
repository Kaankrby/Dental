import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os
from processing import STLAnalyzer, RhinoAnalyzer
from visualization import (
    plot_point_cloud_heatmap, 
    plot_multiple_point_clouds,
    plot_deviation_histogram,
    plot_normal_angle_distribution
)
from utils import validate_file_name
from streamlit.runtime.scriptrunner import get_script_run_ctx
import open3d as o3d

# -------------------------------------------------
# Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="🦷 Dental STL Analyzer Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🦷"
)

# Performance optimization
ctx = get_script_run_ctx()
if ctx and not hasattr(ctx, "_is_replicated"):
    if 'analyzer' not in st.session_state:
        st.session_state['analyzer'] = STLAnalyzer()

# -------------------------------------------------
# Sidebar Controls
# -------------------------------------------------
with st.sidebar:
    st.header("⚙️ Analysis Parameters")
    
    # Processing modes
    processing_mode = st.radio(
        "Processing Mode",
        ["Balanced", "Precision", "Speed"],
        help="Predefined parameter sets for different use cases"
    )
    
    # Point Cloud Generation
    st.subheader("🔵 Point Cloud")
    num_points = st.slider(
        "Sample Points", 
        1000, 100000, 
        value=15000 if processing_mode == "Balanced" else 30000 if processing_mode == "Precision" else 5000,
        help="Number of points to sample from the mesh"
    )
    
    # Registration Parameters
    st.subheader("🔧 Registration")
    use_global_registration = st.checkbox(
        "Enable Global Registration", 
        value=processing_mode != "Speed",
        help="Use RANSAC-based global registration for initial alignment"
    )
    
    voxel_size_global = st.slider(
        "Global Voxel Size (mm)", 
        0.1, 5.0, 
        value=1.5 if processing_mode == "Balanced" else 0.5 if processing_mode == "Precision" else 3.0,
        disabled=not use_global_registration
    )
    
    st.subheader("🔩 ICP Parameters")
    icp_threshold = st.slider(
        "ICP Threshold (mm)", 
        0.01, 2.0, 
        value=0.3 if processing_mode == "Balanced" else 0.1 if processing_mode == "Precision" else 0.5
    )
    
    icp_max_iter = st.slider(
        "ICP Max Iterations", 
        10, 2000, 
        value=200 if processing_mode == "Balanced" else 500 if processing_mode == "Precision" else 100
    )
    
    # Visualization
    st.subheader("📊 Visualization")
    point_size = st.slider("Point Size", 1, 10, 3)
    color_scale = st.selectbox(
        "Color Scale", 
        ["viridis", "plasma", "turbo", "hot"]
    )
    
# -------------------------------------------------
# Main Interface
# -------------------------------------------------
st.title("🦷 Dental STL Analyzer Pro")
st.markdown("""
    *Compare dental scan STL files with professional-grade metrics and visualization*
    """)

# File Upload Sections
with st.expander("📤 Upload Files", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Reference STL")
        ref_file = st.file_uploader(
            "Upload Reference .3dm", 
            type=["3dm"],
            help="Rhino file with layered meshes (NOTIMPORTANT, Layer1, Layer2, BaseLayer)"
        )
        
    with col2:
        st.subheader("Test STL(s)")
        test_file = st.file_uploader(
            "Upload Test STL", 
            type=["stl"],
            help="Scan file to analyze against reference"
        )

# Layer weight configuration
LAYER_WEIGHTS = {
    "NOTIMPORTANT": 0.1,
    "Layer1": 0.4,
    "Layer2": 1.0,
    "BaseLayer": 1.0  # Example additional layer
}

def main():
    st.title("3DM Weighted Deviation Analyzer")
    
    # File upload
    ref_file = st.file_uploader("Upload Reference .3dm", type=["3dm"])
    test_file = st.file_uploader("Upload Test STL", type=["stl"])
    
    if ref_file and test_file:
        # Save uploaded files
        ref_path = save_uploaded_file(ref_file)
        test_path = save_uploaded_file(test_file)
        
        # Initialize analyzer with weights
        analyzer = RhinoAnalyzer(LAYER_WEIGHTS)
        try:
            analyzer.load_reference_3dm(ref_path)
        except ValueError as e:
            st.error(f"Reference Error: {str(e)}")
            st.stop()
        
        # Process test file
        try:
            test_mesh = o3d.io.read_triangle_mesh(test_path)
        except Exception as e:
            st.error(f"Test STL Error: {str(e)}")
            st.stop()
        
        test_pcd = test_mesh.sample_points_uniformly(10000)
        test_points = np.asarray(test_pcd.points)
        
        # Calculate deviations
        results = analyzer.calculate_weighted_deviation(test_points)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Raw Deviation", f"{results['mean_raw']:.3f} mm")
            st.metric("Max Raw Deviation", f"{results['max_raw']:.3f} mm")
        with col2:
            st.metric("Mean Weighted Deviation", f"{results['mean_weighted']:.3f} mm", 
                    delta=f"{(results['mean_weighted']-results['mean_raw']):.3f} vs raw")
            st.metric("Max Weighted Deviation", f"{results['max_weighted']:.3f} mm")

        with st.expander("🩺 Mesh Diagnostics"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Reference .3dm Structure**")
                st.json({
                    "Layers": list(LAYER_WEIGHTS.keys()),
                    "Loaded Layers": list(analyzer.reference_layers.keys())
                })
            with col2:
                st.write("**Test Mesh Statistics**")
                st.metric("Vertices", len(test_mesh.vertices))
                st.metric("Triangles", len(test_mesh.triangles))
                st.metric("Watertight", test_mesh.is_watertight())

        with st.expander("🔍 Conversion Report"):
            st.write(f"**Reference Layers Loaded:** {len(analyzer.reference_layers)}")
            st.write(f"**Total Weighted Points:** {len(analyzer.reference.points)}")
            st.write("**Layer Details:**")
            for layer in LAYER_WEIGHTS:
                st.write(f"- {layer}: {analyzer.layer_weights[layer]} weight")

# Analysis Execution
if st.button("🚀 Start Analysis", type="primary"):
    if not ref_file:
        st.error("Please upload a reference STL file!")
        st.stop()
        
    if not test_file:
        st.error("Please upload a test STL file!")
        st.stop()
        
    try:
        analyzer = st.session_state['analyzer']
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Process reference
            status_text.markdown("🔍 **Processing reference file...**")
            ref_path = os.path.join(temp_dir, "reference.stl")
            with open(ref_path, "wb") as f:
                f.write(ref_file.getbuffer())
                
            analyzer.load_reference(ref_path, num_points, 20, 2.0)
            progress_bar.progress(10)
            
            # Process test file
            status_text.markdown("🔬 **Processing test file...**")
            test_path = os.path.join(temp_dir, "test.stl")
            with open(test_path, "wb") as f:
                f.write(test_file.getbuffer())
            
            analyzer.add_test_file(test_path, num_points, 20, 2.0)
            progress_bar.progress(40)
            
            # Run analysis
            result = analyzer.process_test_file(
                test_path,
                use_global_registration,
                voxel_size_global,
                icp_threshold,
                icp_max_iter,
                True
            )
            
            # Display results
            status_text.markdown("📊 **Generating visualizations...**")
            progress_bar.progress(90)
            
            with st.expander("📌 Results", expanded=True):
                col_metrics, col_viz = st.columns([1, 2])
                
                with col_metrics:
                    st.subheader("📈 Metrics Summary")
                    metrics = result['metrics']
                    
                    metric_cols = st.columns(2)
                    with metric_cols[0]:
                        st.metric("Mean Deviation", f"{metrics['mean_deviation']:.3f} mm")
                        st.metric("Volume Similarity", f"{metrics['volume_similarity']*100:.1f}%")
                        
                    with metric_cols[1]:
                        st.metric("Max Deviation", f"{metrics['max_deviation']:.3f} mm")
                        st.metric("Normal Alignment", f"{metrics['mean_normal_angle']:.1f}°")
                        
                    metrics_df = pd.DataFrame({
                        'Value': [
                            f"{metrics['fitness']:.4f}",
                            f"{metrics['inlier_rmse']:.4f}",
                            f"{metrics['hausdorff_distance']:.4f}",
                            f"{metrics['volume_difference']:.4f}",
                            f"{metrics['center_of_mass_distance']:.4f}"
                        ]
                    }, index=['Fitness', 'RMSE', 'Hausdorff', 'Volume Diff', 'CoM Distance'])
                    st.dataframe(metrics_df)
                
                with col_viz:
                    st.subheader("Deviation Analysis")
                    aligned_points = np.asarray(result['aligned_pcd'].points)
                    distances = np.asarray(metrics['distances'])

                    col3, col4 = st.columns(2)
                    with col3:
                        # 3D deviation map
                        fig_heatmap = plot_point_cloud_heatmap(
                            aligned_points,
                            distances,
                            point_size,
                            color_scale,
                            "Deviation Map"
                        )
                        st.plotly_chart(fig_heatmap, use_container_width=True)

                    with col4:
                        # Histogram
                        fig_hist = plot_deviation_histogram(
                            distances,
                            title="Deviation Distribution"
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                    # Export options
                    export_data = pd.DataFrame(
                        np.column_stack((aligned_points, distances)),
                        columns=['X', 'Y', 'Z', 'Deviation']
                    )

                    st.download_button(
                        "📥 Download Results (CSV)",
                        export_data.to_csv(index=False).encode('utf-8'),
                        "results.csv",
                        "text/csv"
                    )
            
            progress_bar.progress(100)
            st.success("✅ Analysis completed successfully!")
            st.balloons()
            
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.exception(e)