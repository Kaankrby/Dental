import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os
from processing import STLAnalyzer
from visualization import (
    plot_point_cloud_heatmap, 
    plot_multiple_point_clouds,
    plot_deviation_histogram,
    plot_normal_angle_distribution
)
from utils import validate_file_name
from streamlit.runtime.scriptrunner import get_script_run_ctx

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
        reference_file = st.file_uploader(
            "Upload reference (ideal) STL",
            type=["stl"],
            key="ref_uploader"
        )
        
    with col2:
        st.subheader("Test STL(s)")
        test_files = st.file_uploader(
            "Upload test STL(s) for comparison",
            type=["stl"],
            accept_multiple_files=True,
            key="test_uploader"
        )

# Analysis Execution
if st.button("🚀 Start Analysis", type="primary"):
    if not reference_file:
        st.error("Please upload a reference STL file!")
        st.stop()
        
    if not test_files:
        st.error("Please upload at least one test STL file!")
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
                f.write(reference_file.getbuffer())
                
            analyzer.load_reference(ref_path, num_points, 20, 2.0)
            progress_bar.progress(10)
            
            # Process test files
            results = {}
            for i, test_file in enumerate(test_files):
                status_text.markdown(f"🔬 **Processing test file {i+1}/{len(test_files)}: {test_file.name}...**")
                test_path = os.path.join(temp_dir, f"test_{test_file.name}")
                with open(test_path, "wb") as f:
                    test_file.seek(0)
                    f.write(test_file.getbuffer())
                
                analyzer.add_test_file(test_path, num_points, 20, 2.0)
                progress_bar.progress(10 + int(30*(i+1)/len(test_files)))
                
                # Run analysis
                result = analyzer.process_test_file(
                    test_path,
                    use_global_registration,
                    voxel_size_global,
                    icp_threshold,
                    icp_max_iter,
                    True
                )
                results[test_file.name] = result
                progress_bar.progress(40 + int(50*(i+1)/len(test_files)))
            
            # Display results
            status_text.markdown("📊 **Generating visualizations...**")
            progress_bar.progress(90)
            
            for test_name, result in results.items():
                with st.expander(f"📌 Results: {test_name}", expanded=True):
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
                        st.subheader("📐 3D Visualization")
                        tab1, tab2, tab3 = st.tabs(["Deviation Map", "Normal Angles", "Histograms"])
                        
                        with tab1:
                            fig = plot_point_cloud_heatmap(
                                np.asarray(result['aligned_pcd'].points),
                                result['metrics']['distances'],
                                point_size,
                                color_scale,
                                f"Deviation Map: {test_name}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                        with tab2:
                            fig = plot_normal_angle_distribution(
                                np.asarray(result['aligned_pcd'].points),
                                result['metrics']['normal_angles'],
                                point_size,
                                color_scale
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                        with tab3:
                            col_hist1, col_hist2 = st.columns(2)
                            with col_hist1:
                                fig = plot_deviation_histogram(
                                    result['metrics']['distances'],
                                    title="Deviation Distribution"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            with col_hist2:
                                fig = plot_deviation_histogram(
                                    result['metrics']['normal_angles'],
                                    title="Normal Angle Distribution"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # Export section
                    st.download_button(
                        label="⬇️ Download Full Report",
                        data=generate_report(result),
                        file_name=f"report_{test_name}.pdf",
                        mime="application/pdf"
                    )
            
            progress_bar.progress(100)
            st.success("✅ Analysis completed successfully!")
            st.balloons()
            
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.exception(e)