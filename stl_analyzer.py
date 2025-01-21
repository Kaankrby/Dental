import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os
from processing import STLAnalyzer
from visualization import plot_point_cloud_heatmap, plot_multiple_point_clouds, plot_deviation_histogram
from utils import validate_file_name

# -------------------------------------------------
# Streamlit Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Dental STL Analyzer (Cropped Reference)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🦷 Dental STL Deviation Analyzer (With Cropped Reference)")

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state['analyzer'] = STLAnalyzer()
    
# -------------------------------------------------
# Sidebar Parameters
# -------------------------------------------------
with st.sidebar:
    st.header("Analysis Parameters")
    
    # Point Cloud Parameters
    st.subheader("Point Cloud Generation")
    num_points = st.number_input("Number of Points", 1000, 100000, 10000)
    nb_neighbors = st.number_input("Neighbors for Outlier Removal", 10, 100, 20)
    std_ratio = st.slider("Standard Deviation Ratio", 0.1, 5.0, 2.0)
    
    # Registration Parameters
    st.subheader("Registration")
    use_global_registration = st.checkbox(
        "Use Global Registration",
        value=True,
        help="If unchecked, only ICP will be used"
    )
    
    if use_global_registration:
        voxel_size_global = st.slider(
            "Voxel Size (Global)",
            0.1, 5.0, 2.0,
            help="Larger value = faster but less accurate"
        )
        
    icp_threshold = st.slider(
        "ICP Distance Threshold",
        0.01, 2.0, 0.2,
        help="Maximum correspondence distance"
    )
    
    icp_max_iter = st.slider(
        "ICP Max Iterations",
        10, 1000, 100,
        help="Maximum ICP iterations"
    )
    
    # Visualization Parameters
    st.subheader("Visualization")
    point_size = st.slider("Point Size", 1, 10, 2)
    color_scale = st.selectbox(
        "Color Scale",
        ["viridis", "plasma", "inferno", "magma"]
    )
    
    ignore_outside_bbox = st.checkbox(
        "Ignore Outside Reference",
        value=True,
        help="Only analyze points within reference bounding box"
    )
    
    show_raw_clouds = st.checkbox(
        "Show Raw Point Clouds",
        value=False,
        help="Display unaligned point clouds"
    )

# -------------------------------------------------
# Main Panel: File Upload and Processing
# -------------------------------------------------
st.subheader("1. Upload Cropped Reference STL")
reference_file = st.file_uploader(
    "Reference (Ideal, Cropped) STL",
    type=["stl"],
    help="Upload your cropped reference STL here"
)

st.subheader("2. Upload Test STL File(s)")
test_files = st.file_uploader(
    "Test (Comparison) STL(s)",
    type=["stl"],
    accept_multiple_files=True,
    help="Upload one or multiple test STL files"
)

run_pressed = st.button("Run Analysis")

if run_pressed:
    if reference_file is None:
        st.error("Please upload a reference STL file!")
    elif len(test_files) == 0:
        st.error("Please upload at least one test STL file!")
    else:
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Process reference
                with st.spinner("Processing reference STL..."):
                    if not validate_file_name(reference_file.name):
                        st.error("Invalid reference file name!")
                        st.stop()
                        
                    ref_path = os.path.join(temp_dir, "reference.stl")
                    with open(ref_path, "wb") as f:
                        f.write(reference_file.getbuffer())
                    
                    analyzer = st.session_state['analyzer']
                    analyzer.load_reference(
                        ref_path,
                        num_points,
                        nb_neighbors,
                        std_ratio
                    )
                    
                    if show_raw_clouds:
                        st.subheader("Reference Point Cloud (Raw)")
                        ref_fig = plot_multiple_point_clouds(
                            [(np.asarray(analyzer.reference_pcd.points),
                              "#00CC96",
                              "Reference")],
                            point_size
                        )
                        st.plotly_chart(ref_fig, use_container_width=True)
                
                # Process test files
                for test_file in test_files:
                    if not validate_file_name(test_file.name):
                        st.error(f"Invalid test file name: {test_file.name}")
                        continue
                        
                    with st.spinner(f"Processing: {test_file.name}"):
                        # Save and process test file
                        test_path = os.path.join(temp_dir, f"test_{test_file.name}")
                        with open(test_path, "wb") as f:
                            f.write(test_file.getbuffer())
                            
                        analyzer.add_test_file(
                            test_path,
                            num_points,
                            nb_neighbors,
                            std_ratio
                        )
                        
                        if show_raw_clouds:
                            st.subheader(f"Test Point Cloud (Raw): {test_file.name}")
                            test_fig = plot_multiple_point_clouds(
                                [(np.asarray(analyzer.test_meshes[test_path]['pcd'].points),
                                  "#EF553B",
                                  f"Test: {test_file.name}")],
                                point_size
                            )
                            st.plotly_chart(test_fig, use_container_width=True)
                        
                        # Process and analyze
                        result = analyzer.process_test_file(
                            test_path,
                            use_global_registration,
                            voxel_size_global if use_global_registration else 1.0,
                            icp_threshold,
                            icp_max_iter,
                            ignore_outside_bbox
                        )
                        
                        # Display results
                        with st.expander(f"Results: {test_file.name}", expanded=True):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Registration Metrics**")
                                metrics_df = pd.DataFrame({
                                    'Metric': ['Fitness', 'RMSE', 'Hausdorff', 'Volume Diff'],
                                    'Value': [
                                        f"{result['metrics']['fitness']:.4f}",
                                        f"{result['metrics']['inlier_rmse']:.4f}",
                                        f"{result['metrics']['hausdorff_distance']:.4f}",
                                        f"{result['metrics']['volume_difference']:.4f}"
                                    ]
                                })
                                st.dataframe(metrics_df.set_index('Metric'))
                            
                            with col2:
                                st.write("**Position Offset**")
                                offset = result['metrics']['center_of_mass_distance']
                                st.metric("Center of Mass Offset (mm)", f"{offset:.4f}")
                            
                            # Deviation visualization
                            st.subheader("Deviation Analysis")
                            aligned_points = np.asarray(result['aligned_pcd'].points)
                            distances = np.asarray(result['metrics']['distances'])
                            
                            col3, col4 = st.columns(2)
                            with col3:
                                # 3D deviation map
                                fig_heatmap = plot_point_cloud_heatmap(
                                    aligned_points,
                                    distances,
                                    point_size,
                                    color_scale,
                                    f"Deviation Map: {test_file.name}"
                                )
                                st.plotly_chart(fig_heatmap, use_container_width=True)
                            
                            with col4:
                                # Histogram
                                fig_hist = plot_deviation_histogram(
                                    distances,
                                    title=f"Deviation Distribution: {test_file.name}"
                                )
                                st.plotly_chart(fig_hist, use_container_width=True)
                            
                            # Export options
                            export_data = pd.DataFrame(
                                np.column_stack((aligned_points, distances)),
                                columns=['X', 'Y', 'Z', 'Deviation']
                            )
                            
                            st.download_button(
                                "Download Results (CSV)",
                                export_data.to_csv(index=False).encode('utf-8'),
                                f"results_{test_file.name}.csv",
                                "text/csv"
                            )
                
                st.success("Analysis complete! 🎉")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
else:
    st.info("Upload files and click 'Run Analysis' to begin.")