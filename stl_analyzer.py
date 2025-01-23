import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os
from processing import STLAnalyzer
from visualization import (
    plot_point_cloud_heatmap,
    plot_multiple_point_clouds,
    plot_deviation_histogram
)

st.set_page_config(
    page_title="Dental STL Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'analyzer' not in st.session_state:
    st.session_state.analyzer = STLAnalyzer()

# Sidebar Controls
with st.sidebar:
    st.header("Parameters")
    num_points = st.slider("Sample Points", 1000, 50000, 10000)
    voxel_size = st.slider("Voxel Size (mm)", 0.1, 5.0, 1.5)
    icp_threshold = st.slider("ICP Threshold (mm)", 0.1, 2.0, 0.5)
    point_size = st.slider("Point Size", 1, 5, 2)

# Main Interface
st.title("Dental STL Analyzer")

# File Upload
ref_file = st.file_uploader("Reference STL", type=["stl"])
test_files = st.file_uploader("Test STLs", type=["stl"], accept_multiple_files=True)

if st.button("Run Analysis"):
    if ref_file and test_files:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Process reference
                ref_path = os.path.join(temp_dir, "ref.stl")
                ref_file.seek(0)
                with open(ref_path, "wb") as f:
                    f.write(ref_file.getbuffer())
                
                st.session_state.analyzer.load_reference(ref_path, num_points, 20, 2.0)
                
                # Process test files
                results = {}
                for test_file in test_files:
                    test_path = os.path.join(temp_dir, test_file.name)
                    with open(test_path, "wb") as f:
                        test_file.seek(0)
                        f.write(test_file.getbuffer())
                    
                    st.session_state.analyzer.add_test_file(test_path, num_points, 20, 2.0)
                    results[test_file.name] = st.session_state.analyzer.process_test_file(
                        test_path, voxel_size, icp_threshold
                    )
                
                # Display results
                for name, result in results.items():
                    with st.expander(f"Results: {name}", expanded=True):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.dataframe(pd.DataFrame(result['metrics'], index=['Value']))
                            
                        with col2:
                            fig = plot_point_cloud_heatmap(
                                np.asarray(result['aligned_pcd'].points),
                                result['aligned_pcd'].compute_point_cloud_distance(
                                    st.session_state.analyzer.reference_pcd
                                ),
                                point_size,
                                "viridis",
                                f"Deviations: {name}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                st.success("Analysis complete!")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")