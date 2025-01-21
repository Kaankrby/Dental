import streamlit as st
import tempfile
import os
from processing import STLAnalyzer
from visualization import plot_points, plot_deviation_histogram

# Page config
st.set_page_config(
    page_title="STL Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("STL Analyzer")
st.markdown("""
This tool analyzes STL files by comparing them with a reference model.
Upload your reference model first, then upload test models to analyze.
""")

# Sidebar parameters
with st.sidebar:
    st.header("Parameters")
    
    num_points = st.number_input(
        "Number of Points",
        min_value=1000,
        max_value=100000,
        value=10000,
        step=1000,
        help="Number of points to sample from the mesh surface"
    )
    
    tolerance = st.number_input(
        "Tolerance (mm)",
        min_value=0.01,
        max_value=2.0,
        value=0.1,
        step=0.05,
        help="Maximum acceptable deviation"
    )
    
    point_size = st.slider(
        "Point Size",
        min_value=1,
        max_value=10,
        value=3,
        help="Size of points in visualization"
    )

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state['analyzer'] = STLAnalyzer()

# File upload section
st.header("Upload Files")

col1, col2 = st.columns(2)

with col1:
    reference_file = st.file_uploader(
        "Upload Reference STL",
        type=['stl'],
        help="Upload the reference model"
    )

with col2:
    test_files = st.file_uploader(
        "Upload Test STL(s)",
        type=['stl'],
        accept_multiple_files=True,
        help="Upload one or more test models to analyze"
    )

# Process reference file
if reference_file:
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            ref_path = os.path.join(temp_dir, "reference.stl")
            with open(ref_path, "wb") as f:
                f.write(reference_file.getbuffer())
            
            analyzer = st.session_state['analyzer']
            analyzer.load_reference(ref_path, num_points)
            
            st.success("Reference model loaded successfully!")
            
            # Visualize reference points
            st.subheader("Reference Model")
            ref_fig = plot_points(
                analyzer.reference_points,
                point_size,
                "Reference Points"
            )
            st.plotly_chart(ref_fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error loading reference file: {str(e)}")

# Process test files
if reference_file and test_files:
    st.header("Analysis Results")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            for test_file in test_files:
                st.subheader(f"Analyzing: {test_file.name}")
                
                # Save and process test file
                test_path = os.path.join(temp_dir, test_file.name)
                with open(test_path, "wb") as f:
                    f.write(test_file.getbuffer())
                
                # Analyze test file
                analyzer = st.session_state['analyzer']
                result = analyzer.process_test_file(test_path, num_points, tolerance)
                
                # Display metrics
                metrics = result['metrics']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Max Deviation", f"{metrics['max_deviation']:.3f} mm")
                
                with col2:
                    st.metric("Mean Deviation", f"{metrics['mean_deviation']:.3f} mm")
                
                with col3:
                    st.metric(
                        "Points Within Tolerance",
                        f"{metrics['tolerance_percentage']:.1f}%"
                    )
                
                # Visualize results
                col1, col2 = st.columns(2)
                
                with col1:
                    test_fig = plot_points(
                        result['test_points'],
                        point_size,
                        "Test Points"
                    )
                    st.plotly_chart(test_fig, use_container_width=True)
                
                with col2:
                    hist_fig = plot_deviation_histogram(
                        metrics['max_deviation'],
                        metrics['mean_deviation'],
                        metrics['std_deviation']
                    )
                    st.plotly_chart(hist_fig, use_container_width=True)
                
                st.divider()
                
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
else:
    st.info("Please upload both reference and test files to start analysis.")