import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os
from processing import STLAnalyzer
from visualization import (
    plot_cavity_deviation_map,
    plot_cavity_analysis_summary,
    plot_deviation_histogram,
    plot_reference_points,
    plot_preparation_zones
)
from utils import validate_file_name, compute_region_weights
import plotly.express as px

# -------------------------------------------------
# Streamlit Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Dental Cavity Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add help text and tooltips
with st.sidebar:
    st.markdown("""
    ### 📋 Instructions
    1. Upload a reference cavity model (STL)
    2. Adjust analysis parameters if needed
    3. Upload student models to analyze
    4. Click 'Analyze Preparations' to start
    
    ### 🎯 Tips
    - Enable region detection to analyze specific areas
    - Adjust weights to prioritize important regions
    - Use the tolerance range to set acceptable deviations
    """)
    
    with st.expander("ℹ️ About Parameters"):
        st.markdown("""
        **Point Cloud Generation**
        - Higher point counts give more detailed analysis but take longer
        - Neighbors and cleaning strength control outlier removal
        
        **Region Detection**
        - Larger region size merges nearby areas
        - Minimum points prevents tiny regions
        
        **Visualization**
        - Adjust point size for better visibility
        - Different color scales highlight different aspects
        """)

# Add file type validation
def validate_stl_file(uploaded_file) -> bool:
    """Validate uploaded STL file."""
    if uploaded_file is None:
        return False
    
    if not uploaded_file.name.lower().endswith('.stl'):
        st.error(f"Invalid file type: {uploaded_file.name}. Please upload an STL file.")
        return False
    
    if uploaded_file.size > 50 * 1024 * 1024:  # 50MB limit
        st.error(f"File too large: {uploaded_file.name}. Maximum size is 50MB.")
        return False
    
    return True

# Add session state for analysis results
if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = {}

# Add export options
def export_results_to_csv(results: dict, filename: str):
    """Export analysis results to CSV."""
    try:
        metrics_df = pd.DataFrame(results['metrics'])
        csv = metrics_df.to_csv(index=False)
        st.download_button(
            "📊 Download Results (CSV)",
            csv,
            filename,
            "text/csv",
            key=f'download_{filename}'
        )
    except Exception as e:
        st.error(f"Error exporting results: {str(e)}")

# Add comparison features
def compare_results(results_dict: dict):
    """Compare multiple analysis results."""
    if len(results_dict) > 1:
        st.subheader("📈 Comparison Analysis")
        
        # Prepare comparison data
        comparison_data = []
        for filename, result in results_dict.items():
            metrics = result['metrics']
            comparison_data.append({
                'File': filename,
                'Max Deviation': metrics['max_deviation'],
                'Mean Deviation': metrics['mean_deviation'],
                'Score': metrics.get('weighted_score', 0),
                'Points': metrics['points_in_cavity']
            })
        
        # Create comparison table
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df.set_index('File'))
        
        # Create comparison charts
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                comparison_df,
                x='File',
                y=['Max Deviation', 'Mean Deviation'],
                title='Deviation Comparison',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Score' in comparison_df.columns:
                fig = px.bar(
                    comparison_df,
                    x='File',
                    y='Score',
                    title='Score Comparison'
                )
                st.plotly_chart(fig, use_container_width=True)

# Main interface
st.title("🦷 Dental Cavity Preparation Analyzer")
st.markdown("""
This tool analyzes student cavity preparations against a reference model. 
The reference model should be pre-cropped to show only the cavity region of interest.
""")

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state['analyzer'] = STLAnalyzer()
    st.session_state['reference_regions'] = None
    st.session_state['region_weights'] = None

# -------------------------------------------------
# Sidebar Parameters
# -------------------------------------------------
with st.sidebar:
    st.header("Analysis Parameters")
    
    # Point Cloud Parameters
    st.subheader("Point Cloud Generation")
    num_points = st.number_input(
        "Number of Points",
        1000, 100000, 10000,
        help="Number of points to sample from each model"
    )
    nb_neighbors = st.number_input(
        "Neighbors for Cleaning",
        10, 100, 20,
        help="Number of neighbors for outlier removal"
    )
    std_ratio = st.slider(
        "Cleaning Strength",
        0.1, 5.0, 2.0,
        help="Higher values remove more outliers"
    )
    
    # Region Detection Parameters
    st.subheader("Region Detection")
    region_detection = st.checkbox(
        "Enable Region Detection",
        value=True,
        help="Detect and analyze distinct regions in the cavity"
    )
    
    if region_detection:
        eps = st.slider(
            "Region Size",
            0.1, 2.0, 0.5,
            help="Larger values merge regions, smaller values create more regions"
        )
        min_samples = st.slider(
            "Minimum Region Points",
            5, 50, 10,
            help="Minimum points to form a region"
        )
    
    # Analysis Parameters
    st.subheader("Cavity Analysis")
    tolerance_range = st.slider(
        "Tolerance Range (mm)",
        0.1, 2.0, 0.5,
        help="Acceptable deviation range for cavity preparation"
    )
    
    # Visualization Parameters
    st.subheader("Visualization")
    point_size = st.slider("Point Size", 1, 10, 2)
    color_scale = st.selectbox(
        "Color Scale",
        ["RdYlBu", "viridis", "plasma", "inferno"]
    )

# -------------------------------------------------
# Main Panel: File Upload and Processing
# -------------------------------------------------
st.subheader("1. Upload Pre-cropped Reference Cavity")
reference_file = st.file_uploader(
    "Reference Cavity Model (STL)",
    type=["stl"],
    help="Upload the pre-cropped reference cavity model"
)

if reference_file and validate_stl_file(reference_file):
    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save and process reference file
            ref_path = os.path.join(temp_dir, "reference.stl")
            with open(ref_path, "wb") as f:
                f.write(reference_file.getbuffer())
            
            # Process reference with region detection if enabled
            analyzer = st.session_state['analyzer']
            analyzer.load_reference(
                ref_path,
                num_points,
                nb_neighbors,
                std_ratio,
                detect_regions=region_detection,
                eps=eps if region_detection else None,
                min_samples=min_samples if region_detection else None
            )
            
            # Display reference model
            st.subheader("Reference Model Visualization")
            col1, col2 = st.columns(2)
            
            with col1:
                # Show reference points
                ref_fig = plot_reference_points(
                    analyzer.reference_points,
                    point_size,
                    "Reference Cavity Points"
                )
                st.plotly_chart(ref_fig, use_container_width=True)
            
            if region_detection and analyzer.reference_regions:
                with col2:
                    # Show detected regions
                    st.subheader("Detected Cavity Regions")
                    region_fig = plot_preparation_zones(
                        analyzer.reference_points,
                        analyzer.region_labels,
                        point_size,
                        "Cavity Regions"
                    )
                    st.plotly_chart(region_fig, use_container_width=True)
                
                # Region weights
                st.subheader("Region Weights")
                weights = compute_region_weights(analyzer.reference_regions)
                
                # Allow manual weight adjustment
                adjusted_weights = []
                cols = st.columns(len(weights))
                for i, (w, col) in enumerate(zip(weights, cols)):
                    with col:
                        adjusted_weight = st.slider(
                            f"Region {i+1}",
                            0.0, 1.0, float(w),
                            help=f"Weight for region {i+1}"
                        )
                        adjusted_weights.append(adjusted_weight)
                
                # Normalize adjusted weights
                adjusted_weights = np.array(adjusted_weights)
                adjusted_weights = adjusted_weights / np.sum(adjusted_weights)
                st.session_state['region_weights'] = adjusted_weights
                
    except Exception as e:
        st.error(f"Error processing reference file: {str(e)}")

# -------------------------------------------------
# Student Model Analysis
# -------------------------------------------------
st.subheader("2. Upload Student Models")
test_files = st.file_uploader(
    "Student Models (STL)",
    type=["stl"],
    accept_multiple_files=True,
    help="Upload one or more student cavity preparations"
)

run_pressed = st.button("Analyze Preparations")

if run_pressed:
    if reference_file is None:
        st.error("Please upload a reference cavity model!")
    elif len(test_files) == 0:
        st.error("Please upload at least one student model!")
    else:
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Process student models
                for test_file in test_files:
                    if not validate_stl_file(test_file):
                        continue
                        
                    with st.spinner(f"Analyzing: {test_file.name}"):
                        # Save and process test file
                        test_path = os.path.join(temp_dir, f"test_{test_file.name}")
                        with open(test_path, "wb") as f:
                            f.write(test_file.getbuffer())
                            
                        analyzer = st.session_state['analyzer']
                        analyzer.add_test_file(
                            test_path,
                            num_points,
                            nb_neighbors,
                            std_ratio
                        )
                        
                        # Process and analyze
                        result = analyzer.process_test_file(
                            test_path,
                            use_regions=region_detection,
                            region_weights=st.session_state.get('region_weights'),
                            tolerance=tolerance_range
                        )
                        
                        # Display results
                        with st.expander(f"Analysis Results: {test_file.name}", expanded=True):
                            metrics = result['metrics']
                            
                            # Summary statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Maximum Deviation",
                                    f"{metrics['max_deviation']:.2f} mm"
                                )
                            with col2:
                                st.metric(
                                    "Average Deviation",
                                    f"{metrics['mean_deviation']:.2f} mm"
                                )
                            with col3:
                                if 'weighted_score' in metrics:
                                    st.metric(
                                        "Overall Score",
                                        f"{metrics['weighted_score']:.1f}%"
                                    )
                                else:
                                    st.metric(
                                        "Points Analyzed",
                                        f"{metrics['points_in_cavity']}"
                                    )
                            
                            # Preparation quality summary
                            st.subheader("Preparation Quality Analysis")
                            quality_fig = plot_cavity_analysis_summary(
                                metrics,
                                f"Preparation Quality: {test_file.name}"
                            )
                            st.plotly_chart(quality_fig, use_container_width=True)
                            
                            # Region-specific results
                            if 'region_metrics' in metrics:
                                st.subheader("Region Analysis")
                                region_df = pd.DataFrame(metrics['region_metrics'])
                                st.dataframe(
                                    region_df.set_index('label'),
                                    use_container_width=True
                                )
                            
                            # Detailed visualizations
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("3D Deviation Map")
                                cavity_points = metrics['cavity_points']
                                distances = metrics['distances']
                                
                                fig_map = plot_cavity_deviation_map(
                                    cavity_points,
                                    distances,
                                    point_size,
                                    color_scale,
                                    f"Cavity Deviations: {test_file.name}",
                                    tolerance_range=tolerance_range
                                )
                                st.plotly_chart(fig_map, use_container_width=True)
                            
                            with col2:
                                st.subheader("Deviation Distribution")
                                fig_hist = plot_deviation_histogram(
                                    distances,
                                    title=f"Deviation Distribution: {test_file.name}",
                                    tolerance_range=tolerance_range
                                )
                                st.plotly_chart(fig_hist, use_container_width=True)
                            
                            # Export options
                            st.subheader("Export Results")
                            export_data = pd.DataFrame(
                                np.column_stack((cavity_points, distances)),
                                columns=['X', 'Y', 'Z', 'Deviation']
                            )
                            
                            export_results_to_csv(result, f"cavity_analysis_{test_file.name}")
                
                st.success("Analysis complete! 🎉")
                
                # Compare results
                compare_results(st.session_state['analysis_results'])
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
else:
    st.info("Upload models and click 'Analyze Preparations' to begin.")