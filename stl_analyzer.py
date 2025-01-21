import streamlit as st
import open3d as o3d
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tempfile
import os

# -------------------------------------------------
# Streamlit Page Configuration
# -------------------------------------------------
st.set_page_config(page_title="Dental STL Analyzer", layout="wide")
st.title("🦷 Dental STL Deviation Analyzer")

# -------------------------------------------------
# Utility Functions
# -------------------------------------------------
def load_and_preprocess(
    stl_path: str,
    num_points: int = 2000,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0
) -> o3d.geometry.PointCloud:
    """
    Load, clean, and downsample an STL mesh, then convert to a point cloud.
    """
    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    
    # Convert to point cloud (uniform sampling)
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    
    # Remove outliers from point cloud
    cl, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    
    return cl

def align_point_clouds(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float = 0.1,
    threshold: float = 0.2,
    max_iterations: int = 200
) -> (o3d.geometry.PointCloud, o3d.pipelines.registration.RegistrationResult):
    """
    Align two point clouds using ICP (point-to-point).
    """
    source_down = source.voxel_down_sample(voxel_size=voxel_size)
    target_down = target.voxel_down_sample(voxel_size=voxel_size)

    trans_init = np.identity(4)
    reg_result = o3d.pipelines.registration.registration_icp(
        source_down, 
        target_down, 
        threshold, 
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )
    
    # Transform the original source cloud with the resulting transform
    source.transform(reg_result.transformation)
    
    return source, reg_result

def calculate_deviations(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud
) -> np.ndarray:
    """
    Calculate point-to-point distances from source to target.
    """
    dists = source.compute_point_cloud_distance(target)
    return np.asarray(dists)

def plotly_heatmap(
    points: np.ndarray, 
    deviations: np.ndarray,
    point_size: int = 3,
    color_scale: str = "Hot",
    title: str = "Deviation Map"
) -> go.Figure:
    """
    Create an interactive 3D heatmap using Plotly.
    """
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=deviations,
                colorscale=color_scale,
                opacity=0.8,
                colorbar=dict(title='Deviation (mm)')
            ),
            name="Aligned Test"
        )
    )
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)'
        ),
        margin=dict(l=0, r=0, b=0, t=35)
    )
    return fig

def plot_point_clouds(
    pcd_data: list,
    point_size: int = 3
) -> go.Figure:
    """
    Plot multiple point clouds in a single 3D figure with different colors.
    
    :param pcd_data: List of tuples [(points, color, label), ...].
    :param point_size: Marker size.
    """
    fig = go.Figure()
    
    for (points, color, label) in pcd_data:
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=color,
                    opacity=0.8
                ),
                name=label
            )
        )
    
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        margin=dict(l=0, r=0, b=0, t=35)
    )
    return fig

# -------------------------------------------------
# Sidebar Controls with "help" Text
# -------------------------------------------------
st.sidebar.header("Parameters")

num_points = st.sidebar.slider(
    "Points to Sample",
    min_value=500,
    max_value=20000,
    value=2000,
    step=500,
    help="Number of points to uniformly sample from the STL mesh. Higher = more detail, but slower."
)

nb_neighbors = st.sidebar.slider(
    "Outlier Neighbors",
    min_value=5,
    max_value=50,
    value=20,
    help="Number of neighbors to consider for outlier removal. Larger values can remove more noise, but might remove valid points."
)

std_ratio = st.sidebar.slider(
    "Outlier Std Ratio",
    min_value=1.0,
    max_value=5.0,
    value=2.0,
    step=0.5,
    help="Points outside this standard deviation ratio (in local neighborhoods) are removed as outliers."
)

voxel_size = st.sidebar.slider(
    "Downsample Voxel Size (ICP)",
    min_value=0.01,
    max_value=1.0,
    value=0.1,
    step=0.01,
    help="Voxel size for downsampling during ICP. Helps reduce processing time and noise."
)

threshold = st.sidebar.slider(
    "ICP Threshold (mm)",
    min_value=0.01,
    max_value=2.0,
    value=0.2,
    step=0.01,
    help="Maximum distance (in mm) between corresponding points in ICP. For small dental models, use a smaller threshold."
)

max_iterations = st.sidebar.slider(
    "ICP Max Iterations",
    min_value=50,
    max_value=1000,
    value=200,
    step=50,
    help="Maximum number of ICP iterations. More iterations can refine alignment, but increase computation time."
)

point_size = st.sidebar.slider(
    "3D Plot Point Size",
    min_value=1,
    max_value=10,
    value=3,
    help="Marker size for the 3D scatter plot."
)

color_scale = st.sidebar.selectbox(
    "Heatmap Color Scale",
    ["Hot", "Viridis", "Rainbow", "Blues", "Reds"],
    help="Color mapping scheme for visualizing deviation values in the 3D scatter plot."
)

# Toggles to show reference/test raw point clouds
show_ref_raw = st.sidebar.checkbox(
    "Show Reference (raw)",
    value=False,
    help="Toggle to display the original (unaligned) reference point cloud in a separate 3D plot."
)
show_test_raw = st.sidebar.checkbox(
    "Show Test (raw)",
    value=False,
    help="Toggle to display the original (unaligned) test point cloud in a separate 3D plot."
)

# -------------------------------------------------
# File Uploader
# -------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload exactly two STL files: [Reference first, Test second]",
    type=["stl"],
    accept_multiple_files=True,
    help="Please select 2 STL files. The first is the reference (ideal), the second is the test (to compare)."
)

if uploaded_files:
    if len(uploaded_files) != 2:
        st.error("Please upload exactly 2 STL files (Reference first, then Test).")
    else:
        with st.spinner("Processing..."):
            # Save temp files
            temp_dir = tempfile.TemporaryDirectory()
            file_paths = []
            for i, f in enumerate(uploaded_files):
                path = os.path.join(temp_dir.name, f"scan_{i}.stl")
                with open(path, "wb") as out_file:
                    out_file.write(f.getbuffer())
                file_paths.append(path)
            
            # Load & preprocess
            ref_pcd = load_and_preprocess(
                file_paths[0],
                num_points=num_points,
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
            test_pcd = load_and_preprocess(
                file_paths[1],
                num_points=num_points,
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
            
            # Alignment
            aligned_pcd, reg_result = align_point_clouds(
                test_pcd,
                ref_pcd,
                voxel_size=voxel_size,
                threshold=threshold,
                max_iterations=max_iterations
            )
            
            # Deviation
            deviations = calculate_deviations(aligned_pcd, ref_pcd)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Alignment Metrics")
                df_metrics = pd.DataFrame({
                    "Metric": ["Fitness", "RMSE"],
                    "Value": [reg_result.fitness, reg_result.inlier_rmse]
                })
                st.dataframe(df_metrics.set_index("Metric"), use_container_width=True)
                
                # Offset
                offset = np.asarray(aligned_pcd.points).mean(axis=0) - np.asarray(ref_pcd.points).mean(axis=0)
                st.write("**Mean Position Offset (mm)**:")
                st.write(f"X: {offset[0]:.4f}, Y: {offset[1]:.4f}, Z: {offset[2]:.4f}")
            
            with col2:
                st.subheader("Deviation Statistics")
                st.metric("Max Deviation (mm)", f"{np.max(deviations):.3f}")
                st.metric("Avg Deviation (mm)", f"{np.mean(deviations):.3f}")
                st.metric("Std Dev (mm)", f"{np.std(deviations):.3f}")
                
                st.write("**Deviation Distribution**:")
                hist_data = pd.DataFrame({"Deviation (mm)": deviations})
                st.bar_chart(hist_data, y="Deviation (mm)")
            
            # 3D Deviation Map
            st.subheader("3D Deviation Map")
            aligned_points = np.asarray(aligned_pcd.points)
            fig_heatmap = plotly_heatmap(
                aligned_points,
                deviations,
                point_size=point_size,
                color_scale=color_scale,
                title="Deviation after Alignment"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Optional: Show raw reference/test points
            if show_ref_raw or show_test_raw:
                st.subheader("Raw (Unaligned) Point Clouds")
                data_to_plot = []
                if show_ref_raw:
                    data_to_plot.append((
                        np.asarray(ref_pcd.points),
                        "#00CC96",  # greenish
                        "Reference (raw)"
                    ))
                if show_test_raw:
                    data_to_plot.append((
                        np.asarray(test_pcd.points),
                        "#EF553B",  # reddish
                        "Test (raw)"
                    ))
                
                raw_fig = plot_point_clouds(data_to_plot, point_size=point_size)
                st.plotly_chart(raw_fig, use_container_width=True)
            
            # Export
            aligned_df = pd.DataFrame(
                np.hstack((aligned_points, deviations.reshape(-1, 1))),
                columns=["X", "Y", "Z", "Deviation"]
            )
            st.download_button(
                label="Download Aligned + Deviation Data (CSV)",
                data=aligned_df.to_csv(index=False).encode("utf-8"),
                file_name="aligned_deviation_data.csv"
            )
            
else:
    st.info("Please upload two STL files to begin.")
