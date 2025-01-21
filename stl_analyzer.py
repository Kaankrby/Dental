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
st.set_page_config(page_title="Dental STL Analyzer - Multiple Tests", layout="wide")
st.title("🦷 Dental STL Deviation Analyzer")

# -------------------------------------------------
# Utility Functions
# -------------------------------------------------
def load_mesh(stl_path: str) -> o3d.geometry.TriangleMesh:
    """
    Load and clean an STL mesh.
    """
    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    return mesh

def sample_point_cloud(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int,
    nb_neighbors: int,
    std_ratio: float
) -> o3d.geometry.PointCloud:
    """
    Convert mesh to point cloud via uniform sampling, then remove outliers.
    """
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    
    # Remove outliers
    pcd_clean, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, 
        std_ratio=std_ratio
    )
    return pcd_clean

def prepare_for_global_registration(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float
) -> (o3d.geometry.PointCloud, o3d.pipelines.registration.Feature):
    """
    Downsample the point cloud, estimate normals, and compute FPFH features
    to prepare for RANSAC-based global registration.
    """
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0*voxel_size, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=5.0*voxel_size, max_nn=100)
    )
    return pcd_down, fpfh

def global_registration_ransac(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    source_fpfh: o3d.pipelines.registration.Feature,
    target_fpfh: o3d.pipelines.registration.Feature,
    voxel_size: float
) -> o3d.pipelines.registration.RegistrationResult:
    """
    Perform RANSAC-based global registration to get a coarse alignment.
    """
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result

def refine_registration_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    icp_threshold: float,
    max_iterations: int
) -> o3d.pipelines.registration.RegistrationResult:
    """
    Refine alignment using ICP, starting from the given initial transform.
    """
    # Estimate normals
    source.estimate_normals()
    target.estimate_normals()
    
    result_icp = o3d.pipelines.registration.registration_icp(
        source,
        target,
        icp_threshold,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )
    return result_icp

def transform_point_cloud(
    pcd: o3d.geometry.PointCloud,
    transformation: np.ndarray
) -> o3d.geometry.PointCloud:
    """
    Return a copy of the point cloud transformed by the given 4x4 matrix.
    """
    pcd_copy = o3d.geometry.PointCloud(pcd)
    pcd_copy.transform(transformation)
    return pcd_copy

def compute_deviations(
    source_aligned: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud
) -> np.ndarray:
    """
    Compute point-to-point distances from source to target.
    """
    dists = source_aligned.compute_point_cloud_distance(target)
    return np.asarray(dists)

def plot_point_cloud_heatmap(
    points: np.ndarray,
    values: np.ndarray,
    point_size: int,
    color_scale: str,
    title: str
) -> go.Figure:
    """
    Plot a 3D scatter of points with scalar values as color.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=points[:,0],
            y=points[:,1],
            z=points[:,2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=values,
                colorscale=color_scale,
                opacity=0.8,
                colorbar=dict(title='Deviation (mm)')
            ),
            name="Aligned Source"
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        margin=dict(l=0, r=0, b=0, t=35)
    )
    return fig

# -------------------------------------------------
# Sidebar with Parameters
# -------------------------------------------------
st.sidebar.header("Global Registration (RANSAC)")

use_global_registration = st.sidebar.checkbox(
    "Enable RANSAC?",
    value=True,
    help="Enable a coarse alignment step using RANSAC & FPFH features. If unchecked, only ICP is used."
)

voxel_size_global = st.sidebar.slider(
    "RANSAC Voxel Size",
    min_value=0.01,
    max_value=2.0,
    value=0.5,
    step=0.01,
    help="Voxel size for downsampling & FPFH. For small dental models, ~0.2-0.5 often works."
)

st.sidebar.header("ICP Parameters")

icp_threshold = st.sidebar.slider(
    "ICP Threshold (mm)",
    min_value=0.01,
    max_value=2.0,
    value=0.2,
    step=0.01,
    help="Max distance for ICP correspondences. Smaller for tiny dental models."
)

icp_max_iter = st.sidebar.slider(
    "ICP Max Iterations",
    min_value=50,
    max_value=2000,
    value=300,
    step=50,
    help="Higher iterations can improve alignment, but take longer."
)

st.sidebar.header("Point Cloud Sampling / Outlier Removal")

num_points = st.sidebar.slider(
    "Points to Sample",
    min_value=500,
    max_value=30000,
    value=3000,
    step=500,
    help="Number of points to uniformly sample from each mesh."
)

nb_neighbors = st.sidebar.slider(
    "Outlier Neighbors",
    min_value=5,
    max_value=50,
    value=20,
    help="Neighbor count for statistical outlier removal."
)

std_ratio = st.sidebar.slider(
    "Outlier Std Ratio",
    min_value=1.0,
    max_value=5.0,
    value=2.0,
    step=0.5,
    help="Std ratio for outlier removal. Points beyond this are removed."
)

st.sidebar.header("Visualization")

point_size = st.sidebar.slider(
    "3D Point Size",
    1, 10, 3,
    help="Marker size in the 3D plot."
)

color_scale = st.sidebar.selectbox(
    "Heatmap Color Scale",
    ["Hot", "Viridis", "Rainbow", "Blues", "Reds"],
    help="Color mapping for deviation values."
)

# -------------------------------------------------
# Main File Upload Panels
# -------------------------------------------------
st.subheader("1) Upload Reference STL")
ref_file = st.file_uploader(
    "Reference STL (Ideal)",
    type=["stl"],
    help="Upload the main reference/ideal dental model here."
)

st.subheader("2) Upload Test STL(s)")
test_files = st.file_uploader(
    "Test STL(s)",
    type=["stl"],
    accept_multiple_files=True,
    help="Upload one or more test models to compare against the reference."
)

# -------------------------------------------------
# Run Button
# -------------------------------------------------
if st.button("Run Alignment and Deviation Analysis"):
    # We only proceed if we have a reference and at least one test STL
    if ref_file is None:
        st.warning("Please upload a reference STL before running.")
        st.stop()
    if not test_files:
        st.warning("Please upload at least one test STL before running.")
        st.stop()

    with st.spinner("Processing..."):
        temp_dir = tempfile.TemporaryDirectory()

        # --- Load Reference ---
        ref_path = os.path.join(temp_dir.name, "ref.stl")
        with open(ref_path, "wb") as out_file:
            out_file.write(ref_file.getbuffer())
        ref_mesh = load_mesh(ref_path)
        ref_pcd = sample_point_cloud(ref_mesh, num_points, nb_neighbors, std_ratio)

        # Precompute data for RANSAC if needed
        if use_global_registration:
            ref_down, ref_fpfh = prepare_for_global_registration(ref_pcd, voxel_size_global)

        # We'll collect results from each test file in a list
        all_results = []

        # --- Process each Test File ---
        for i, test_file in enumerate(test_files):
            test_path = os.path.join(temp_dir.name, f"test_{i}.stl")
            with open(test_path, "wb") as out_file:
                out_file.write(test_file.getbuffer())

            test_mesh = load_mesh(test_path)
            test_pcd_raw = sample_point_cloud(test_mesh, num_points, nb_neighbors, std_ratio)

            # --- Global Registration (RANSAC) ---
            transform_init = np.eye(4)
            if use_global_registration:
                test_down, test_fpfh = prepare_for_global_registration(test_pcd_raw, voxel_size_global)
                ransac_result = global_registration_ransac(
                    test_down, 
                    ref_down, 
                    test_fpfh, 
                    ref_fpfh, 
                    voxel_size_global
                )
                transform_init = ransac_result.transformation

            # --- ICP Refinement ---
            icp_result = refine_registration_icp(
                test_pcd_raw,
                ref_pcd,
                transform_init,
                icp_threshold,
                icp_max_iter
            )

            # Transform test cloud using final ICP matrix
            test_aligned = transform_point_cloud(test_pcd_raw, icp_result.transformation)

            # Compute deviations
            distances = compute_deviations(test_aligned, ref_pcd)
            offset = test_aligned.points.mean(axis=0) - ref_pcd.points.mean(axis=0)

            # Prepare results dictionary
            result_dict = {
                "test_filename": test_file.name,
                "fitness": icp_result.fitness,
                "rmse": icp_result.inlier_rmse,
                "max_deviation": distances.max(),
                "avg_deviation": distances.mean(),
                "std_deviation": distances.std(),
                "offset_x": offset[0],
                "offset_y": offset[1],
                "offset_z": offset[2],
                "aligned_points": np.asarray(test_aligned.points),
                "distances": distances
            }
            all_results.append(result_dict)

        # --- Display Results ---
        st.success("All test models have been aligned successfully!")
        
        # Summarize each test in a collapsible section
        for result in all_results:
            with st.expander(f"Results for {result['test_filename']}"):
                c1, c2 = st.columns(2)
                
                with c1:
                    st.subheader("Alignment Metrics")
                    st.write("**Fitness & RMSE**")
                    df_metrics = pd.DataFrame({
                        "Metric": ["Fitness", "RMSE"],
                        "Value": [result["fitness"], result["rmse"]]
                    })
                    st.dataframe(df_metrics.set_index("Metric"))
                    
                    st.write("**Mean Position Offset (mm)**")
                    st.write(f"X: {result['offset_x']:.4f}, "
                             f"Y: {result['offset_y']:.4f}, "
                             f"Z: {result['offset_z']:.4f}")
                
                with c2:
                    st.subheader("Deviation Statistics")
                    st.metric("Max Deviation (mm)", f"{result['max_deviation']:.3f}")
                    st.metric("Avg Deviation (mm)", f"{result['avg_deviation']:.3f}")
                    st.metric("Std Dev (mm)", f"{result['std_deviation']:.3f}")
                    
                    st.write("**Deviation Distribution**")
                    dist_df = pd.DataFrame({"Deviation (mm)": result["distances"]})
                    st.bar_chart(dist_df, y="Deviation (mm)")

                # 3D Heatmap
                st.subheader("3D Deviation Map")
                fig_heatmap = plot_point_cloud_heatmap(
                    result["aligned_points"],
                    result["distances"],
                    point_size,
                    color_scale,
                    title=f"Deviation: {result['test_filename']}"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Download CSV with aligned coords + deviation
                output_df = pd.DataFrame(
                    np.column_stack([
                        result["aligned_points"],
                        result["distances"]
                    ]),
                    columns=["X", "Y", "Z", "Deviation"]
                )
                st.download_button(
                    label="Download Aligned + Deviation (CSV)",
                    data=output_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{result['test_filename']}_aligned_data.csv"
                )

else:
    st.info("Please upload your reference and test files, then click 'Run Alignment and Deviation Analysis'.")
