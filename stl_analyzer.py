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
st.set_page_config(page_title="Dental STL Analyzer (Global+ICP, Multi-Test)", layout="wide")
st.title("🦷 Dental STL Deviation Analyzer")

# -------------------------------------------------
# Utility Functions
# -------------------------------------------------
def load_mesh(stl_path: str) -> o3d.geometry.TriangleMesh:
    """
    Load and clean an STL mesh (removing degenerate, duplicated, or non-manifold elements).
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
    Convert mesh to point cloud via uniform sampling, then remove statistical outliers.
    """
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    
    # Remove outliers
    pcd_clean, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
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

def plot_multiple_point_clouds(
    pcd_data: list,
    point_size: int
) -> go.Figure:
    """
    Plot multiple point clouds in one 3D figure, each with a distinct color or label.
    :param pcd_data: List[ (points_ndarray, color_str, label_str) ]
    """
    fig = go.Figure()
    for (points, color, label) in pcd_data:
        fig.add_trace(
            go.Scatter3d(
                x=points[:,0],
                y=points[:,1],
                z=points[:,2],
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
# Sidebar with Parameters
# -------------------------------------------------
st.sidebar.header("Global Registration (RANSAC)")
use_global_registration = st.sidebar.checkbox(
    "Use Global Registration (RANSAC)?",
    value=True,
    help="Enable a coarse alignment step using RANSAC & FPFH features. If unchecked, only ICP is used."
)

voxel_size_global = st.sidebar.slider(
    "Global Reg. Voxel Size",
    min_value=0.01,
    max_value=2.0,
    value=0.5,
    step=0.01,
    help="Downsampling voxel size for RANSAC feature extraction. For small dental models, ~0.2-0.5 might work."
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
    help="Higher iterations can improve accuracy but take longer."
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
    help="Number of neighbors for statistical outlier removal."
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
    help="Marker size in 3D plots."
)

color_scale = st.sidebar.selectbox(
    "Heatmap Color Scale",
    ["Hot", "Viridis", "Rainbow", "Blues", "Reds"],
    help="Color mapping for deviation values."
)

show_ref_raw = st.sidebar.checkbox(
    "Show Reference (raw)?",
    value=False,
    help="If checked, displays unaligned reference point cloud."
)

show_test_raw = st.sidebar.checkbox(
    "Show Test (raw)?",
    value=False,
    help="If checked, displays unaligned test point cloud(s)."
)

# -------------------------------------------------
# Main Panel: Separate Upload for Reference & Test
# -------------------------------------------------
st.subheader("1. Upload Reference STL")
reference_file = st.file_uploader(
    label="Reference (Ideal) STL",
    type=["stl"],
    help="Upload your reference STL here."
)

st.subheader("2. Upload One or More Test STL Files")
test_files = st.file_uploader(
    label="Test (Comparison) STL(s)",
    type=["stl"],
    accept_multiple_files=True,
    help="Upload one or multiple test STL files here."
)

# We also provide a button to run the registration
run_pressed = st.button("Run Registration")

if run_pressed:
    # Check if we have a valid reference file and at least one test file
    if reference_file is None:
        st.error("No reference file uploaded!")
    elif len(test_files) == 0:
        st.error("No test files uploaded!")
    else:
        # Process reference first
        with st.spinner("Loading & processing reference STL..."):
            temp_dir = tempfile.TemporaryDirectory()
            ref_path = os.path.join(temp_dir.name, "reference.stl")
            with open(ref_path, "wb") as f:
                f.write(reference_file.getbuffer())
            
            ref_mesh = load_mesh(ref_path)
            ref_pcd = sample_point_cloud(ref_mesh, num_points, nb_neighbors, std_ratio)
        
        # Optionally show raw reference points
        if show_ref_raw:
            st.markdown("### Reference (Raw) Point Cloud")
            raw_fig = plot_multiple_point_clouds(
                [(np.asarray(ref_pcd.points), "#00CC96", "Reference Raw")],
                point_size
            )
            st.plotly_chart(raw_fig, use_container_width=True)
        
        # For each test file, do the alignment pipeline
        for test_file in test_files:
            with st.spinner(f"Processing test file: {test_file.name}"):
                test_path = os.path.join(temp_dir.name, f"test_{test_file.name}")
                with open(test_path, "wb") as t:
                    t.write(test_file.getbuffer())
                
                # Load and sample test
                test_mesh = load_mesh(test_path)
                test_pcd = sample_point_cloud(test_mesh, num_points, nb_neighbors, std_ratio)
                
                # Show test raw
                if show_test_raw:
                    st.markdown(f"### Raw Test Point Cloud: {test_file.name}")
                    fig_test_raw = plot_multiple_point_clouds(
                        [(np.asarray(test_pcd.points), "#EF553B", f"Test Raw: {test_file.name}")],
                        point_size
                    )
                    st.plotly_chart(fig_test_raw, use_container_width=True)
                
                # Global registration (optional)
                transformation_init = np.eye(4)
                if use_global_registration:
                    # Downsample & compute FPFH
                    ref_down, ref_fpfh = prepare_for_global_registration(ref_pcd, voxel_size_global)
                    test_down, test_fpfh = prepare_for_global_registration(test_pcd, voxel_size_global)
                    
                    ransac_result = global_registration_ransac(
                        test_down, ref_down, test_fpfh, ref_fpfh, voxel_size_global
                    )
                    transformation_init = ransac_result.transformation
                
                # ICP refinement
                icp_result = refine_registration_icp(
                    test_pcd,
                    ref_pcd,
                    transformation_init,
                    icp_threshold,
                    icp_max_iter
                )
                
                # Transform test pcd
                test_aligned = transform_point_cloud(test_pcd, icp_result.transformation)
                
                # Compute deviations
                distances = compute_deviations(test_aligned, ref_pcd)
                
                # Show results in an expander for each test file
                with st.expander(f"Results for {test_file.name}", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Registration Metrics (ICP)**")
                        df_metrics = pd.DataFrame({
                            "Metric": ["Fitness", "RMSE"],
                            "Value": [icp_result.fitness, icp_result.inlier_rmse]
                        })
                        st.dataframe(df_metrics.set_index("Metric"), use_container_width=True)
                    
                    with col2:
                        st.write("**Deviation Statistics**")
                        st.metric("Max Deviation (mm)", f"{distances.max():.3f}")
                        st.metric("Avg Deviation (mm)", f"{distances.mean():.3f}")
                        st.metric("Std Dev (mm)", f"{distances.std():.3f}")
                        
                        st.write("**Deviation Distribution**:")
                        dist_df = pd.DataFrame({"Deviation (mm)": distances})
                        st.bar_chart(dist_df, y="Deviation (mm)")
                    
                    # 3D Heatmap
                    st.write("**Deviation Map**")
                    aligned_points = np.asarray(test_aligned.points)
                    fig_heatmap = plot_point_cloud_heatmap(
                        aligned_points,
                        distances,
                        point_size,
                        color_scale,
                        title=f"Aligned Deviation: {test_file.name}"
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    # Mean position offset
                    offset = aligned_points.mean(axis=0) - np.asarray(ref_pcd.points).mean(axis=0)
                    st.markdown(
                        f"**Mean Position Offset** (mm): "
                        f"X={offset[0]:.4f}, Y={offset[1]:.4f}, Z={offset[2]:.4f}"
                    )
                    
                    # Download CSV for this test
                    download_df = pd.DataFrame(
                        np.hstack((aligned_points, distances.reshape(-1,1))),
                        columns=["X", "Y", "Z", "Deviation"]
                    )
                    st.download_button(
                        label="Download Aligned Deviation Data (CSV)",
                        data=download_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"aligned_deviation_{test_file.name}.csv"
                    )
                
        st.success("All test files processed!")
else:
    st.info("Upload a reference STL, test STL(s), and click 'Run Registration' to begin.")
