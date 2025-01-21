import streamlit as st
import open3d as o3d
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tempfile
import os

# ----------------------------------------------
# Streamlit Page Configuration
# ----------------------------------------------
st.set_page_config(
    page_title="STL Deviation Analyzer (Improved)",
    layout="wide"
)
st.title("🦷 Dental STL Deviation Analyzer (Improved)")

# ----------------------------------------------
# Utility Functions
# ----------------------------------------------
def load_mesh(stl_path: str) -> o3d.geometry.TriangleMesh:
    """
    Load a mesh from STL file and perform basic cleaning.
    """
    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    return mesh

def sample_point_cloud_from_mesh(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int = 3000,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0
) -> o3d.geometry.PointCloud:
    """
    Convert mesh to point cloud via uniform sampling, then remove outliers.
    """
    # Sample points
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    
    # Remove outliers
    cl, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    return cl

def prepare_for_global_registration(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float
):
    """
    Downsample the point cloud, estimate normals, and compute FPFH features 
    to prepare for global RANSAC registration.
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=2 * voxel_size, max_nn=30
        )
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel_size, max_nn=100)
    )
    return pcd_down, fpfh

def global_registration_ransac(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    source_fpfh,
    target_fpfh,
    voxel_size: float
) -> o3d.pipelines.registration.RegistrationResult:
    """
    Perform RANSAC-based global registration to get coarse alignment.
    """
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, 
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result

def refine_registration_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    voxel_size: float,
    icp_threshold: float,
    max_iterations: int
) -> o3d.pipelines.registration.RegistrationResult:
    """
    Refine an initial transform using point-to-point ICP.
    """
    radius_normal = voxel_size * 2
    source.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    
    result_icp = o3d.pipelines.registration.registration_icp(
        source, 
        target, 
        icp_threshold, 
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )
    return result_icp

def transform_mesh(mesh: o3d.geometry.TriangleMesh, transform: np.ndarray) -> o3d.geometry.TriangleMesh:
    """
    Apply a 4x4 transformation matrix to a mesh.
    """
    mesh_copy = o3d.geometry.TriangleMesh(mesh)
    mesh_copy.transform(transform)
    return mesh_copy

def compute_signed_distance(
    source_mesh: o3d.geometry.TriangleMesh, 
    target_mesh: o3d.geometry.TriangleMesh
) -> np.ndarray:
    """
    Compute approximate signed distances from source mesh vertices to target mesh surface.
    
    In Open3D, we can create a signed distance field using a VoxelGrid approximation.
    The sign is determined by whether the point is inside or outside the target mesh.
    
    This method is approximate and works best for watertight meshes. 
    If your meshes have holes, results may be unreliable.
    """
    # Convert target mesh to a distance field
    target_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
        target_mesh, 
        voxel_size=0.5  # Adjust voxel size based on the scale of your model
    )
    
    # For each vertex in source, compute nearest voxel distance
    source_vertices = np.asarray(source_mesh.vertices)
    distances = []
    for v in source_vertices:
        # Evaluate distance within the voxel grid
        dist = target_grid.get_distance(v)
        # Note: If 'dist' is negative, the point is inside the surface (approx).
        distances.append(dist)
    return np.array(distances)

def create_3d_scatter_plot(
    xyz_points: np.ndarray, 
    values: np.ndarray, 
    title: str = "Deviation Map",
    color_scale: str = "Hot", 
    point_size: int = 3,
    value_range: tuple = None
) -> go.Figure:
    """
    Create a Plotly 3D scatter plot with color-coded points.
    
    :param xyz_points: Nx3 array of coordinates.
    :param values: Nx1 array of scalar values.
    :param title: Figure title.
    :param color_scale: Name of Plotly color scale.
    :param point_size: Size of the marker points.
    :param value_range: (min, max) range for color scaling. If None, auto-scales.
    """
    colorbar_dict = dict(title='Deviation (mm)')
    
    marker_dict = dict(
        size=point_size,
        color=values,
        colorscale=color_scale,
        opacity=0.8,
        colorbar=colorbar_dict
    )
    
    if value_range is not None:
        marker_dict['cmin'] = value_range[0]
        marker_dict['cmax'] = value_range[1]
    
    scatter = go.Scatter3d(
        x=xyz_points[:,0],
        y=xyz_points[:,1],
        z=xyz_points[:,2],
        mode='markers',
        marker=marker_dict
    )
    
    fig = go.Figure(data=[scatter])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)"
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

def compute_surface_area_and_volume(mesh: o3d.geometry.TriangleMesh):
    """
    Compute the surface area and volume of a closed mesh.
    If the mesh is not closed, volume may be 0 or inaccurate.
    """
    # Check if mesh is watertight for volume calculation
    if mesh.is_watertight():
        volume = mesh.get_volume()
    else:
        volume = float('nan')  # Not a closed mesh => volume is undefined
    
    area = mesh.get_surface_area()
    return area, volume

# ----------------------------------------------
# Sidebar parameters
# ----------------------------------------------
st.sidebar.header("Processing Parameters")

# Sampling
num_points = st.sidebar.slider("Points to Sample", 500, 30000, 3000, step=500)
nb_neighbors = st.sidebar.slider("Outlier Removal (Neighbors)", 5, 50, 20, step=5)
std_ratio = st.sidebar.slider("Outlier Removal (Std Ratio)", 1.0, 5.0, 2.0, step=0.5)

# Global registration
use_global_registration = st.sidebar.checkbox("Use Global Registration (RANSAC)?", value=True)
voxel_size_ransac = st.sidebar.slider("RANSAC Voxel Size", 0.1, 10.0, 1.0, step=0.1)

# ICP refinement
icp_threshold = st.sidebar.slider("ICP Threshold (mm)", 0.1, 5.0, 1.5, step=0.1)
icp_max_iterations = st.sidebar.slider("ICP Max Iterations", 50, 2000, 500, step=50)

# Signed distance
use_signed_distance = st.sidebar.checkbox("Compute Signed Distance?", value=False)

# Visualization
color_scale = st.sidebar.selectbox("Color Scale", ["Hot", "Viridis", "Rainbow", "Blues", "Reds"])
point_size = st.sidebar.slider("3D Plot Point Size", 1, 10, 3)
custom_range = st.sidebar.checkbox("Set Custom Color Range?", value=False)
if custom_range:
    min_val = st.sidebar.number_input("Minimum Value", value=-2.0)
    max_val = st.sidebar.number_input("Maximum Value", value=2.0)
    value_range = (min_val, max_val)
else:
    value_range = None

# ----------------------------------------------
# File Uploader
# ----------------------------------------------
uploaded_files = st.file_uploader(
    "Upload exactly two STL files: [Reference (Ideal) first, Test (Comparison) second]",
    type=["stl"],
    accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) != 2:
        st.error("Please upload exactly TWO STL files.")
    else:
        with st.spinner("Processing..."):
            temp_dir = tempfile.TemporaryDirectory()
            file_paths = []
            
            for i, f in enumerate(uploaded_files):
                file_path = os.path.join(temp_dir.name, f"mesh_{i}.stl")
                with open(file_path, "wb") as h:
                    h.write(f.getbuffer())
                file_paths.append(file_path)
            
            # Load & preprocess
            ref_mesh = load_mesh(file_paths[0])
            test_mesh = load_mesh(file_paths[1])
            
            ref_pcd = sample_point_cloud_from_mesh(
                ref_mesh,
                num_points=num_points,
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
            test_pcd = sample_point_cloud_from_mesh(
                test_mesh,
                num_points=num_points,
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
            
            # Surface area & volume for reference
            ref_area, ref_vol = compute_surface_area_and_volume(ref_mesh)
            test_area, test_vol = compute_surface_area_and_volume(test_mesh)
            
            # Optional: Global Registration (RANSAC)
            transform_init = np.eye(4)
            if use_global_registration:
                ref_down, ref_fpfh = prepare_for_global_registration(ref_pcd, voxel_size_ransac)
                test_down, test_fpfh = prepare_for_global_registration(test_pcd, voxel_size_ransac)
                
                ransac_result = global_registration_ransac(
                    test_down, 
                    ref_down, 
                    test_fpfh, 
                    ref_fpfh, 
                    voxel_size_ransac
                )
                transform_init = ransac_result.transformation
            
            # Refine using ICP
            result_icp = refine_registration_icp(
                test_pcd,
                ref_pcd,
                transform_init,
                voxel_size_ransac,
                icp_threshold,
                icp_max_iterations
            )
            
            # Apply final transform to test mesh
            final_transform = result_icp.transformation
            test_mesh_aligned = transform_mesh(test_mesh, final_transform)
            
            # Compute surface area & volume for aligned test mesh (should be the same, but verifying transform)
            test_area_aligned, test_vol_aligned = compute_surface_area_and_volume(test_mesh_aligned)
            
            # Signed or unsigned distance
            if use_signed_distance:
                st.subheader("Signed Distance Computation (Approx)")
                # Requires the mesh to be somewhat watertight for correct sign
                distances = compute_signed_distance(test_mesh_aligned, ref_mesh)
            else:
                st.subheader("Unsigned Distance Computation")
                # Convert the aligned mesh to pcd for measuring distance
                test_aligned_pcd = test_mesh_aligned.sample_points_uniformly(number_of_points=num_points)
                distances = np.asarray(test_aligned_pcd.compute_point_cloud_distance(ref_pcd))
            
            # Prepare to plot either the aligned mesh vertices or the aligned point cloud
            if use_signed_distance:
                # Plot the test mesh's vertices
                aligned_xyz = np.asarray(test_mesh_aligned.vertices)
            else:
                # Plot the test point cloud
                aligned_xyz = np.asarray(test_aligned_pcd.points)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**ICP Results**")
                icp_data = pd.DataFrame({
                    "Metric": ["Fitness", "RMSE"],
                    "Value": [result_icp.fitness, result_icp.inlier_rmse]
                })
                st.dataframe(icp_data.set_index("Metric"))
                
                # Export transformation matrix
                st.download_button(
                    label="Download Transform Matrix (CSV)",
                    data=pd.DataFrame(final_transform).to_csv(index=False).encode("utf-8"),
                    file_name="transform_matrix.csv"
                )
            
            with col2:
                st.markdown("**Reference Mesh**")
                st.write(f"Surface Area: {ref_area:.2f} mm²")
                st.write(f"Volume: {ref_vol:.2f} mm³ (NaN if not watertight)")
            
            with col3:
                st.markdown("**Test Mesh (Aligned)**")
                st.write(f"Surface Area: {test_area_aligned:.2f} mm²")
                st.write(f"Volume: {test_vol_aligned:.2f} mm³ (NaN if not watertight)")
            
            # Compute some stats on distances
            with st.expander("Deviation Statistics"):
                abs_distances = np.abs(distances)
                st.write(f"**Minimum Deviation:** {abs_distances.min():.3f} mm")
                st.write(f"**Mean Deviation:** {abs_distances.mean():.3f} mm")
                st.write(f"**Maximum Deviation:** {abs_distances.max():.3f} mm")
                st.write(f"**Std Dev:** {abs_distances.std():.3f} mm")
                
                st.bar_chart(pd.DataFrame(abs_distances, columns=["Deviation (mm)"]))
            
            # 3D Visualization
            st.subheader("3D Deviation Map")
            fig = create_3d_scatter_plot(
                aligned_xyz, 
                distances, 
                title="Deviation (Signed)" if use_signed_distance else "Deviation (Unsigned)",
                color_scale=color_scale,
                point_size=point_size,
                value_range=value_range
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Download aligned coordinates & distances
            output_df = pd.DataFrame(
                np.hstack((aligned_xyz, distances.reshape(-1,1))),
                columns=["X", "Y", "Z", "Distance"]
            )
            st.download_button(
                "Download Deviation Data (CSV)",
                output_df.to_csv(index=False).encode("utf-8"),
                "deviation_data.csv"
            )
else:
    st.info("Please upload two STL files to begin.")
