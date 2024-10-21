import numpy as np
import open3d as o3d
import os
from depth_file_generator import ViewFile, translate, scale, rotate
from depth.utils import *
# from objectsimilaritymetrics.loss import metrics
# from objectsimilaritymetrics.compare import extract_data, print_metrics


# Point cloud sampling function
def sample_points_from_mesh(mesh, num_points=10000):
    """
    Sample points from the surface of the mesh.
    
    Args:
    - mesh: open3d TriangleMesh object.
    - num_points: Number of points to sample.

    Returns:
    - numpy.ndarray: Sampled point cloud (num_points, 3).
    """
    sampled_points = mesh.sample_points_uniformly(number_of_points=num_points)
    return np.asarray(sampled_points.points)



def chamfer_hausdorff_distance(pcd1, pcd2):
    """
    Compute both Chamfer and Hausdorff Distances between two point clouds.

    Parameters:
    - pcd1: open3d.geometry.PointCloud, first point cloud (ground truth).
    - pcd2: open3d.geometry.PointCloud, second point cloud (predicted points).

    Returns:
    - chamfer_dist: Chamfer Distance between the two point clouds.
    - hausdorff_dist: Hausdorff Distance between the two point clouds.
    """
    # Convert point clouds to NumPy arrays
    pcd1_points = np.asarray(pcd1.points)
    pcd2_points = np.asarray(pcd2.points)

    # Build KDTree for efficient nearest neighbor search
    pcd1_kd_tree = o3d.geometry.KDTreeFlann(pcd1)
    pcd2_kd_tree = o3d.geometry.KDTreeFlann(pcd2)

    # Chamfer Distance calculations
    dist_pcd1_to_pcd2 = 0.0
    dist_pcd2_to_pcd1 = 0.0
    hausdorff_pcd1_to_pcd2 = 0.0
    hausdorff_pcd2_to_pcd1 = 0.0

    # Calculate Chamfer and Hausdorff distances from pcd1 to pcd2
    for point in pcd1_points:
        [_, idx, _] = pcd2_kd_tree.search_knn_vector_3d(point, 1)
        nearest_point = pcd2_points[idx[0]]
        distance = np.linalg.norm(point - nearest_point)
        dist_pcd1_to_pcd2 += distance
        hausdorff_pcd1_to_pcd2 = max(hausdorff_pcd1_to_pcd2, distance)

    # Calculate Chamfer and Hausdorff distances from pcd2 to pcd1
    for point in pcd2_points:
        [_, idx, _] = pcd1_kd_tree.search_knn_vector_3d(point, 1)
        nearest_point = pcd1_points[idx[0]]
        distance = np.linalg.norm(point - nearest_point)
        dist_pcd2_to_pcd1 += distance
        hausdorff_pcd2_to_pcd1 = max(hausdorff_pcd2_to_pcd1, distance)

    # Average distances to get Chamfer Distance
    chamfer_dist = (dist_pcd1_to_pcd2 + dist_pcd2_to_pcd1) / (len(pcd1_points) + len(pcd2_points))

    # Hausdorff Distance is the maximum of the two directional Hausdorff distances
    hausdorff_dist = max(hausdorff_pcd1_to_pcd2, hausdorff_pcd2_to_pcd1)
    
    return chamfer_dist, hausdorff_dist

def extract_data(obj):
    data = {}
    if isinstance(obj, o3d.geometry.TriangleMesh):
        data['points'] = np.array(obj.vertices)
    elif isinstance(obj, o3d.geometry.PointCloud):
        data['points'] = np.array(obj.points)

    return data


def print_metrics(data):
    for k, v in data.items():
        print(f'{k}: {v}')

def filter_closest_points(sampled_gt, depthdeepsdf_pcd):
    """
    Filters the depthdeepsdf_pcd point cloud to return only the points
    that are closest to each point in the sampled_gt point cloud.
    
    Args:
    - sampled_gt: open3d.geometry.PointCloud, ground truth point cloud.
    - depthdeepsdf_pcd: open3d.geometry.PointCloud, predicted point cloud.

    Returns:
    - filtered_pcd: open3d.geometry.PointCloud, point cloud with the closest points.
    """
    # Convert point clouds to NumPy arrays for processing
    gt_points = np.asarray(sampled_gt.points)
    pred_points = np.asarray(depthdeepsdf_pcd.points)

    # Build KDTree for depthdeepsdf_pcd to find nearest neighbors
    pred_kd_tree = o3d.geometry.KDTreeFlann(depthdeepsdf_pcd)

    # Collect the closest points from depthdeepsdf_pcd to sampled_gt
    closest_points = []

    for gt_point in gt_points:
        # Search for the nearest neighbor in depthdeepsdf_pcd for each point in sampled_gt
        [_, idx, _] = pred_kd_tree.search_knn_vector_3d(gt_point, 1)
        closest_point = pred_points[idx[0]]  # Nearest point
        closest_points.append(closest_point)

    # Create a new point cloud with the closest points
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(np.array(closest_points))

    return filtered_pcd


if __name__ == '__main__':
    vis = o3d.visualization.Visualizer()
    k = 200
    rejection_angle = 25
    experiment_name = 'new_exp_10'
    categories = ['mug', 'bowl', 'bottle']
    text_to_file = []
    mean_result = []
    for category in categories:
        exp = "new_exp_10"
        results_path = f'examples/{exp}/Reconstructions/600/Meshes/dataset_YCB_test/test_{exp}_{category}_old'
        names_txt = [name for name in os.listdir(results_path) if name.endswith('.npz')]

        iterations = 0
        chamfer = 0
        hausdorff = 0
        for name in names_txt:
            print(name)
            object_name = name.split('_')[0]
            view = int(name.split('view')[1][0])

            DEEPSDF_RESULTS = f'{category}_classic_deepsdf/1eaf8db2dd2b710c7d5b1b70ae595e60_5_a25_view4_k150_inp_test.ply'
            GT_PATH = os.path.join(f'ShapeNetCore/{category}', name.split('_')[0], 'models/model_normalized.obj')
            SOURCE_PATH = f"data_YCB/SdfSamples/dataset_YCB_train/train_{exp}_{category}/{name.replace(f'_k{k}_inp_test.npz', '.json')}"
            TEST_QUERY_PATH = f"data_YCB/SdfSamples/dataset_YCB_test/test_{exp}_{category}_old/{name.replace('.npz', '_query.json')}" #_k150_inp_train.json'
            RESULTS_PATH = os.path.join(results_path, name)
            TRAINING_PATH = f"data_YCB/SdfSamples/dataset_YCB_train/train_{exp}_{category}/{name.replace(f'test.npz', 'train.json')}"
            TEST_PATH = f"data_YCB/SdfSamples/dataset_YCB_test/test_{exp}_{category}_old/{name.replace('.npz', '.json')}"
            DEPTH_RESULT_PATH = TEST_QUERY_PATH.replace('_query.json', '_th.pcd')

            MESH_SOURCE_PATH = os.path.join(f"examples/{experiment_name}/data/{category}", object_name + f"_5.json")
            MESH_PATH =  os.path.join(f"ShapeNetCore/{category}", object_name, 'models/model_normalized.obj')

            input_mesh_file = ViewFile(object_name)
            input_mesh_file.load(MESH_SOURCE_PATH)
            input_mesh = load_file(MESH_PATH)
            input_mesh = rotate(input_mesh, np.array([90, 0, 0]))
            centered_mesh = translate(input_mesh, input_mesh_file.s_o_transformation[:3])
            scaled_mesh, _ = scale(centered_mesh, input_mesh_file.scale)

            frame = input_mesh_file.frames[view]
            scaled_mesh = translate(scaled_mesh, frame[:3])
            scaled_mesh = rotate(scaled_mesh, frame[3:])
            scaled_mesh = rotate(scaled_mesh, [0, 0, -90])
            scaled_mesh = rotate(scaled_mesh, np.array([135, 0, 0]))
            scaled_mesh = translate(scaled_mesh, [0, 0, 1.5])

            points = sample_points_from_mesh(scaled_mesh, 15_000)
            sampled_gt = o3d.geometry.PointCloud()  # create point cloud object
            sampled_gt.points = o3d.utility.Vector3dVector(points)
       
            depthdeepsdf_pcd = o3d.io.read_point_cloud(DEPTH_RESULT_PATH)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            o3d.visualization.draw_geometries([sampled_gt, origin], mesh_show_back_face=True, window_name='orginal point cloud')
            dest_filepath = os.path.join('DepthDeepSDFfillinggaps', category, object_name + f'_view{view}_gt.pcd')
            print("SAVE FILE: ", dest_filepath)
            o3d.io.write_point_cloud(dest_filepath, sampled_gt)
            continue
            print(RESULTS_PATH.split('/')[-1])
            
            iterations += 1
            input_data = extract_data(depthdeepsdf_pcd)
            reference_data = extract_data(sampled_gt)

            losses = {}

            # for m in ['chamfer', 'hausdorff']:
            #     if metrics.exists(m):
            #         losses[m] = metrics.calculate(input_data, reference_data, m)

            # Filter the depthdeepsdf_pcd to get the points closest to the sampled_gt
            # filtered_pcd = filter_closest_points(sampled_gt, depthdeepsdf_pcd)

            # Use the chamfer_hausdorff_distance function to calculate the distances
            chamfer_dist, hausdorff_dist = chamfer_hausdorff_distance(depthdeepsdf_pcd, sampled_gt)
            losses['chamfer'] = chamfer_dist
            losses['hausdorff'] = hausdorff_dist

            print_metrics(losses)
            comparison_result = f"object: {object_name}, view: {view}, category: {category.title()}, chamfer: {losses['chamfer']}, hausdorff: {losses['hausdorff']}"
            text_to_file.append(comparison_result)
            chamfer += losses['chamfer']
            hausdorff += losses['hausdorff']
            o3d.visualization.draw_geometries([sampled_gt, origin], mesh_show_back_face=True, window_name='orginal point cloud')
        continue
        mean_chamfer = chamfer/iterations
        mean_hausdorff = hausdorff/iterations
        mean_values = f"for class {category} mean chamfer: {mean_chamfer}, mean hausdorff: {mean_hausdorff}"
        mean_result.append(mean_values)
    exit(777)
    print(mean_result)
    with open('comparison_depthdeepsdf2.txt', 'w') as f:
        for line in text_to_file:
            f.write(f"{line}\n")
        for line in mean_result:
            f.write(f"{line}\n")

    # load reconstructed mesh from deepsdf
    # load reconstructed mesh from depthdeepsdf
    # check where reconstructions are placed (deepsdf probably needs to be moved)
    # sample gt and deepsdf