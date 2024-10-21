import argparse
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import random
import json
from skimage import measure
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from depth.utils import *
from depth.camera import Camera
from depth_image_generator import DepthImageFile
from sklearn.preprocessing import StandardScaler
import alphashape
import torch
# import open3d.ml.torch as ml3d
import numpy as np
from collections import defaultdict

from depth_file_generator import ViewFile
from depth_image_generator import translate, scale, rotate
from depth_training_data_generator import TrainingFile

class File():
    def __init__(self, source_path, destination_dir):
        self.source_path = source_path
        self.destination_dir = destination_dir 
        self.name = self.get_name_()
        self.version = self.get_version_()
        self.o_c_transformation = np.zeros(6)
        self.pixels = []
        self.ds = 0

    def get_name_(self):
        tail = os.path.split(self.source_path)[1]
        return tail.split('.')[0]
    
    def get_version_(self):
        dir_files = os.listdir(self.destination_dir)
        file_number = len([x for x in dir_files if x.startswith(self.name) and x.endswith('.txt')]) + 1
        return file_number
    
    def get_camera_parameters(self, f, cx, cy):
        self.f = f
        self.cx = cx
        self.cy = cy

    def get_image_resolution(self, Ndx, Ndy):
        self.Ndx = Ndx
        self.Ndy = Ndy

    def get_outliar_factor(self, ds):
        self.ds = ds

    def get_bounding_box_coords(self, nx, ny, z):
        self.nx = nx
        self.ny = ny
        self.z = z

    def get_bounding_box_size(self, ndx, ndy, dz, dz2):
        self.ndx = ndx
        self.ndy = ndy
        self.dz = dz
        self.dz2 = dz2

    def save(self):
        with open(os.path.join(self.destination_dir, self.name + '_inp' +'.txt'), 'w') as f:
            for pixel in self.pixels:
                f.write(f"{' '.join(map(str, pixel))}\n")
        print("Saved:", os.path.join(self.destination_dir, self.name + '_inp' +'.txt'))
                # print(pixel)
                # for p in pixel:
                    # print(p)

class PointsFromDepth:
    def __init__(self, data_file, pixels) -> None:
        self.cx = data_file.cx
        self.cy = data_file.cy
        self.f = data_file.f
        self.image = np.vstack(pixels)

    def to_point_cloud(self, scaled_colors=False):
        print(np.min(self.image, axis=0), np.max(self.image, axis=0))
        # self.image = self.image[self.image[:, 3] <= 0.01]
        z = np.array(self.image[:, 2])
        x = (self.image[:, 0] - self.cx) * z / self.f  # y on image is x in real world
        y = (self.image[:, 1] - self.cy) * z / self.f  # x on image is y in real world

        self.points = np.column_stack((x, y, z))
        pcd = o3d.geometry.PointCloud()  # create point cloud object
        pcd.points = o3d.utility.Vector3dVector(self.points)  # set pcd_np as the point cloud points
        
        if scaled_colors:
            # Normalize the SDF values to the range [0, 1]
            sdf_min, sdf_max = self.image[:, 3].min(), self.image[:, 3].max()
            sdf_normalized = (self.image[:, 3] - sdf_min) / (sdf_max - sdf_min)
            
            # Use a colormap to map normalized SDF values to colors
            colormap = plt.get_cmap('viridis')
            colors = colormap(sdf_normalized)[:, :3]  # Extract RGB values from colormap

            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd
    
    def visualize_as_point_cloud(self, pixels, additional=[]):
        points = []
        for (x, y), depth_values in pixels.items():
            for z in depth_values:
                if z > 0:  # Ignore points with zero depth
                    X = (x - self.cx) * z / self.f
                    Y = (y - self.cy) * z / self.f
                    points.append([X, Y, z])
        
        if not points:
            print("No valid points to display.")
            return
        sdf = self.image[:, 3]

        # Normalize the SDF values to the range [0, 1]
        sdf_min, sdf_max = sdf.min(), sdf.max()
        sdf_normalized = (sdf - sdf_min) / (sdf_max - sdf_min)
        
        # Use a colormap to map normalized SDF values to colors
        colormap = plt.get_cmap('autumn')
        colors = colormap(sdf_normalized)[:, :3]  # Extract RGB values from colormap

        points = np.array(points)
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(points)
        self.point_cloud.colors = o3d.utility.Vector3dVector(colors)

        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        vis_list = [self.point_cloud, origin]
        vis_list.extend(additional)
        # o3d.visualization.draw_geometries(vis_list, window_name=self.__class__.__name__)


def load_querry_points(path):
    if path.endswith(".npz"):
        dict_data = np.load(path)
        pos_data = dict_data[dict_data.files[0]]
        data = np.concatenate([pos_data])
        return data
    
    return None

def load_depth_file(input_file):
    with open(input_file.source_path, "r") as file:
        input_file.o_c_transformation = np.array(file.readline().split(), dtype=np.float32)
        f, cx, cy = file.readline().split()
        input_file.get_camera_parameters(float(f), float(cx), float(cy))
        Ndx, Ndy = file.readline().split()
        input_file.get_image_resolution(int(Ndx), int(Ndy))
        input_file.ds = float(file.readline())
        nx, ny, z = file.readline().split()
        input_file.get_bounding_box_coords(int(nx), int(ny), float(z))
        ndx, ndy, dz, dz2 = file.readline().split()
        input_file.get_bounding_box_size(int(ndx), int(ndy), float(dz), float(dz2))
        pixels = file.readlines()
        input_file.pixels = [np.array(pixel.split(), dtype=np.float32) for pixel in pixels]

def generate_input_pcd(input_file):
    visible_depth_points = []

    for key, value in input_file.pixels.items():
        unique = np.unique(value[value!=0])
        if not unique.any():
            continue
        else:
            x, y = key
            z = unique[0]
            visible_depth_points.append(np.array([x, y, z]))
    depth_image = np.vstack(visible_depth_points)

    z = depth_image[:, 2]
    x = (input_file.cx - depth_image[:, 0]) * z / input_file.f  # y on image is x in real world
    y = (input_file.cy - depth_image[:, 1]) * z / input_file.f  # x on image is y in real world

    points = np.column_stack([x, y, z])
    color_array = np.zeros(points.shape)
    color_array[:, 0] = 1

    pcd = o3d.geometry.PointCloud()  # create point cloud object
    pcd.points = o3d.utility.Vector3dVector(points)  # set pcd_np as the point cloud points
    pcd.colors = o3d.utility.Vector3dVector(color_array)

    return pcd

def create_volumetric_grid(point_cloud, resolution):
    # Define the grid resolution
    x_min, x_max = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
    y_min, y_max = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
    z_min, z_max = np.min(point_cloud[:, 2]), np.max(point_cloud[:, 2])

    # Calculate the dimensions of the bounding box
    x_dim = x_max - x_min
    y_dim = y_max - y_min
    z_dim = z_max - z_min

    # Create the 3D grid
    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    z_grid = np.linspace(z_min, z_max, resolution)

    # Create a 3D volume initialized with high positive values (empty space)
    volume = np.full((resolution, resolution, resolution), np.inf)

    # Populate the grid with sdf values
    for point in point_cloud:
        x, y, z, sdf = point

        # Find the closest grid indices
        x_idx = np.searchsorted(x_grid, x) - 1
        y_idx = np.searchsorted(y_grid, y) - 1
        z_idx = np.searchsorted(z_grid, z) - 1

        # Assign the sdf value to the grid point
        volume[x_idx, y_idx, z_idx] = sdf

    # Calculate the correct spacing
    spacing = (x_dim / resolution, y_dim / resolution, z_dim / resolution)
    
    return volume, spacing

def apply_marching_cubes(volume, spacing):
    # Extract the isosurface using Marching Cubes
    verts, faces, normals, values = measure.marching_cubes(volume, level=0.0, spacing=spacing)
    
    return verts, faces, normals, values

def critical_points_point_cloud(object_points, h_begin=0.9, h_end=0.5, min_thickness=0.1):
    # sdf * -1 + 1 wartość progowa
    depth_image = object_points.image

    print(depth_image, depth_image.shape)

    grouped_points = {}
    for point in depth_image:
        key = f"{point[0]}, {point[1]}"  # współrzędne (x, y)
        grouped_points[key].append(point)

    critical_points = []
    # Znajdujemy punkty powierzchni
    for key in grouped_points:
        points = grouped_points[key]
        current_thickness = min_thickness
        i = 0
        j = 1
        missed_points = 0
        while j <= len(points)-1:
            if (h_end <= points[i][3] <= h_begin) and (h_end <= points[j][3] <= h_begin):
                j += 1
            else:
                i += 1
                j += 1

    result_depth_image = np.array(critical_points)
    return result_depth_image

def filter_point_cloud(pcd):
    points = torch.randn([20,3])
    queries = torch.randn([10,3])
    k = 8

    nsearch = o3d.ml.torch.layers.KNNSearch(return_distances=True)
    ans = nsearch(points, queries, k)
    # returns a tuple of neighbors_index, neighbors_row_splits, and neighbors_distance
    # Since there are more than k points and we do not ignore any points we can
    # reshape the output to [num_queries, k] with
    neighbors_index = ans.neighbors_index.reshape(10,k)
    neighbors_distance = ans.neighbors_distance.reshape(10,k)
    a = 1+2

def visualize_dictionary(depth_file, dictionary, add=[], training=False, window_name='window name'):
    points = []
    sdf_values = []
    for key, value in dictionary.items():
        u, v = map(int, key.split(','))
        for point in value:
            try:
                rd, dd, sdf = point
                z = depth_file.dz + rd + dd
                if z > 0:  # Ignore points with zero depth
                    x = (u - depth_file.cx) * z / depth_file.f
                    y = (v - depth_file.cy) * z / depth_file.f
                    points.append([x, y, z])
                    sdf_values.append(sdf)
            except:
                print(point)

    if not points:
        print("No valid points to display.")
        return

    points = np.array(points)
    sdf_values = np.array(sdf_values)

    # Apply an SDF threshold filter
    threshold = 1
    filtered_indices = np.where(sdf_values < threshold)[0]
    filtered_points = points[filtered_indices]
    filtered_sdf_values = sdf_values[filtered_indices]

    # plt.hist(filtered_sdf_values)
    # plt.show()

    # Normalize the SDF values to the range [0, 1]
    sdf_min, sdf_max = filtered_sdf_values.min(), filtered_sdf_values.max()
    sdf_normalized = (filtered_sdf_values - sdf_min) / (sdf_max - sdf_min)
    # plt.hist(sdf_normalized)
    # plt.show()
    
    # Use a colormap to map normalized SDF values to colors
    if training:
        colormap = plt.get_cmap('cool')    
    else:
        colormap = plt.get_cmap('viridis')
    colors = colormap(sdf_normalized)[:, :3]  # Extract RGB values from colormap

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    visualize_list = [point_cloud, origin]
    if add:
        visualize_list.extend(add)
    o3d.visualization.draw_geometries(visualize_list, window_name=window_name)
    return point_cloud

def scale_back(normalized_data):
    
    min_u = 534
    max_u = 711
    min_v = 232
    max_v = 419
    min_dd = 0.10000000000000009
    max_dd = 0.30145583152771005
    min_rd = -0.30101666450500497
    max_rd = 0.3999999999999999
    min_sdf = 0
    max_sdf = 0.2930116653442383

    u_normalized = normalized_data[:, 0]
    v_normalized = normalized_data[:, 1]
    dd_normalized = normalized_data[:, 2]
    rd_normalized = normalized_data[:, 3]
    sdf_normalized = normalized_data[:, 4]

    # Scale back each variable to its original range
    u_original = u_normalized * (max_u - min_u) + min_u
    v_original = v_normalized * (max_v - min_v) + min_v
    dd_original = dd_normalized * (max_dd - min_dd) + min_dd
    rd_original = rd_normalized * (max_rd - min_rd) + min_rd
    sdf_original = sdf_normalized * (max_sdf - min_sdf) + min_sdf

    # Stack the results back into a single array of shape (90000, 5)
    original_data = np.stack((u_original, v_original, dd_original, rd_original, sdf_original), axis=1)
    
    return original_data


def filter_points_by_neighbors(object_image, min_neighbors=3, fixed_distance=0.1, sdf_threshold=0.05):
    """
    Filters out points that don't have a certain amount of neighbors along the rays.

    Parameters:
    - object_image: List or ndarray of points where each point is [u, v, z, sdf].
    - min_neighbors: Minimum number of neighbors required along the ray to keep the points.
    - fixed_distance: The fixed distance between consecutive points along the ray (z-axis).

    Returns:
    - filtered_points: ndarray of filtered points that meet the neighbor threshold.
    """

    # Step 1: Group points by (u, v) coordinates
    uv_groups = defaultdict(list)
    for point in object_image:
        u, v, z, sdf = point
        uv_groups[(u, v)].append(point)

    # Step 2: Filter out points based on the number of neighbors along z-axis
    filtered_points = []
    z_dict = defaultdict(list)  # To store z-values for each (u, v)

    for (u, v), points in uv_groups.items():
        # Sort points along the z-axis (depth)
        points_sorted_by_z = sorted(points, key=lambda p: p[2])  # Sorting by z (depth)
        # Count neighbors along z-axis
        neighbors = []
        for i in range(len(points_sorted_by_z) - 2):
            current_point = points_sorted_by_z[i]
            next_point = points_sorted_by_z[i + 1]
            third_point = points_sorted_by_z[i + 2]
            z_diff = abs(current_point[2] - next_point[2]) + abs(next_point[2] - third_point[2])  # Difference in z (depth)

            if z_diff <= 2*fixed_distance:  # If the points are close enough along z
                neighbors.append(current_point)
                neighbors.append(next_point)
                neighbors.append(third_point)
        
        # Deduplicate neighbors
        neighbors = list({tuple(p): p for p in neighbors}.values())  # Remove duplicates
        
        # Check if the number of neighbors meets the threshold
        if len(neighbors) >= min_neighbors:
            filtered_points.extend(neighbors)
            # Add z-values to the dictionary
            z_dict[(u, v)] = [point[2] for point in neighbors]  # Store only the z-values

    return np.array(filtered_points), z_dict


if __name__ == '__main__':
    vis = o3d.visualization.Visualizer()
    k = 200
    rejection_angle = 25
    categories = ['mug', 'bowl', 'bottle']
    for category in categories:
        exp = "new_exp_10"
        results_path = f'examples/{exp}/Reconstructions/600/Meshes/dataset_YCB_test/test_{exp}_{category}_old'
        names_txt = [name for name in os.listdir(results_path) if name.endswith('.npz')]
        for name in names_txt:
            print(name)
            SOURCE_PATH = f"data_YCB/SdfSamples/dataset_YCB_train/train_{exp}_{category}/{name.replace(f'_k{k}_inp_test.npz', '.json')}"
            TEST_QUERY_PATH = f"data_YCB/SdfSamples/dataset_YCB_test/test_{exp}_{category}_old/{name.replace('.npz', '_query.json')}" #_k150_inp_train.json'
            RESULTS_PATH = os.path.join(results_path, name)
            TRAINING_PATH = f"data_YCB/SdfSamples/dataset_YCB_train/train_{exp}_{category}/{name.replace(f'test.npz', 'train.json')}"
            TEST_PATH = f"data_YCB/SdfSamples/dataset_YCB_test/test_{exp}_{category}_old/{name.replace('.npz', '.json')}"

            print(RESULTS_PATH.split('/')[-1])
            npz = load_querry_points(RESULTS_PATH)

            npz = scale_back(npz)
            print("NPZ SHAPE:", npz.shape)
            data_file = DepthImageFile(name.split("_")[0])
            data_file.load(SOURCE_PATH)
            # depth_pcd is point cloud that see camera   
            data_file.visualize_as_point_cloud()         
            # depth_pcd = generate_input_pcd(data_file)
            depth_pcd = data_file.point_cloud

            with open(TRAINING_PATH, 'r') as f:
                training_file = json.load(f)
            training_pcd = visualize_dictionary(data_file, training_file, add=[depth_pcd], training=True, window_name='training pcd')

            with open(TEST_PATH, 'r') as f:
                test_file = json.load(f)
            test_pcd = visualize_dictionary(data_file, test_file, add=[depth_pcd], window_name='test pcd')

            input_file = []
            with open(TEST_QUERY_PATH, 'r') as f:
                input_file = json.load(f)
            print(type(input_file), len(input_file))
            print(type(npz), npz.shape)
            miss_match = 0
            object_image = []
            depth_image = []
            visualize_dict = {}
            itr = 0
            halo = 0
            dupplicated_sdf = 0

            u = []
            v = []
            for key in input_file.keys():
                u.append(int(key.split(', ')[0]))
                v.append(int(key.split(', ')[1]))

            min_u = min(u)
            max_u = max(u)

            min_v = min(v)
            max_v = max(v)

            for key, value in input_file.items():
                pixel = []
                x, y = map(int, key.split(', '))
                visualize_dict[(x, y)] = []

                if len(value) == 1 and not np.any(np.isnan(value[0])):
                    halo += 1

                duplicate_counter = 0
                previous_sdf = 100
                for i, row in enumerate(value):
                    if np.any(np.isnan(row)):
                        break
                    else: #np.float32(row[0]) == npz[itr][0] and np.float32(row[1]) == npz[itr][1]:
                        z = npz[itr][2] + npz[itr][3] + data_file.dz
                        sdf = npz[itr][4]
                        if npz[itr][3] < 0:
                            previous_sdf = sdf
                            itr += 1
                            continue
                        itr += 1
                        if sdf == previous_sdf:
                            dupplicated_sdf += 1
                            previous_sdf = sdf
                            continue
                        visualize_dict[(x, y)].append(z)
                        object_image.append(np.array([x, y, z, sdf]))
                        previous_sdf = sdf
                    # else:
                    #     miss_match += 1

            print("MISS MATCH:", miss_match)
            print("HALO", halo)
            fixed_distance = 0.5/98
            threshold = 0.5/99
            filtered_points, z_dict = filter_points_by_neighbors(object_image, min_neighbors=3, fixed_distance=fixed_distance, sdf_threshold=threshold)
            object_points = PointsFromDepth(data_file, filtered_points)
            object_points.visualize_as_point_cloud(z_dict, [training_pcd])
            # plt.hist(object_points.image[:, 3], bins=50)
            # plt.show()
            print("object_points shape", object_points.image.shape)
            # continue
            # critical_points = critical_points_point_cloud(object_points, h_begin=0.00002, h_end=-0.00003, min_thickness=0.00004)
            MC = False
            resolution = 100  # Adjust the resolution as needed

            # orginal_pcd = object_points.to_point_cloud()
            orginal_pcd = object_points.point_cloud

            if MC:
                valid_points = np.column_stack((np.asarray(orginal_pcd.points), object_points.image[:, 3]))
                volume, spacing = create_volumetric_grid(valid_points, resolution)
                verts, faces, normals, values = apply_marching_cubes(volume, spacing)

                pcd = o3d.geometry.PointCloud()  # create point cloud object
                pcd.points = o3d.utility.Vector3dVector(verts)

            object_points.image = object_points.image[object_points.image[:, 3] >= -threshold]
            object_points.image = object_points.image[object_points.image[:, 3] <= threshold]  # zrobić jutro z prior knowledge albo reverse sampling
            object_pcd_th = object_points.to_point_cloud(True)
            # filter_point_cloud(object_pcd)

            if MC:
                valid_points_th = np.column_stack((np.asarray(object_pcd_th.points), object_points.image[:, 3]))
                volume_th, spacing_th = create_volumetric_grid(valid_points_th, resolution)
                verts_th, faces_th, normals, values = apply_marching_cubes(volume_th, spacing_th)

                pcd_th = o3d.geometry.PointCloud()  # create point cloud object
                pcd_th.points = o3d.utility.Vector3dVector(verts_th)

            # mesh = o3d.geometry.TriangleMesh()
            # mesh.vertices = object_pcd.points 
            #  o3d.utility.Vector3dVector(object_points)
            # mesh.triangles = o3d.utility.Vector3iVector(faces)

            # alpha = 10
            # alpha_shape = alphashape.alphashape(verts, alpha)
            # print(type(alpha_shape), alpha_shape, alpha)

            # alpha_verts = np.asarray(alpha_shape.vertices)
            # alpha_faces = np.asarray(alpha_shape.faces)

            if MC:
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(verts_th)
                mesh.triangles = o3d.utility.Vector3iVector(faces_th)

            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

            # o3d.visualization.draw_geometries([orginal_pcd, origin], mesh_show_back_face=True, window_name='orginal point cloud')
            # o3d.visualization.draw_geometries([object_pcd_th, origin], mesh_show_back_face=True, window_name='thresholded point cloud')

            if MC:
                # o3d.visualization.draw_geometries([pcd, origin], mesh_show_back_face=True, window_name='marching cubes point cloud')
                # o3d.visualization.draw_geometries([pcd_th, origin], mesh_show_back_face=True, window_name='thresholded marching cubes point cloud')

                print(1, verts_th.shape[0], faces_th.shape)
                print(2, np.asarray(mesh.vertices).shape[0], np.asarray(mesh.triangles).shape[0])

            mesh_pcd = o3d.geometry.TriangleMesh()
            mesh_pcd.vertices = o3d.utility.Vector3dVector(np.asarray(orginal_pcd.points))

            mesh_pcd_th = o3d.geometry.TriangleMesh()
            mesh_pcd_th.vertices = o3d.utility.Vector3dVector(np.asarray(object_pcd_th.points))
            save_filepath = os.path.join('DepthDeepSDFfillinggaps', category, 'reconstruction', name.replace('.npz', '_th'))
            # o3d.io.write_triangle_mesh(TEST_QUERY_PATH.replace('.json', '_orginal.ply'), mesh_pcd)
            o3d.io.write_triangle_mesh(save_filepath + '.ply', mesh_pcd_th)
            # o3d.io.write_point_cloud(TEST_QUERY_PATH.replace('test_query.json', 'train.pcd'), training_pcd)
            # o3d.io.write_point_cloud(TEST_QUERY_PATH.replace('test_query.json', 'test.pcd'), test_pcd)

            # o3d.io.write_point_cloud(TEST_QUERY_PATH.replace('.json', '_orginal.pcd'), orginal_pcd)
            o3d.io.write_point_cloud(save_filepath + '.pcd', object_pcd_th)
            # o3d.io.write_point_cloud(TEST_QUERY_PATH.replace('_query.json', 'mc_1000.pcd'), pcd)
            # o3d.io.write_point_cloud(TEST_QUERY_PATH.replace('_query.json', 'th_neg003_pos003_mc_1000.pcd'), pcd)
            print(f"\nSAVED: {TEST_QUERY_PATH}\n\n")
            # exit(777)

            # mesh2 = o3d.io.read_triangle_mesh(TEST_QUERY_PATH.replace('_query.json', '_mesh.ply'))
            # print(3, np.asarray(mesh2.vertices).shape[0], np.asarray(mesh2.triangles).shape[0])

            # <class 'shapely.geometry.polygon.Polygon'> POLYGON Z ((0.1032942517043708 0 0.1386363677680492, 0.0888330564657589 0.0049663662297643 0.1431818224489689, 0.0785036312953218 0.0099327324595286 0.1477272771298885, 0.0702400911589722 0.0148990986892929 0.154545459151268, 0.0599106659885351 0.0248318311488216 0.1636363685131073, 0 0.1167096063994613 0.1295454584062099, 0 0.1738228180417508 0.0931818209588528, 0.0061976551022622 0.1812723673863973 0.0886363662779331, 0.0661083210907973 0.2259696634542761 0.1045454576611519, 0.0867671714316715 0.2383855790286869 0.1227272763848305, 0.1094919068066331 0.2458351283733334 0.1386363677680492, 0.1404801823179443 0.2458351283733334 0.1386363677680492, 0.1694025727951682 0.2359023959138047 0.1090909123420715, 0.1817978829996927 0.2284528465691583 0.1136363670229912, 0.1941931932042172 0.2135537478798653 0.0840909115970135, 0.1962590782383046 0.2110705647649832 0.0886363662779331, 0.2024567333405668 0.1961714660756902 0.0886363662779331, 0.2045226183746542 0.1390582544334007 0.1386363677680492, 0.2003908483064794 0.0471804791827609 0.1795454598963261, 0.1962590782383046 0.0347645636083502 0.1681818231940269, 0.1859296530678675 0.0198654649190572 0.1590909138321876, 0.173534342863343 0.0099327324595286 0.1477272771298885, 0.1611390326588185 0.0049663662297643 0.1431818224489689, 0.1466778374202066 0 0.1386363677680492, 0.1032942517043708 0 0.1386363677680492)) 0.0

            # print("Critical points shape:", critical_points.shape)
            # object_points.image = critical_points
            # object_pcd = object_points.to_point_cloud()

            # # Optionally, convert to Open3D point cloud for visualization
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(critical_points)
            # o3d.visualization.draw_geometries([object_pcd], mesh_show_back_face=True)
            # continue


            # # Get only rows where sdf value > 0

            # Create volumetric grid and populate it with sdf values
  # set pcd_np as the point cloud points
            # if category == 'bottle':
            #     mesh, some_list = pcd.compute_convex_hull()
            # o3d.visualization.draw_geometries([object_pcd, pcd], mesh_show_back_face=True)
            # o3d.io.write_triangle_mesh(TEST_QUERY_PATH.replace('_query.json', '_mesh_convex_hull.ply'), mesh)
            # o3d.io.write_point_cloud(TEST_QUERY_PATH.replace('_query.json', '_verts_from_mc.pcd'), pcd)

            # # Apply Marching Cubes
            # mc_params = {
            #     'level': [-.000001, 0.0, 0.000001],
            #     'allow_degenerate': [True, False],
            #     'gradient_direction': ['descent', 'ascent'],
            #     'step_size': [1],
            #     'method': ['lewiner', 'lorensen']
            # }

            # for level in mc_params['level']:
            #     for allow_degenerate in mc_params['allow_degenerate']:
            #         for gradient_direction in mc_params['gradient_direction']: 
            #             for step_size in mc_params['step_size']:
            #                 for method in mc_params['method']:
            #                     # verts, faces, normals, values = apply_marching_cubes(volume, spacing)
            #                     verts, faces, normals, values = measure.marching_cubes(
            #                         volume=volume,
            #                         level=level,
            #                         spacing=spacing,
            #                         gradient_direction=gradient_direction,
            #                         step_size=step_size,
            #                         allow_degenerate=allow_degenerate,
            #                         method=method
            #                     )
            #                     print(level, allow_degenerate, gradient_direction, step_size, method, verts.shape[0], faces.shape[0])

            #                     # Visualize the resulting mesh using open3d
            #                     mesh = o3d.geometry.TriangleMesh()
            #                     mesh.vertices = o3d.utility.Vector3dVector(verts)
            #                     mesh.triangles = o3d.utility.Vector3iVector(faces)
            #                     # mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

            #                     # alpha_shape = alphashape.alphashape(verts, 0.5)
            #                     # print(type(alpha_shape), alpha_shape)
            #                     # alpha_verts = np.asarray(alpha_shape.vertices)
            #                     # alpha_faces = np.asarray(alpha_shape.faces)
            #                     # mesh = o3d.geometry.TriangleMesh()
            #                     # mesh.vertices = o3d.utility.Vector3dVector(alpha_verts)
            #                     # mesh.triangles = o3d.utility.Vector3iVector(alpha_faces)

            #                     pcd = o3d.geometry.PointCloud()  # create point cloud object
            #                     pcd.points = o3d.utility.Vector3dVector(verts)  # set pcd_np as the point cloud points
            #                     # if category == 'bottle':
            #                     #     mesh, some_list = pcd.compute_convex_hull()
            #                     o3d.visualization.draw_geometries([mesh, object_pcd, pcd], mesh_show_back_face=True)
            #                     # o3d.io.write_triangle_mesh(TEST_QUERY_PATH.replace('_query.json', '_mesh_convex_hull.ply'), mesh)
            #                     # o3d.io.write_point_cloud(TEST_QUERY_PATH.replace('_query.json', '_verts_from_mc.pcd'), pcd)

            continue

            object_points.image = object_points.image[object_points.image[:, 3] < 0]
            object_points.image = object_points.image[object_points.image[:, 3] <= 0.01]
            object_pcd_th = object_points.to_point_cloud()
            object_sdf_array = object_points.image[:, 3]
            print("object_points shape", object_points.image.shape)
            # exit(888)
            # plt.hist(object_sdf_array)
            # plt.show()

            object_color_array = np.zeros((len(object_sdf_array), 3))

            min_sdf = np.min(object_sdf_array)
            mean_sdf = np.mean(object_sdf_array)
            max_sdf = np.max(object_sdf_array)
            normalized_sdf_array = (object_sdf_array - min_sdf) / (max_sdf - min_sdf)

            # # plt.hist(normalized_sdf_array, bins=100)
            # # plt.show()

            print(min_sdf, max_sdf)
            object_color_array[:, 2] = (1 - normalized_sdf_array) ** 2
            # object_color_array[normalized_sdf_array == 0, 1] = 1
            object_pcd_th.colors = o3d.utility.Vector3dVector(object_color_array)

            pcd_points_ndarray = np.concatenate((np.asarray(depth_pcd.points), np.asarray(object_pcd_th.points)), axis=0)
            pcd_colors_ndarray = np.concatenate((np.asarray(depth_pcd.colors), np.asarray(object_pcd_th.colors)), axis=0)
            pcd_th = o3d.geometry.PointCloud()  # create point cloud object
            pcd_th.points = o3d.utility.Vector3dVector(pcd_points_ndarray)  # set pcd_np as the point cloud points
            pcd_th.colors = o3d.utility.Vector3dVector(pcd_colors_ndarray)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

            # pcd_points_ndarray = np.concatenate((np.asarray(depth_pcd.points), np.asarray(object_pcd.points)), axis=0)
            # pcd_colors_ndarray = np.concatenate((np.asarray(depth_pcd.colors), np.asarray(object_pcd.colors)), axis=0)
            # pcd = o3d.geometry.PointCloud()  # create point cloud object
            # pcd.points = o3d.utility.Vector3dVector(np.asarray(object_pcd.points))  # set pcd_np as the point cloud points
            # pcd.colors = o3d.utility.Vector3dVector(np.asarray(object_pcd.colors))
            # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

            # vis.create_window()

            # vis.add_geometry(pcd)

            # opt = vis.get_render_option()

            # opt.background_color = np.asarray([0, 0, 0])

            # vis.run()

            # vis.destroy_window()
            # o3d.visualization.draw_geometries([pcd, origin])
            # o3d.io.write_point_cloud(TEST_QUERY_PATH.replace('_query.json', '.pcd'), pcd)
            # exit(777)        
            #nasycenie nieliniowe kolorów do wizualizacji
            # dane do json
            # odwrócić wartości sdf - na zewnątrz 0 wewnątrz dodatnie wartości
            # ASAP wysłać dane
            # badanie na 1 obiekcie, 50 widoków

            # zwiększyć rozdzielczość i zmniejszyć liczbę sampli
            # grid search parametrów do marching cubes
            
            # zrobić 5 kategorii
            # przygotować mesh'e z meshlaba ręcznie

            # dodać współrzędne u, v względem środka centroida

            # do poniedziałku 14.10
            # usunąć niepotrzebne rzeczy z dysku google,
            # zapisać aktualne wyniki,
            # dodać laptopy - nauczyć model w nocy z 4 kategoriami
            # filtrowanie w 3D z dostępnych w MeshLabie,
            # zrekonstruować meshe
            # porównać z DeepSDF

            # To remove outliers from a point cloud in MeshLab using a radius-based method, you can follow these steps:
            
            # Import the Point Cloud: Open MeshLab and import your point cloud file.
            # Select the Filter: Go to Filters > Cleaning and Repairing > Remove Isolated Pieces (wrt Diameter).
            # Set Parameters: In the dialog box, set the Max Diameter to define the radius within which points must have neighbors to be retained. Adjust the Min Number of Neighbors to specify the minimum number of points required within

            # spotkanie 16.10.2024
            # wygenerować chmurę punktów z meshy deepsdf za pomocą kamery
            # i odpalić algorytm do liczenia błędów
            # Wykorzystać przewagę tego że zachowujemy widoczne punkty
            # sprawdzić błąd dla części tylko widocznej i tylko niewidocznej
            # porównania:
            # 1. całościowe wynik pcd - mesh gt (shapenet) dla depthdeepsdf bierzemy najbliższy
            # 2. tylko to co jest widoczne - obliczyć najbliższy punkt wygenerowany na meshu (punkt z gt mesh z deepsdf)
            # 3. tylko to co jest niewidoczne - 
            # deepsdf tak samo jest obiczane dla części widocznej i niewidocznej, czyli punkt - mesh, punkt z gt do mesha
            # depthdeepsdf punkt-punkt
            
            # 1. filtr taki jaki mamy teraz
            # 2. filtr po promieniu żeby wywalić te outliery
            # 3. po progowaniu badamy monotoniczność funkcji sdf. Z tych co zostało sprawdzamy gdzie rosną a gdzie maleją
            