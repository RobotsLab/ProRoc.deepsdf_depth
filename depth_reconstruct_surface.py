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
from depth_image_generator import File as DepthFile
from sklearn.preprocessing import StandardScaler
import alphashape

from depth_file_generator import File as ViewsFile
from depth_image_generator import load_generator_file, translate, scale, rotate

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

    def to_point_cloud(self):
        print(np.min(self.image, axis=0), np.max(self.image, axis=0))
        # self.image = self.image[self.image[:, 3] <= 0.01]
        z = np.array(self.image[:, 2])
        x = (self.cx - self.image[:, 0]) * z / self.f  # y on image is x in real world
        y = (self.cy - self.image[:, 1]) * z / self.f  # x on image is y in real world

        self.points = np.column_stack((x, y, z))

        pcd = o3d.geometry.PointCloud()  # create point cloud object
        pcd.points = o3d.utility.Vector3dVector(self.points)  # set pcd_np as the point cloud points

        return pcd

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

    for i, pixel in enumerate(input_file.pixels):
        unique = np.unique(pixel[pixel!=0])
        x = (i % input_file.ndx) + input_file.nx
        y = (i // input_file.ndx) + input_file.ny
        if unique.any() and len(unique)%2 == 0:
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

if __name__ == '__main__':
    vis = o3d.visualization.Visualizer()
    k = 150
    rejection_angle = 25
    categories = ['mug', 'bottle','bowl', 'laptop']
    for category in categories:
        results_path = f'examples/new4/Reconstructions/1000/Meshes/dataset_YCB_test/test_new4_{category}'
        names_txt = [name for name in os.listdir(results_path) if name.endswith('.npz')]
        for name in names_txt:
            print(name)
            SOURCE_PATH = f"dataset_YCB_train/DepthDeepSDF/files/{category}/{name.replace('_k150_inp_test.npz', '.txt')}"
            TEST_QUERY_PATH = f"data_YCB/SdfSamples/dataset_YCB_test/test_new4_{category}/{name.replace('.npz', '_query.json')}" #_k150_inp_train.json'
            RESULTS_PATH = os.path.join(results_path, name)
            print(RESULTS_PATH.split('/')[-1])
            npz = load_querry_points(RESULTS_PATH)
            print("NPZ SHAPE:", npz.shape)
            data_file = DepthFile(SOURCE_PATH)
            load_depth_file(data_file)
            # depth_pcd is point cloud that see camera            
            depth_pcd = generate_input_pcd(data_file)

            input_file = []
            with open(TEST_QUERY_PATH, 'r') as f:
                input_file = json.load(f)
            print(type(input_file), len(input_file))
            print(type(npz), npz.shape)
            miss_match = 0
            object_image = []
            depth_image = []
            itr = 0
            halo = 0
            for key, value in input_file.items():
                pixel = []
                if len(value) == 1 and not np.any(np.isnan(value[0])):
                    halo += 1

                for i, row in enumerate(value):
                    x, y = map(int, key.split(', '))
                    if np.any(np.isnan(row)):
                        break
                    elif np.float32(row[0]) == npz[itr][0] and np.float32(row[1]) == npz[itr][1]:
                        z = npz[itr][0] + npz[itr][1] + data_file.dz
                        sdf = npz[itr][2]
                        object_image.append(np.array([x, y, z, sdf]))
                        itr += 1
                    else:
                        miss_match += 1

            print("MISS MATCH:", miss_match)
            print("HALO", halo)
            object_points = PointsFromDepth(data_file, object_image)
            print("object_points shape", object_points.image.shape)
            
            # critical_points = critical_points_point_cloud(object_points, h_begin=0.00002, h_end=-0.00003, min_thickness=0.00004)

            object_points.image = object_points.image[object_points.image[:, 3] >= -0.00003]
            object_points.image = object_points.image[object_points.image[:, 3] <= 0.00002]

            object_pcd = object_points.to_point_cloud()

            valid_points = np.column_stack((np.asarray(object_pcd.points), object_points.image[:, 3]))

            resolution = 100  # Adjust the resolution as needed
            volume, spacing = create_volumetric_grid(valid_points, resolution)
            verts, faces, normals, values = apply_marching_cubes(volume, spacing)

            pcd = o3d.geometry.PointCloud()  # create point cloud object
            pcd.points = o3d.utility.Vector3dVector(verts)

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = object_pcd.points #  o3d.utility.Vector3dVector(object_points)
            # mesh.triangles = o3d.utility.Vector3iVector(faces)

            alpha = 10
            alpha_shape = alphashape.alphashape(verts, alpha)
            print(type(alpha_shape), alpha_shape, alpha)

            alpha_verts = np.asarray(alpha_shape.vertices)
            alpha_faces = np.asarray(alpha_shape.faces)
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(alpha_verts)
            mesh.triangles = o3d.utility.Vector3iVector(alpha_faces)

            o3d.visualization.draw_geometries([object_pcd, mesh], mesh_show_back_face=True)
            print(1, verts.shape[0], faces.shape)
            print(2, np.asarray(mesh.vertices).shape[0], np.asarray(mesh.triangles).shape[0])

            # o3d.io.write_triangle_mesh(TEST_QUERY_PATH.replace('_query.json', '_mesh.ply'), mesh)
            # o3d.io.write_point_cloud(TEST_QUERY_PATH.replace('_query.json', '_points.pcd'), pcd)
            # mesh2 = o3d.io.read_triangle_mesh(TEST_QUERY_PATH.replace('_query.json', '_mesh.ply'))
            # print(3, np.asarray(mesh2.vertices).shape[0], np.asarray(mesh2.triangles).shape[0])

            exit(888)
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
            object_pcd = object_points.to_point_cloud()
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
            object_pcd.colors = o3d.utility.Vector3dVector(object_color_array)

            pcd_points_ndarray = np.concatenate((np.asarray(depth_pcd.points), np.asarray(object_pcd.points)), axis=0)
            pcd_colors_ndarray = np.concatenate((np.asarray(depth_pcd.colors), np.asarray(object_pcd.colors)), axis=0)
            pcd = o3d.geometry.PointCloud()  # create point cloud object
            pcd.points = o3d.utility.Vector3dVector(pcd_points_ndarray)  # set pcd_np as the point cloud points
            pcd.colors = o3d.utility.Vector3dVector(pcd_colors_ndarray)
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