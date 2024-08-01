import argparse
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import random
import json
import numpy as np
import os

# from depth.utils import *
from depth.camera import Camera
from depth_image_generator import File as DepthFile
# from sklearn.preprocessing import StandardScaler

from depth_file_generator import File as ViewsFile
from depth_image_generator import load_generator_file, translate, scale, rotate

class Camera():
    def __init__(self, Fx, Fy, Cx, Cy, width, height, intrinsic_matrix):

        self.Fx = Fx
        self.Fy = Fy
        self.Cx = Cx
        self.Cy = Cy
        self.width = width
        self.height = height
        self.intrinsic_matrix = intrinsic_matrix
        self.extrinsic_matrix = np.eye(4)

    def translate(self, tx, ty, tz):
        self.extrinsic_matrix[:3, 3] = [tx, ty, tz]
        # print('Extrinsic matrix:\n',self.extrinsic_matrix)

    def rotate(self, roll, pitch, yaw):

        rotation_x = np.array([[1, 0, 0],
                            [0, np.cos(roll), -np.sin(roll)],
                            [0, np.sin(roll), np.cos(roll)]])

        rotation_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                            [0, 1, 0],
                            [-np.sin(pitch), 0, np.cos(pitch)]])

        rotation_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                            [np.sin(yaw), np.cos(yaw), 0],
                            [0, 0, 1]])

        # Combine rotation matrices
        rotation_matrix = np.dot(np.dot(rotation_z, rotation_y), rotation_x)

        self.extrinsic_matrix[:3, :3] = rotation_matrix

    def raycasting(self):
        return o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix=self.intrinsic_matrix,
            extrinsic_matrix=self.extrinsic_matrix,
            width_px=self.width,
            height_px=self.height
        )

class ViewsFile():
    def __init__(self, source_path, destination_dir=''):
        self.source_path = source_path
        self.destination_dir = destination_dir 
        self.name = self.get_name_()
        self.scale = 0
        self.s_o_transformation = np.zeros(6)
        self.o_c_transformation = np.zeros(6)
        self.frames = []

        if destination_dir:
            self.version = self.get_version_()

    def get_name_(self):
        head = os.path.split(self.source_path)[0]
        return head.split('/')[-2]
    
    def get_version_(self):
        dir_files = os.listdir(self.destination_dir)
        file_number = len([x for x in dir_files if x.startswith(self.name) and x.endswith('.txt')]) + 1
        return file_number
    
    def save(self):
        with open(os.path.join(self.destination_dir, self.name + '.txt'), 'w') as f:
            f.write(f'{self.scale}\n')
            f.write(f"{' '.join(map(str, self.s_o_transformation))}\n")
            f.write(f"{' '.join(map(str, self.o_c_transformation))}\n")
            for frame in self.frames:
                f.write(f"{' '.join(map(str, frame))}\n")
        print(f"Saved: {os.path.join(self.destination_dir, self.name + '.txt')}")

class DepthFile():
    def __init__(self, source_path, destination_dir):
        self.source_path = source_path
        self.destination_dir = destination_dir 
        self.name = self.get_name_()
        self.version = 0  # self.get_version_()
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
        z = np.array(self.image[:, 2])
        x = (self.cx - self.image[:, 0]) * z / self.f  # y on image is x in real world
        y = (self.cy - self.image[:, 1]) * z / self.f  # x on image is y in real world

        self.points = np.column_stack((x, y, z))

        pcd = o3d.geometry.PointCloud()  # create point cloud object
        pcd.points = o3d.utility.Vector3dVector(self.points)  # set pcd_np as the point cloud points

        return pcd

def load_generator_file(input_file):
    with open(input_file.source_path, "r") as file:
        input_file.scale = float(file.readline())
        input_file.s_o_transformation = np.array(file.readline().split(), dtype=np.float32)
        input_file.o_c_transformation = np.array(file.readline().split(), dtype=np.float32)
        frames = file.readlines()
        input_file.frames = [np.array(frame.split(), dtype=np.float32) for frame in frames]

def load_querry_points(path):
    if path.endswith(".npz"):
        dict_data = np.load(path)
        pos_data = dict_data[dict_data.files[0]]
        data = np.concatenate([pos_data])
        # print(data.shape)

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

if __name__ == '__main__':
    vis = o3d.visualization.Visualizer()
    k = 150
    rejection_angle = 25
    categories = ['mug', 'bowl', 'laptop', 'bottle']
    for category in categories:
        results_path = f'data_YCB/SdfSamples/dataset_YCB_test/{category}_pos_sdf'
        names_json = [name for name in os.listdir(results_path) if name.endswith('test.json')]
        for name in names_json:
            # print(name)
            # SOURCE_PATH = f"dataset_YCB_train/DepthDeepSDF/files/{category}/{name.replace('_k150_inp_test.npz', '.txt')}"
            # TEST_QUERY_PATH = f"data_YCB/SdfSamples/dataset_YCB_test/test_new5_{category}/{name.replace('.npz', '_query.json')}" #_k150_inp_train.json'
            # RESULTS_PATH = os.path.join(results_path, name)
            # print(RESULTS_PATH.split('/')[-1])
            # npz = load_querry_points(RESULTS_PATH)

            # SOURCE_PATH = f'dataset_YCB_train/DepthDeepSDF/files/untitled_1_{b}_a{rejection_angle}.txt'
            # INPUT_PATH = f'dataset_YCB_train/DepthDeepSDF/files/untitled_1_{b}_a{rejection_angle}_k{k}_inp.json'
            VIEW_PATH = f"data_YCB/SdfSamples/dataset_YCB_test/{category}_pos_sdf/{name.split('_')[0]}.txt"
            SOURCE_PATH = f"data_YCB/SdfSamples/dataset_YCB_test/{category}_pos_sdf/{name.replace('_k150_inp_test.json', '.txt')}"
            INPUT_PATH = f"data_YCB/SdfSamples/dataset_YCB_test/{category}_pos_sdf/{name}"

            view_file = ViewsFile(VIEW_PATH)
            load_generator_file(view_file)

            data_file = DepthFile(SOURCE_PATH, '')
            load_depth_file(data_file)
            
            depth_pcd = generate_input_pcd(data_file)

            input_file = []
            with open(INPUT_PATH, 'r') as f:
                input_file = json.load(f)

            object_image = []
            depth_image = []

            for key, value in input_file.items():
                pixel = []

                for i, row in enumerate(value):
                    x, y = map(int, key.split(', '))

                    if np.any(np.isnan(row)):
                        break
                    else:
                        z = row[0] + row[1] + data_file.dz
                        sdf = row[2]
                        object_image.append(np.array([x, y, z, sdf]))

            object_points = PointsFromDepth(data_file, object_image)
            object_pcd = object_points.to_point_cloud()
            points = np.asarray(object_pcd.points)
            center = np.mean(points, axis=0)
            points -= center
            R = object_pcd.get_rotation_matrix_from_xyz((np.pi / 4, 0, 0))
            object_pcd.rotate(R, center=(0, 0, 0))

            max_distance = 0
            for point in points:
                max_distance = max(max_distance, np.linalg.norm(point))
            
            scale_factor = 1.03
            scale_multiplier = max_distance * scale_factor
            points /= scale_multiplier
            
            object_sdf_array = object_points.image[:, 3]
            object_sdf_array /= scale_multiplier

            npz_result = np.column_stack((points, object_sdf_array))
            pos_data = npz_result[npz_result[:, 3] >= 0]
            neg_data = npz_result[npz_result[:, 3] < 0]
            print("POS DATA SHAPE, NEG DATA SHAPE: ", pos_data.shape, neg_data.shape)

            npz_data = {"pos": pos_data, "neg": neg_data}
            # np.savez(f"{INPUT_PATH.replace('.json', '.npz')}", **npz_data)

            object_color_array = np.zeros((len(object_sdf_array), 3))

            # object_sdf_array = np.clip(object_sdf_array, 0, 0.01)
            min_sdf = np.min(object_sdf_array)
            mean_sdf = np.mean(object_sdf_array)
            max_sdf = np.max(object_sdf_array)
            normalized_sdf_array = (object_sdf_array - min_sdf) / (max_sdf - min_sdf)

            # plt.hist(normalized_sdf_array, bins=100)
            # plt.show()

            print(min_sdf, max_sdf)
            object_color_array[:, 2] = (1 - normalized_sdf_array) ** 2
            # object_color_array[normalized_sdf_array == 0, 1] = 1
            object_pcd.colors = o3d.utility.Vector3dVector(object_color_array)

            pcd_points_ndarray = np.concatenate((pos_data, neg_data), axis=0)
            pcd_colors_ndarray = np.concatenate((np.asarray(depth_pcd.colors), np.asarray(object_pcd.colors)), axis=0)
            pcd = o3d.geometry.PointCloud()  # create point cloud object
            pcd.points = o3d.utility.Vector3dVector(pcd_points_ndarray[:, :3])  # set pcd_np as the point cloud points
            pcd.colors = o3d.utility.Vector3dVector(pcd_colors_ndarray)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)

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

            # exit(777)

            # o3d.visualization.draw_geometries([pcd, origin])
            # continue
            # o3d.io.write_point_cloud(f'dataset_YCB_train/DepthDeepSDF/files/untitled_1_{b}_a{rejection_angle}_k{k}_inp_test.pcd', pcd)
                    
            #nasycenie nieliniowe kolorów do wizualizacji
            # dane do json
            # odwrócić wartości sdf - na zewnątrz 0 wewnątrz dodatnie wartości
            # ASAP wysłać dane
            # badanie na 1 obiekcie, 50 widoków