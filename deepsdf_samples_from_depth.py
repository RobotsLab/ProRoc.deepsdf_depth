import argparse
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import random
import json
import numpy as np
import os

from depth.utils import *
from depth.camera import Camera
from depth_image_generator import DepthImageFile

from depth_file_generator import ViewFile
from depth_file_generator import translate, scale, rotate


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

    for key, values in input_file.pixels.items():
        unique = np.unique(values[values!=0])
        x, y = key
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

def translate_pcd(pcd, translation):
    pcd_points = np.asarray(pcd.points)
    pcd_points += translation
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    return pcd

def rotate_pcd(pcd, rotation):
    R = pcd.get_rotation_matrix_from_xyz(np.radians(rotation))
    pcd.rotate(R, center=(0, 0, 0))
    return pcd

def scale_pcd(pcd, scale_factor):

    def object_translation(points, pos_z=False):
        points_mean = np.mean(points, axis=0)
        points -= points_mean

        if pos_z:
            min_z = np.min(points[:, 2])
        else:
            min_z = 0

        translation_vec = points_mean + np.array([0., 0., min_z])
        return translation_vec
    
    pcd_points = np.copy(np.asarray(pcd.points))    
    max_idx = 0
    max_dist = 0
    for i in range(3):
        input_max_dist = np.max(pcd_points[:, i]) - np.min(pcd_points[:, i])
        max_dist = max(max_dist, input_max_dist)
        if max_dist == input_max_dist:
            max_idx = i

    max_z_dist = np.max(pcd_points[:, max_idx]) - np.min(pcd_points[:, max_idx])
    pcd_points /= max_z_dist
    
    pcd_points *= scale_factor
    output_z_dist = np.max(pcd_points[:, max_idx]) - np.min(pcd_points[:, max_idx])
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    scaled_pcd = translate(pcd, object_translation(pcd_points, True))

    real_scale_factor = output_z_dist / max_dist
    
    print(f"Max dist catured along axis: {max_idx}")

    return scaled_pcd, real_scale_factor

if __name__ == '__main__':
    vis = o3d.visualization.Visualizer()
    k = 150
    rejection_angle = 25
    categories = ['mug', 'bottle', 'bowl']
    experiment_name = 'new_exp_3'
    with open(f'examples/{experiment_name}/data/dataset_config.json', 'r') as json_file:
        config = json.load(json_file)
    
    for category in categories:
        generated_files = [gf.split('.')[0] for gf in os.listdir(f'examples/{experiment_name}/data/test_data/{category}') if gf.endswith('.json')]
        names_json = [name for name in os.listdir(f'examples/{experiment_name}/data/{category}') if name.endswith(f"_{config['rotation_step']}.json")]

        for current_iteration, name_json in enumerate(generated_files):
            name = name_json.split('_')[0]
            view = int(name_json.split('_')[3][-1])
            VIEW_PATH = os.path.join(f'examples/{experiment_name}/data/{category}', name_json.split(f'_a25_view{view}_k150_inp_test')[0]+'.json')
            SOURCE_PATH = os.path.join(f'examples/{experiment_name}/data/training_data/{category}', name_json.split('_k150_inp_test')[0]+'.json')
            INPUT_PATH = os.path.join(f'examples/{experiment_name}/data/test_data/{category}', name_json+'.json')
            MESH_PATH =  os.path.join(f'examples/{experiment_name}/data/deepsdf/{category}', name, 'models/model_normalized.obj')

            view_file = ViewFile(name)
            view_file.load(VIEW_PATH)

            data_file = DepthImageFile(name)
            data_file.load(SOURCE_PATH)
            
            input_mesh = load_file(MESH_PATH)
            # input_mesh = rotate(input_mesh, np.array([90, 0, 0]))
            # centered_mesh = translate(input_mesh, view_file.s_o_transformation[:3])
            # scaled_mesh, _ = scale(centered_mesh, view_file.scale)

            frame = view_file.frames[view]
            # scaled_mesh = translate(input_mesh, frame[:3])
            # scaled_mesh = rotate(scaled_mesh, frame[3:])
            # scaled_mesh = rotate(scaled_mesh, [0,0,90])
            # scaled_mesh = rotate(scaled_mesh, np.array([-135, 0, 0]))
            # scaled_mesh = translate(scaled_mesh, [0, 0, 1.5])

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
            # points -= center
            # R = object_pcd.get_rotation_matrix_from_xyz((np.pi / 4, 0, 0))
            # object_pcd.rotate(R, center=(0, 0, 0))

            max_distance = 0
            for point in points:
                max_distance = max(max_distance, np.linalg.norm(point))
            
            # scale_factor = 1.03
            # scale_multiplier = max_distance * scale_factor
            # points /= scale_multiplier
            
            object_sdf_array = object_points.image[:, 3]
            # object_sdf_array /= scale_multiplier

            npz_result = np.column_stack((points, object_sdf_array))
            pos_data = npz_result[npz_result[:, 3] >= 0]
            neg_data = npz_result[npz_result[:, 3] < 0]
            print("POS DATA SHAPE, NEG DATA SHAPE: ", pos_data.shape, neg_data.shape)

            npz_data = {"pos": pos_data, "neg": neg_data}

            object_color_array = np.zeros((len(object_sdf_array), 3))

            # object_sdf_array = np.clip(object_sdf_array, 0, 0.01)
            min_sdf = np.min(object_sdf_array)
            mean_sdf = np.mean(object_sdf_array)
            max_sdf = np.max(object_sdf_array)
            normalized_sdf_array = (object_sdf_array - min_sdf) / (max_sdf - min_sdf)

            # plt.hist(normalized_sdf_array, bins=100)
            # plt.show()

            print(min_sdf, max_sdf)
            # object_color_array[:, 2] = (1 - normalized_sdf_array) ** 2
            # object_color_array[normalized_sdf_array == 0, 1] = 1
            object_color_array[:, 2] = object_sdf_array

            object_pcd.colors = o3d.utility.Vector3dVector(object_color_array)

            pcd_points_ndarray = np.concatenate((pos_data, neg_data), axis=0)
            pcd_colors_ndarray = np.concatenate((np.asarray(depth_pcd.colors), np.asarray(object_pcd.colors)), axis=0)
            pcd = o3d.geometry.PointCloud()  # create point cloud object
            pcd.points = o3d.utility.Vector3dVector(pcd_points_ndarray[:, :3])  # set pcd_np as the point cloud points
            pcd.colors = o3d.utility.Vector3dVector(object_color_array)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
            pcd = translate_pcd(pcd, -np.array([0, 0, 1.5]))
            pcd = rotate_pcd(pcd, np.array([135, 0, 0]))  # Inverse rotation
            pcd = rotate_pcd(pcd, [0, 0, -90])  # Inverse rotation
            pcd = rotate_pcd(pcd, -np.array(frame[3:]))  # Inverse rotation
            pcd = translate_pcd(pcd, -np.array(frame[:3]))  # Inverse translation

            # Extract the points (x, y, z) and sdf values from pcd
            extracted_points = np.asarray(pcd.points)
            extracted_sdf = np.asarray(pcd.colors)[:, 2]

            # Combine the extracted points with the sdf values
            combined_data = np.column_stack((extracted_points, extracted_sdf))
            if np.array_equal(combined_data, pcd_points_ndarray):
                print("array1 and array2 are equal and in the same order.")
            else:
                print("array1 and array2 are not equal or the order has changed.")
            # Separate into positive and negative sdf arrays for saving
            pos_data_extracted = combined_data[combined_data[:, 3] >= 0]
            neg_data_extracted = combined_data[combined_data[:, 3] < 0]

            npz_data_extracted = {"pos": pos_data_extracted, "neg": neg_data_extracted}

            # Save the extracted data to a .npz file
            np.savez(f"{os.path.join(f'examples/{experiment_name}/data/deepsdf/{category}', name_json)}.npz", **npz_data_extracted)

            # pcd_points_ndarray = np.concatenate((np.asarray(depth_pcd.points), np.asarray(object_pcd.points)), axis=0)
            # pcd_colors_ndarray = np.concatenate((np.asarray(depth_pcd.colors), np.asarray(object_pcd.colors)), axis=0)
            # pcd = o3d.geometry.PointCloud()  # create point cloud object
            # pcd.points = o3d.utility.Vector3dVector(np.asarray(object_pcd.points))  # set pcd_np as the point cloud points
            # pcd.colors = o3d.utility.Vector3dVector(np.asarray(object_pcd.colors))
            # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

            # vis.create_window()

            # vis.add_geometry(pcd)
            # vis.add_geometry(depth_pcd)
            # vis.add_geometry(input_mesh)

            # opt = vis.get_render_option()

            # opt.background_color = np.asarray([255, 255, 255])

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