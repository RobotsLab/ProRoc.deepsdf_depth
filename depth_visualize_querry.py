import argparse
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import random

from depth.utils import *
from depth.camera import Camera
from depth_image_generator import DepthImageFile as DepthFile

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


if __name__ == '__main__':
    SOURCE_PATH = f'dataset_YCB_train/DepthDeepSDF/files/untitled_1_0.txt'
    NPZ_PATH = 'examples/depth/Reconstructions/1000/Meshes/dataset_YCB_test/mug_depth/untitled_1_0_inp_test.npz'
    INPUT_PATH = 'data_YCB/SdfSamples/dataset_YCB_test/mug_depth/untitled_1_0_inp_test.txt'

    data_file = DepthFile(SOURCE_PATH)
    load_depth_file(data_file)

    input_file = []
    with open(INPUT_PATH, 'r') as f:
        input_file = f.readlines()
    input_pixels_list = [np.array(pixel.split(), dtype=np.float32) for pixel in input_file]
    output_ndarray = load_querry_points(NPZ_PATH)

    max_dim = max(arr.shape[0] for arr in input_pixels_list)
    padded_arrays = [np.pad(arr, [(0, max_dim - arr.shape[0])], constant_values=np.nan) for arr in input_pixels_list]
    input_pixels_array = np.vstack(padded_arrays)
    print(type(input_pixels_array), len(input_pixels_array), input_pixels_array.shape)

    visualize_dict = {}
    itr = 0
    sampled_points = 20

    for i, pixel in enumerate(input_pixels_array):
        if np.all(np.isnan(pixel)):
            visualize_dict[i] = pixel
        else:
            visualize_dict[i] = output_ndarray[sampled_points*itr:sampled_points*itr+10,:]
            itr+=1

    depth_image = []
    for key, value in visualize_dict.items():
        pixel = []
        for row in value:
            x = (key % data_file.ndx) + data_file.nx
            y = (key // data_file.ndy) + data_file.ny
            if np.any(np.isnan(row)):
                # pixel.append(np.array([x, y, np.nan, np.nan]))
                break
            else:
                z = row[0] + row[1] + data_file.dz
                sdf = row[2]
                depth_image.append(np.array([x, y, z, sdf]))

    image = np.vstack(depth_image)  # tu jest coś zjebane
    print(np.min(image, axis=0), np.max(image, axis=0))
    
    z = np.array(image[:, 2])
    x = (data_file.cx - image[:, 0]) * z / data_file.f  # y on image is x in real world
    y = (data_file.cy - image[:, 1]) * z / data_file.f  # x on image is y in real world

    points = np.column_stack((x, y, z))

    pcd = o3d.geometry.PointCloud()  # create point cloud object
    pcd.points = o3d.utility.Vector3dVector(points)  # set pcd_np as the point cloud points

    # Set the colors based on values
    sdf_array = image[:, 3]
    min_sdf = np.min(sdf_array)
    mean_sdf = np.mean(sdf_array)
    max_sdf = np.max(sdf_array)
    normalized_sdf_array = (sdf_array - min_sdf) / (max_sdf - min_sdf)

    color_array = np.zeros((len(normalized_sdf_array), 3))
    color_array[:, 0] = normalized_sdf_array

    zero_sdf_array = np.zeros(normalized_sdf_array.shape)
    zero_sdf_array[normalized_sdf_array <= 0.0001] = 1
    zero_sdf_array[normalized_sdf_array >= 0.0001] = 0
    color_array[:, 1] = zero_sdf_array

    color_array[:, 2] = normalized_sdf_array

    pcd.colors = o3d.utility.Vector3dVector(color_array)

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    o3d.visualization.draw_geometries([pcd, origin])
    # o3d.io.write_point_cloud('examples/depth/Reconstructions/1000/Meshes/dataset_YCB_test/mug_depth/untitled_1_0_inp_test.pcd', pcd)
