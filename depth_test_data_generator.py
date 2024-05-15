import argparse
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import random
import json

from scipy.spatial import KDTree

from depth.utils import *
from depth.camera import Camera
from depth_image_generator import File as DepthFile

from depth_file_generator import File as ViewsFile
from depth_image_generator import load_generator_file, translate, scale, rotate


K = 150
REJECTION_ANGLE = 10

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

    def save(self, dictionary):
        with open(os.path.join(self.destination_dir, self.name + f'_k{K}_inp_test' +'.json'), "w") as outfile:
            json.dump(dictionary, outfile)
        print("Saved:", os.path.join(self.destination_dir, self.name + f'_k{K}_inp_test' +'.json'))


def load_depth_file(input_file):
    with open(input_file.source_path, "r") as file:
        print(input_file.source_path)
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

def generate_pcd(input_file):
    pixels = np.asarray(input_file.pixels)
    pixels = np.reshape(pixels, (input_file.ndy, input_file.ndx, -1))

    points = np.zeros((1,3))

    for image in range(pixels.shape[2]):
        img = np.zeros((input_file.Ndy, input_file.Ndx))
        img[input_file.ny:input_file.ny+input_file.ndy,input_file.nx:input_file.nx+input_file.ndx] = pixels[:, :, image]
        roi_y, roi_x = np.where(img!=0)

        # plt.imshow(img, cmap='gray')
        # plt.show()

        z = np.array(img[img!=0])
        x = (input_file.cx - roi_x) * z / input_file.f  # y on image is x in real world
        y = (input_file.cy - roi_y) * z / input_file.f  # x on image is y in real world

        points_data = np.column_stack([x, y, z])
        points = np.concatenate((points, points_data), axis=0) 

    points = np.delete(points, 0, axis=0)
    pcd = o3d.geometry.PointCloud()  # create point cloud object
    pcd.points = o3d.utility.Vector3dVector(points)  # set pcd_np as the point cloud points

    # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # o3d.visualization.draw_geometries([pcd, origin])
    
    return pcd

def rejection_sampling(sdf):
    probability = random.random()
    if probability < np.exp(-K * sdf):
        return sdf
    else:
        return -1

def linspace_sampling(rd, fornt_bbox_z, back_bbox_z, num_samples, unique, visualize_dict, input_file, u, v, scene):
    sampled_points = np.linspace(fornt_bbox_z, back_bbox_z, num_samples)
    # object_points = np.asarray(pcd.points)
    # tree = KDTree(object_points, leafsize=object_points.shape[0]+1)
    
    for sample in sampled_points:
        dd = sample - rd - fornt_bbox_z
        passed_surfaces = 0
        for point_z in unique:
            if sample > point_z:
                passed_surfaces += 1

        if passed_surfaces >= 2:
            continue

        if passed_surfaces % 2 == 1:
            sdf = 0
            
        elif len(unique) > passed_surfaces:
            z = sample
            x = (input_file.cx - u) * z / input_file.f  # y on image is x in real world
            y = (input_file.cy - v) * z / input_file.f  # x on image is y in real world
            # distances, ndx = tree.query(np.array([x, y, z]), k=1)  # distances is the same as sdf

            query_point = o3d.core.Tensor([[x, y, z]], dtype=o3d.core.Dtype.Float32)

            # Compute distance of the query point from the surface
            sdf = scene.compute_distance(query_point).item()

            # rejection sampling
            sdf = rejection_sampling(sdf)
            if sdf < 0:
                continue

        else:
            z = sample
            x = (input_file.cx - u) * z / input_file.f  # y on image is x in real world
            y = (input_file.cy - v) * z / input_file.f  # x on image is y in real world
            # distances, ndx = tree.query(np.array([x, y, z]), k=1)  # distances is the same as sdf

            query_point = o3d.core.Tensor([[x, y, z]], dtype=o3d.core.Dtype.Float32)

            # Compute distance of the query point from the surface
            sdf = scene.compute_distance(query_point).item()

            # rejection sampling
            sdf = rejection_sampling(sdf)
            if sdf < 0:
                continue

        visualize_dict[key].append([rd, dd, sdf])

if __name__ == '__main__':
    names_txt = [
        'dataset_YCB_train/DepthDeepSDF/files/2c1df84ec01cea4e525b133235812833_5_a20.txt',
        'dataset_YCB_train/DepthDeepSDF/files/4227b58665eadcefc0dc3ed657ab97f0_8_a20.txt',
        'dataset_YCB_train/DepthDeepSDF/files/11547e8d8f143557525b133235812833_5_a20.txt',
        'dataset_YCB_train/DepthDeepSDF/files/4b32d2c623b54dd4fe296ad57d60d898_0_a20.txt'

    ]
    names_gt = [
        'dataset_YCB_train/DepthDeepSDF/files/2c1df84ec01cea4e525b133235812833_5_gt.txt',
        'dataset_YCB_train/DepthDeepSDF/files/4227b58665eadcefc0dc3ed657ab97f0_8_gt.txt',
        'dataset_YCB_train/DepthDeepSDF/files/11547e8d8f143557525b133235812833_5_gt.txt',
        'dataset_YCB_train/DepthDeepSDF/files/4b32d2c623b54dd4fe296ad57d60d898_0_gt.txt'
    ]
    for name_txt, name_gt in zip(names_txt, names_gt):
        SOURCE_PATH = name_txt
        GT_PATH = name_gt
        DESTINATION_PATH = 'dataset_YCB_train/DepthDeepSDF/files'

        input_file = DepthFile(SOURCE_PATH)
        load_depth_file(input_file)
        print("INPUT FILE LOADED")
        output_file = File(SOURCE_PATH, DESTINATION_PATH)

        gt_file = DepthFile(GT_PATH)
        load_depth_file(gt_file)
        print("GT FILE LOADED")

        pcd = generate_pcd(gt_file)
        points = np.asarray(pcd.points)
        print('pcd', np.mean(points, axis=0))
        pcd.estimate_normals()
        
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = avg_dist
        ply = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))

        scene = o3d.t.geometry.RaycastingScene()
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(ply)
        _ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh

        nans = 0
        problems = 0
        num_samples = 100
        max_sdf = 0.02
        max_saved_sdf = 0
        samples = 1
        output_file.pixels = []
        visualize_dict = {}
        fornt_bbox_z = input_file.dz  # + 0.05
        back_bbox_z = input_file.dz2  # - 0.1
        print(len(input_file.pixels))

        for i, pixel in enumerate(input_file.pixels):
            unique = np.unique(pixel[pixel!=0])
            x = (i % input_file.ndx) + input_file.nx
            y = (i // input_file.ndx) + input_file.ny
            key = f"{x}, {y}"
            visualize_dict[key] = []

            # sprawdzamy czy liczba przecięć jest parzysta
            if unique.any() and len(unique)%2 == 0:
                # obliczamy podstawowe parametry
                first_surface = unique[0]
                rd = first_surface - fornt_bbox_z

                # samplujemy punkty po promieniu
                linspace_sampling(rd, fornt_bbox_z, back_bbox_z, num_samples, unique, visualize_dict, input_file, x, y, scene)
                # obliczamy sdf
                # punktom, które znajdują się za nieparzystą liczbą ścian przypisujemy wartość 0
                # pozostałym punktom szukamy najbliższej powierzchni
            else:
                output_file.pixels.append(np.array([np.nan]))
                visualize_dict[key].append([np.nan])
                nans += 1

        output_file.save(visualize_dict)

        print("Total:", len(visualize_dict))
        print("Max saved sdf:", max_saved_sdf)

        print("\nNans:", nans)
        print("Total:", samples)
        print("Ratio:", format(nans/samples, ".00%"))
        print("PROBLEMS", problems)
        print("Samples", samples)
        print("--------------------------------------")

            # print(f'FAILED TO LOAD: {name_txt}')
            # exit(777)

