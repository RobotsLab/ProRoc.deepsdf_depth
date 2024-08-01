import argparse
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import random
import json

from scipy.spatial import KDTree

from depth.utils import *
from depth.camera import Camera
from depth_image_generator import DepthImageFile

from depth_file_generator import ViewFile
from depth_image_generator import translate, scale, rotate
from depth_training_data_generator import generate_pcd


K = 150
REJECTION_ANGLE = 25
QUERY = False

class TestFile():
    def __init__(self, source_path, destination_dir):
        self.source_path = source_path
        self.destination_dir = destination_dir 
        self.name = self.get_name_()
        self.o_c_transformation = np.zeros(6)
        self.pixels = []
        self.ds = 0

    def get_name_(self):
        tail = os.path.split(self.source_path)[1]
        return tail.split('.')[0]
    
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
        if QUERY:
            save_path = os.path.join(self.destination_dir, self.name + f'_k{K}_inp_test_query' +'.json')
        else:
            save_path = os.path.join(self.destination_dir, self.name + f'_k{K}_inp_test' +'.json')
        with open(save_path, "w") as outfile:
            json.dump(dictionary, outfile)
        print("Saved:", save_path)


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

def rejection_sampling(sdf):
    probability = random.random()
    rejection_function = np.exp(-K * sdf)
    if rejection_function < 0.3:
        rejection_function += 0.3
    if probability < np.exp(-K * sdf):
        return sdf
    else:
        return -1

def linspace_sampling(rd, fornt_bbox_z, back_bbox_z, num_samples, unique, visualize_dict, input_file, u, v, scene):
    sampled_points = np.linspace(fornt_bbox_z, back_bbox_z, num_samples)
    step = (back_bbox_z - fornt_bbox_z) / num_samples

    insiders = 0
    outsiders = 0
    for sample in sampled_points:
        dd = sample - rd - fornt_bbox_z
        passed_surfaces = 0
        for point_z in unique:
            if sample > point_z:
                passed_surfaces += 1

        if passed_surfaces == 0:
            outsiders += 1

            z = sample
            x = (input_file.cx - u) * z / input_file.f  # y on image is x in real world
            y = (input_file.cy - v) * z / input_file.f  # x on image is y in real world

            query_point = o3d.core.Tensor([[x, y, z]], dtype=o3d.core.Dtype.Float32)

            # Compute distance of the query point from the surface
            sdf = scene.compute_distance(query_point).item()

            visualize_dict[key].append([rd, dd, sdf])

    dd = random.uniform(0., 0.001)

    z = fornt_bbox_z + rd - dd
    x = (input_file.cx - u) * z / input_file.f  # y on image is x in real world
    y = (input_file.cy - v) * z / input_file.f  # x on image is y in real world

    query_point = o3d.core.Tensor([[x, y, z]], dtype=o3d.core.Dtype.Float32)

    # Compute distance of the query point from the surface
    sdf = scene.compute_distance(query_point).item()


    visualize_dict[key].append([rd, -dd, sdf])
    visualize_dict[key].append([rd, dd, 0])

    return insiders, outsiders

if __name__ == '__main__':
    train_new4_bottle = [
    # "examples/new_exp_3/data/training_data/bottle/10f709cecfbb8d59c2536abb1e8e5eab_5_a25_view4.json",
    # "examples/new_exp_3/data/training_data/bottle/10f709cecfbb8d59c2536abb1e8e5eab_5_a25_view9.json",
    # "examples/new_exp_3/data/training_data/bottle/13d991326c6e8b14fce33f1a52ee07f2_5_a25_view4.json",
    # "examples/new_exp_3/data/training_data/bottle/13d991326c6e8b14fce33f1a52ee07f2_5_a25_view9.json",
    # "examples/new_exp_3/data/training_data/bottle/109d55a137c042f5760315ac3bf2c13e_5_a25_view4.json",
    # "examples/new_exp_3/data/training_data/bottle/109d55a137c042f5760315ac3bf2c13e_5_a25_view9.json",
    # "examples/new_exp_3/data/training_data/bottle/1349b2169a97a0ff54e1b6f41fdd78a_5_a25_view4.json",
    # "examples/new_exp_3/data/training_data/bottle/1349b2169a97a0ff54e1b6f41fdd78a_5_a25_view9.json",
    # "examples/new_exp_3/data/training_data/bowl/1b4d7803a3298f8477bdcb8816a3fac9_5_a25_view4.json",
    # "examples/new_exp_3/data/training_data/bowl/1b4d7803a3298f8477bdcb8816a3fac9_5_a25_view9.json",
    # "examples/new_exp_3/data/training_data/bowl/2c1df84ec01cea4e525b133235812833_5_a25_view4.json",
    # "examples/new_exp_3/data/training_data/bowl/2c1df84ec01cea4e525b133235812833_5_a25_view9.json",
    # "examples/new_exp_3/data/training_data/bowl/12ddb18397a816c8948bef6886fb4ac_5_a25_view4.json",
    # "examples/new_exp_3/data/training_data/bowl/12ddb18397a816c8948bef6886fb4ac_5_a25_view9.json",
    # "examples/new_exp_3/data/training_data/bowl/292d2dda9923752f3e275dc4ab785b9f_5_a25_view4.json",
    # "examples/new_exp_3/data/training_data/bowl/292d2dda9923752f3e275dc4ab785b9f_5_a25_view9.json",
    # "examples/new_exp_3/data/training_data/laptop/1bb2e873cfbef364cef0dab711014aa8_5_a25_view4.json",
    # "examples/new_exp_3/data/training_data/laptop/1bb2e873cfbef364cef0dab711014aa8_5_a25_view9.json",
    # "examples/new_exp_3/data/training_data/laptop/1f507b26c31ae69be42930af58a36dce_5_a25_view4.json",
    # "examples/new_exp_3/data/training_data/laptop/1f507b26c31ae69be42930af58a36dce_5_a25_view9.json",
    # "examples/new_exp_3/data/training_data/laptop/2c61f0ba3236fe356dae27c417fa89b_5_a25_view4.json",
    # "examples/new_exp_3/data/training_data/laptop/2c61f0ba3236fe356dae27c417fa89b_5_a25_view9.json",
    # "examples/new_exp_3/data/training_data/laptop/16c49793f432cd4b33e4e0fe8cce118e_5_a25_view4.json",
    # "examples/new_exp_3/data/training_data/laptop/16c49793f432cd4b33e4e0fe8cce118e_5_a25_view9.json",
    # "examples/new_exp_3/data/training_data/mug/1eaf8db2dd2b710c7d5b1b70ae595e60_5_a25_view4.json",
    # "examples/new_exp_3/data/training_data/mug/1eaf8db2dd2b710c7d5b1b70ae595e60_5_a25_view9.json",
    "examples/new_exp_4/data/training_data/mug/10f6e09036350e92b3f21f1137c3c347_2_a25_view0.json",
    "examples/new_exp_4/data/training_data/mug/10f6e09036350e92b3f21f1137c3c347_2_a25_view1.json",
    "examples/new_exp_4/data/training_data/mug/10f6e09036350e92b3f21f1137c3c347_2_a25_view2.json",
    "examples/new_exp_4/data/training_data/mug/10f6e09036350e92b3f21f1137c3c347_2_a25_view3.json",
    "examples/new_exp_4/data/training_data/mug/10f6e09036350e92b3f21f1137c3c347_2_a25_view4.json",
    "examples/new_exp_4/data/training_data/mug/10f6e09036350e92b3f21f1137c3c347_2_a25_view5.json",
    "examples/new_exp_4/data/training_data/mug/10f6e09036350e92b3f21f1137c3c347_2_a25_view6.json",
    "examples/new_exp_4/data/training_data/mug/10f6e09036350e92b3f21f1137c3c347_2_a25_view7.json",
    "examples/new_exp_4/data/training_data/mug/10f6e09036350e92b3f21f1137c3c347_2_a25_view8.json",
    "examples/new_exp_4/data/training_data/mug/10f6e09036350e92b3f21f1137c3c347_2_a25_view9.json",
    # "examples/new_exp_3/data/training_data/mug/15bd6225c209a8e3654b0ce7754570c8_5_a25_view4.json",
    # "examples/new_exp_3/data/training_data/mug/15bd6225c209a8e3654b0ce7754570c8_5_a25_view9.json",
    # "examples/new_exp_3/data/training_data/mug/141f1db25095b16dcfb3760e4293e310_5_a25_view4.json",
    # "examples/new_exp_3/data/training_data/mug/141f1db25095b16dcfb3760e4293e310_5_a25_view9.json"
]
    experiment_name = 'new_exp_4'
    categories = ['mug']  # ['bottle', 'bowl', 'mug']

    with open(f'examples/{experiment_name}/data/dataset_config.json', 'r') as json_file:
        config = json.load(json_file)
    
    for category in categories:
        names_json = [name.rstrip('.json') for name in os.listdir(f'examples/{experiment_name}/data/training_data/{category}') if name.endswith(".json") and not "trai" in name]
        
        DESTINATION_PATH = f'examples/{experiment_name}/data/test_data/{category}'
        saved_files = 0
        for name_json in names_json:
            view = int(name_json.split('view')[1])
            # if not view in [4, 9]:
            #     continue
            object_name = name_json.split('_')[0]

            SOURCE_PATH = os.path.join(f'examples/{experiment_name}/data/training_data/{category}', name_json + '.json')
            if not SOURCE_PATH in train_new4_bottle:
                continue
            input_file = DepthImageFile(object_name)
            input_file.load(SOURCE_PATH)

            output_file = TestFile(SOURCE_PATH, DESTINATION_PATH)

            MESH_SOURCE_PATH = os.path.join(f"examples/{experiment_name}/data/{category}", object_name + f"_{config['rotation_step']}.json")
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
            scaled_mesh = rotate(scaled_mesh, [0,0,90])
            scaled_mesh = rotate(scaled_mesh, np.array([-135, 0, 0]))
            scaled_mesh = translate(scaled_mesh, [0, 0, 1.5])

            # pcd = generate_pcd(input_file)
            # points = np.asarray(pcd.points)
            # print('pcd', np.mean(points, axis=0))

            # pcd_mean = np.mean(points, axis=0)
            # print('pcd', pcd_mean)
            # pcd.estimate_normals()
            
            # distances = pcd.compute_nearest_neighbor_distance()
            # avg_dist = np.median(distances)
            # radius = avg_dist
            # ply = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 5, radius * 10, radius * 15]))
            
            # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
            # o3d.visualization.draw_geometries([pcd, scaled_mesh, origin])
            # continue

            scene = o3d.t.geometry.RaycastingScene()
            # mesh = o3d.t.geometry.TriangleMesh.from_legacy(ply)
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(scaled_mesh)
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
            count_insiders = 0
            count_outsiders = 0
            count_totals = 0
            for key, value in input_file.pixels.items():
                unique = np.unique(value[value!=0])
                x, y = key
                key = f"{x}, {y}"
                visualize_dict[key] = []

                # sprawdzamy czy liczba przecięć jest parzysta
                if unique.any():  # and len(unique)%2 == 0:
                    # obliczamy podstawowe parametry
                    first_surface = unique[0]
                    rd = first_surface - fornt_bbox_z

                    # samplujemy punkty po promieniu
                    insider, outsider = linspace_sampling(rd, fornt_bbox_z, back_bbox_z, num_samples, unique, visualize_dict, input_file, x, y, scene)
                    # obliczamy sdf
                    # punktom, które znajdują się za nieparzystą liczbą ścian przypisujemy wartość 0
                    # pozostałym punktom szukamy najbliższej powierzchni
                    count_totals += 1
                    count_insiders += insider
                    count_outsiders += outsider
                else:
                    output_file.pixels.append(np.array([np.nan]))
                    visualize_dict[key].append([np.nan])
            output_file.save(visualize_dict)
            print("Total:", len(visualize_dict))
            print("Max saved sdf:", max_saved_sdf)

            print("\nNans:", nans)
            print("Total:", samples)
            print("Ratio:", format(nans/samples, ".00%"))
            print("PROBLEMS", problems)
            print("Samples", samples)
            print("--------------------------------------")