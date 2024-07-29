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

from depth_file_generator import ViewFile, translate, scale, rotate


K = 150
REJECTION_ANGLE = 25

class TrainingFile():
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
        file_number = len([x for x in dir_files if x.startswith(self.name) and x.endswith('.json')]) + 1
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
        with open(os.path.join(self.destination_dir, self.name + f'_k{K}_inp_train' +'.json'), "w") as outfile:
            json.dump(dictionary, outfile)
        print("Saved:", os.path.join(self.destination_dir, self.name + f'_k{K}_inp_train' +'.json'))


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

def generate_pcd(input_file):
    points = []
    for key, value in input_file.pixels.items():
        u, v = key
        z = value[value != 0]

        x = (input_file.cx - int(u)) * z / input_file.f  # y on image is x in real world
        y = (input_file.cy - int(v)) * z / input_file.f  # x on image is y in real world

        points_data = np.column_stack([x, y, z])
        points.extend(points_data)

    points = np.row_stack(points)
    pcd = o3d.geometry.PointCloud()  # create point cloud object
    pcd.points = o3d.utility.Vector3dVector(points)  # set pcd_np as the point cloud points

    # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # o3d.visualization.draw_geometries([pcd, origin])
    
    return pcd

def rejection_sampling(sdf):
    probability = random.random()
    rejection_function = np.exp(-K * sdf)
    if rejection_function < 0.3:
        rejection_function += 0.3
    if probability < rejection_function:
        return sdf
    else:
        return -1

def linspace_sampling(rd, fornt_bbox_z, back_bbox_z, num_samples, unique, visualize_dict, input_file, u, v, scene, halo=False):
    sampled_points = np.linspace(fornt_bbox_z, back_bbox_z, num_samples)
    # object_points = np.asarray(pcd.points)
    # tree = KDTree(object_points, leafsize=object_points.shape[0]+1)
    
    for sample in sampled_points:
        dd = sample - rd - fornt_bbox_z
        passed_surfaces = 0
        for point_z in unique:
            if sample > point_z:
                passed_surfaces += 1

        if passed_surfaces % 2 == 1 and not halo:
            sdf = 0
        elif halo:
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
    
    for i, point_z in enumerate(unique):
        dist = 0.001
        if len(unique) > 1 and i == 0:
            dist = min([dist, unique[i+1] - point_z])
        elif len(unique) > 1 and i == len(unique) - 1:
            dist = min([dist, point_z - unique[i - 1]])
        elif len(unique) > 1:
            dist = min([dist, point_z - unique[i - 1], unique[i+1] - point_z])

        random_dist = random.uniform(0., dist)
        pos_sample = point_z + random_dist
        neg_sample = point_z - random_dist

        pos_dd = pos_sample - rd - fornt_bbox_z
        neg_dd = neg_sample - rd - fornt_bbox_z

        if halo:
            z = fornt_bbox_z + rd - neg_dd
            x = (input_file.cx - u) * z / input_file.f  # y on image is x in real world
            y = (input_file.cy - v) * z / input_file.f  # x on image is y in real world

            query_point = o3d.core.Tensor([[x, y, z]], dtype=o3d.core.Dtype.Float32)

            # Compute distance of the query point from the surface
            sdf = scene.compute_distance(query_point).item()
            # rejection sampling
            sdf = rejection_sampling(sdf)
            if sdf >= 0:
                visualize_dict[key].append([rd, neg_dd, sdf])

            z = fornt_bbox_z + rd - pos_dd
            x = (input_file.cx - u) * z / input_file.f  # y on image is x in real world
            y = (input_file.cy - v) * z / input_file.f  # x on image is y in real world

            query_point = o3d.core.Tensor([[x, y, z]], dtype=o3d.core.Dtype.Float32)

            # Compute distance of the query point from the surface
            sdf = scene.compute_distance(query_point).item()
            # rejection sampling
            sdf = rejection_sampling(sdf)
            if sdf >= 0:
                visualize_dict[key].append([rd, pos_dd, sdf])

        elif i % 2 == 0 and not halo:
            z = fornt_bbox_z + rd - neg_dd
            x = (input_file.cx - u) * z / input_file.f  # y on image is x in real world
            y = (input_file.cy - v) * z / input_file.f  # x on image is y in real world

            query_point = o3d.core.Tensor([[x, y, z]], dtype=o3d.core.Dtype.Float32)

            # Compute distance of the query point from the surface
            sdf = scene.compute_distance(query_point).item()
            # rejection sampling
            sdf = rejection_sampling(sdf)
            if sdf >= 0:
                visualize_dict[key].append([rd, neg_dd, sdf])
                visualize_dict[key].append([rd, pos_dd, 0])
        elif i % 2 == 1 and not halo:
            z = fornt_bbox_z + rd - pos_dd
            x = (input_file.cx - u) * z / input_file.f  # y on image is x in real world
            y = (input_file.cy - v) * z / input_file.f  # x on image is y in real world

            query_point = o3d.core.Tensor([[x, y, z]], dtype=o3d.core.Dtype.Float32)

            # Compute distance of the query point from the surface
            sdf = scene.compute_distance(query_point).item()
            # rejection sampling
            sdf = rejection_sampling(sdf)
            if sdf >= 0:
                visualize_dict[key].append([rd, pos_dd, sdf])
                visualize_dict[key].append([rd, neg_dd, 0])
    # value = visualize_dict[key]
    # print(value)


if __name__ == '__main__':
    files = [
        '10f709cecfbb8d59c2536abb1e8e5eab',
        '13d991326c6e8b14fce33f1a52ee07f2',
        '109d55a137c042f5760315ac3bf2c13e',
        '1349b2169a97a0ff54e1b6f41fdd78a',
        '1b4d7803a3298f8477bdcb8816a3fac9',
        '2c1df84ec01cea4e525b133235812833',
        '12ddb18397a816c8948bef6886fb4ac',
        '292d2dda9923752f3e275dc4ab785b9f',
        '1f507b26c31ae69be42930af58a36dce',
        '2c61f0ba3236fe356dae27c417fa89b',
        '2134ad3fc25a6284193a4c984002ed32',
        '17069b6604fc28bfa2f5beb253216d5b',
        '1eaf8db2dd2b710c7d5b1b70ae595e60',
        '10f6e09036350e92b3f21f1137c3c347',
        '15bd6225c209a8e3654b0ce7754570c8',
        '141f1db25095b16dcfb3760e4293e310'
    ]
    categories = ['mug', 'bottle', 'bowl']
    experiment_name = 'new_exp_3'
    with open(f'examples/{experiment_name}/data/dataset_config.json', 'r') as json_file:
        config = json.load(json_file)
    
    for category in categories:
        generated_files = [name.split('_k150')[0] for name in os.listdir(f'examples/{experiment_name}/data/training_data/{category}') if name.endswith(".json") and "train" in name]
        names_json = [name.rstrip('.json') for name in os.listdir(f'examples/{experiment_name}/data/training_data/{category}') if name.endswith(".json") and not "train" in name]
        
        DESTINATION_PATH = f'examples/{experiment_name}/data/training_data/{category}'
        saved_files = 0
        for name_json in names_json:
            view = int(name_json.split('view')[1])
            object_name = name_json.split('_')[0]
            # if not object_name in files:
            #     continue
            SOURCE_PATH = os.path.join(f'examples/{experiment_name}/data/training_data/{category}', name_json + '.json')
            input_file = DepthImageFile(object_name)
            input_file.load(SOURCE_PATH)

            output_file = TrainingFile(SOURCE_PATH, DESTINATION_PATH)

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

            scene = o3d.t.geometry.RaycastingScene()
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

            for key, value in input_file.pixels.items():
                unique = np.unique(value[value!=0])
                x, y = key
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
                elif len(unique) == 1:
                    first_surface = unique[0]
                    rd = first_surface - fornt_bbox_z
                    linspace_sampling(rd, fornt_bbox_z, back_bbox_z, num_samples, unique, visualize_dict, input_file, x, y, scene, True)
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
            # if '_8_' in name_json:
            #     saved_files += 1
            # if saved_files == 8:
            #     break

            # użyć algorytmu z czerwca - ten który działał
            # bez laptopów - 3 kategorie co 5 stopni
            # 4 modele
            # 1. orginalny deepsdf
            # 2. depth deepsdf samplowanie tak jak było w treningu bez prior w query
            # 3. to co z czerwca czyli query z prior
            # 4. słabsze rejection sampling - sampluje liczbę od 0 do 1, jeżeli wartość jest powyżej 0.3 to biorę gaussa, a poniżej liniowe - czyli zapisuję natychmiast
            # * 5. to co w 4 tylko że z prior