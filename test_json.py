import open3d as o3d
import numpy as np
import os
import argparse
import json
from scipy.spatial import KDTree


def add_to_json():
    with open("/home/piotr/Desktop/ProRoc/DeepSDF/examples/splits/PPRAI_test_golem.json", "r") as f:
        files = json.load(f)
    values = sorted(os.listdir("/home/piotr/Desktop/ProRoc/CloudClientServer/input/"))

    for itr, value in enumerate(values):
        filename = value.split('.')
        if value.endswith('.pcd'):
            values[itr] = filename[0]
        else:
            values.remove(value)

    files['Dataset_PPRAI']['golem'] = values

    json_train = json.dumps(files, indent=4)
    
    with open("/home/piotr/Desktop/ProRoc/DeepSDF/examples/splits/PPRAI_test_golem.json", "w") as f:
        f.write(json_train)


def pcd_to_obj():
    input_files = sorted(os.listdir("/home/piotr/Desktop/ProRoc/CloudClientServer/input/"))
    output_files = sorted(os.listdir("/home/piotr/Desktop/ProRoc/CloudClientServer/output/"))
    add_to_json()
    for file in input_files:
        if not file in output_files:
            pcd = o3d.io.read_point_cloud(f"/home/piotr/Desktop/ProRoc/CloudClientServer/input/{input_files[-1]}")
            pcd.estimate_normals()
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = abs(np.mean(distances))
            radius = 3 * avg_dist
            bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))

            bpa_mesh.triangle_normals = o3d.utility.Vector3dVector([])

            with open("/home/piotr/Desktop/ProRoc/DeepSDF/examples/splits/PPRAI_test_golem.json", "r") as f:
                files = json.load(f)
            golem_values = files['Dataset_PPRAI']['golem']

            destination_path = f"/home/piotr/Desktop/ProRoc/DeepSDF/Dataset_PPRAI/golem/{golem_values[-1].split('.')[0]}/models"
            if not os.path.exists(destination_path):
                os.makedirs(destination_path)
            
            o3d.io.write_triangle_mesh(destination_path + '/model_normalized.obj', bpa_mesh, write_vertex_normals=False, write_vertex_colors=False)
            print(destination_path)


def calculate_distance(vertice, center=''):
    if center:
        vertice -= center
        return np.sqrt(sum([np.power(vertice[0], 2), np.power(vertice[1], 2), np.power(vertice[2], 2)]))
    else:
        return np.sqrt(sum([np.power(vertice[0], 2), np.power(vertice[1], 2), np.power(vertice[2], 2)]))

def rescaling(ply, filename):
    obj_path = f"/home/piotr/Desktop/ProRoc/DeepSDF/Dataset_PPRAI/golem/{filename.split('.')[0]}/models/model_normalized.obj"
    # ply_path = '/home/piotr/Desktop/ProRoc/DeepSDF/examples/PPRAI/Reconstructions/2000/Meshes/Dataset_PPRAI_th005/ycb//data-item-1-1.ply'
    
    obj = o3d.io.read_triangle_mesh(obj_path)
    obj_vertices = np.asarray(obj.vertices)

    obj_center = obj_vertices.mean(axis=0)
    print("Obj file center:", obj_center)

    obj_maxDistance = 0
    for v in obj_vertices:
        distance = calculate_distance(np.copy(v), center=list(obj_center))
        obj_maxDistance = max(obj_maxDistance, distance)

    buffer = 1.03
    obj_maxDistance *= buffer
    print("Obj file max distance:", obj_maxDistance)

    # ply = o3d.io.read_triangle_mesh(ply_path)
    ply_vertices = np.asarray(ply.vertices)

    ply_center = ply_vertices.mean(axis=0)
    print("Original ply file center:", ply_center)
    # ply_vertices -= ply_center

    ply_vertices *= obj_maxDistance

    itr = 1
    result = np.array((ply_vertices.shape))
    best_score = 1.
    while itr < 1000:

        tree = KDTree(ply_vertices)
        # Query the KDTree to find the farthest points
        distances, indices = tree.query(obj_vertices)
        # indices = np.flip(indices, axis=0)
        closest_points = ply_vertices[indices]
        # Compute the squared differences between array1 and the closest points
        squared_diff = np.square(obj_vertices - closest_points)
        mean_diff = np.mean(obj_vertices - closest_points, axis=0)
        directions_diff = mean_diff / abs(mean_diff)

        mse = np.mean(squared_diff, axis=0)
        rmse = np.sqrt(mse)
        
        ply_vertices += rmse * directions_diff

        ply.vertices = o3d.utility.Vector3dVector(ply_vertices)

        if np.sum(rmse) < best_score:
            result = np.copy(ply_vertices)
            print(itr)
            itr = 0
            best_score = np.sum(rmse)
            print("Mean Square Error:", mse, rmse)

        itr += 1

    ply.vertices = o3d.utility.Vector3dVector(result)

    print("Translated ply file center:", ply_vertices.mean(0))

    return ply

def ply_to_pcd():
    reconstruct_files = sorted(os.listdir("/home/piotr/Desktop/ProRoc/DeepSDF/examples/PPRAI/Reconstructions/2000/Meshes/Dataset_PPRAI/golem/"))
    output_files = sorted(os.listdir("/home/piotr/Desktop/ProRoc/CloudClientServer/output/"))
    for reconstruct_file in reconstruct_files:
        if not reconstruct_file in output_files:
            reconstructed_ply = o3d.io.read_triangle_mesh(f"/home/piotr/Desktop/ProRoc/DeepSDF/examples/PPRAI/Reconstructions/2000/Meshes/Dataset_PPRAI/golem/{reconstruct_file}")
            ply = rescaling(ply=reconstructed_ply, filename=reconstruct_file)
            points = o3d.utility.Vector3dVector(np.array(ply.vertices))
            pcd = o3d.geometry.PointCloud(points)
            pcd.estimate_normals()
            o3d.io.write_point_cloud(f"/home/piotr/Desktop/ProRoc/CloudClientServer/output/{reconstruct_file.split('.')[0]}.pcd", pcd)

def npz_overview():
    source_path = '/home/piotr/Desktop/ProRoc/DeepSDF/data_PPRAI/SdfSamples/Dataset_PPRAI'  
    classess = os.listdir(source_path)
    list_by_samples = []
    class_neg_value = []
    
    for c in classess:
        avoid = ['golem']
        if not c in avoid:
            continue
        class_path = os.path.join(source_path, c)
        names_list = sorted(os.listdir(class_path))
        # add th for laptop and mug
        for n in names_list:
            npz_file = np.load(os.path.join(class_path, n))
            print(c, n)

            pos_data = npz_file[npz_file.files[0]]
            neg_data = npz_file[npz_file.files[1]]
            # Extract x, y, z, and value from the data
            pos_x = pos_data[:, 0]
            pos_y = pos_data[:, 1]
            pos_z = pos_data[:, 2]
            pos_value = pos_data[:, 3]

            neg_x = neg_data[:, 0]
            neg_y = neg_data[:, 1]
            neg_z = neg_data[:, 2]
            neg_value = neg_data[:, 3]

            print('mean: ', np.mean(neg_value))
            print('std_dev: ', np.std(neg_value))
            print('min: ', min(neg_value))
            print('max: ', max(neg_value))
            print('----------------')
            source = os.path.join(c, n)
            samples = len(pos_data) + len(neg_data)
            list_by_samples.append((source, samples))

            # neg_mean = np.mean(neg_data[:, 3])
            # neg_std = np.std(neg_data[:, 3])

            threshold = 0.018337665125727654  # 0.029343101422744895  # mean:- ,median:-0.018337665125727654 
            sdfs_filtered = [sdfs for sdfs in neg_data if abs(sdfs[3]) <= threshold]
            sdfs_filtered = np.asarray(sdfs_filtered)
            print('mean: ', np.mean(sdfs_filtered[:, 3]))
            print('std_dev: ', np.std(sdfs_filtered[:, 3]))
            print('min: ', min(sdfs_filtered[:, 3]))
            print('max: ', max(sdfs_filtered[:, 3]))
            print('----------------')
            print('----------------')

            data = {"pos": pos_data, "neg": sdfs_filtered}
            np.savez(os.path.join(class_path, n), **data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process pcd files.')
    parser.add_argument(
        "--to_obj",
        "-obj",
        dest="to_obj",
        default=False,
        action="store_true",
        help="Converts pcd file to obj",
        )
    parser.add_argument(
        "--to_pcd",
        "-pcd",
        dest="to_pcd",
        default=False,
        action="store_true",
        help="Converts ply file to pcd",
        )
    parser.add_argument(
        "--npz_th",
        "-npz",
        dest="npz_th",
        default=False,
        action="store_true",
        help="Proceed the npz file thresholding",
        )

    args = parser.parse_args()

    if args.to_obj:
        pcd_to_obj()
    elif args.to_pcd:
        ply_to_pcd()
    elif args.npz_th:
        npz_overview()
