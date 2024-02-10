import open3d as o3d
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from depth.utils import *
from depth.camera import Camera

def save_data_for_sdf(pcd):
    pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)  # set pcd_np as the point cloud points
    roll = np.deg2rad(120)
    rotation_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    pcd_o3d.rotate(rotation_x)

    pcd_points = np.asarray(pcd_o3d.points)
    pcd = scale_and_center(pcd_points, 1)

    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)  # set pcd_np as the point cloud points

    return pcd_o3d

def pcd_to_obj(pcd):
    # pcd.estimate_normals()
    pcd = o3d.geometry.PointCloud(pcd)
    pcd.normals = o3d.utility.Vector3dVector(pcd.points)
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))
    
    return bpa_mesh

def load_data(json_path):
    '''Loading filenames with name of category and .npz extension from .json file.
    Afterwards it loads .obj file with particular name.'''
    loaded_data = {}
    with open(json_path) as j:
        json_data = json.load(j)
    folder_name = list(json_data.keys())[0]

    for category, file_name in json_data[folder_name].items():
        for fn in sorted(file_name):
            name = os.path.join(category, fn)
            file_path = os.path.join(folder_name, name, 'models/model_normalized.obj')
            textured_mesh = o3d.io.read_triangle_mesh(file_path)
            loaded_data[name] = textured_mesh

    return loaded_data

def scale_and_center(points, scale_factor=False):
    points[:, 0] -= np.mean(points[:, 0])
    points[:, 1] -= np.mean(points[:, 1])
    points[:, 2] -= np.mean(points[:, 2])

    max_distance = 0
    for point in points:
        max_distance = max(max_distance, np.linalg.norm(point))

    if scale_factor:
        scale_multiplier = max_distance * scale_factor
        points /= scale_multiplier

    return points

def mesh_preprocessing(input_mesh, test=False):
    '''This function normalize mesh size'''
    mesh = o3d.geometry.TriangleMesh(input_mesh)
    mesh_vertices = mesh.vertices
    mesh_vertices = np.asarray(mesh_vertices)
    mesh_center = np.mean(mesh_vertices, axis=0)

    # Centering
    mesh_vertices -= mesh_center

    # z >= 0
    z_dist = abs(np.min(mesh_vertices[:, 2]))
    mesh_vertices[:, 2] += z_dist

    # Scaling
    max_distance = np.max(mesh_vertices[:, 2]) * 5
    mesh_vertices /= max_distance
    print(np.min(mesh_vertices[:, 2]), np.max(mesh_vertices[:, 2]))

    mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)

    return mesh

def remove_duplicated_mesh(mesh):
    '''Remove meshes with the same vertices'''  # może się przydać normalizacja wektorów i ważenie względem dot product
    mesh = mesh.compute_triangle_normals()
    mesh = mesh.normalize_normals()
    normals = mesh.triangle_normals
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    number_of_duplicates = 0

    triangles_to_delete = []
    for itr, triangle in enumerate(triangles):
        for i, duplicat in enumerate(triangles):
            if i <= itr:
                continue
            elif sorted(list(np.asarray(triangle))) == sorted(list(np.asarray(duplicat))):

                number_of_duplicates += 1

                triangle_center = vec_towards_triangle(triangle=triangle, vertices=vertices)
                mesh_center = np.mean(vertices, axis=0)
                vector = triangle_center - mesh_center
                theta1 = np.dot(vector, normals[itr])

                if theta1 > 0:
                    triangles_to_delete.append(i)
                else:
                    triangles_to_delete.append(itr)
    # # visualization
    # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(voting_points)

    # o3d.visualization.draw_geometries([mesh, origin, pcd])

    # print(f"All triangles: {triangles.shape[0]}",
    #       f"\nDuplicated triangles: {number_of_duplicates}",
    #       f"\nDuplication Ratio: {100*number_of_duplicates/triangles.shape[0]}%")
    
    return triangles_to_delete

def sampling(quantity, max_value):

    mu = 0
    long_sigma = max_value/3
    short_sigma = long_sigma/10
    long_samples = np.random.normal(mu, long_sigma, quantity)
    short_samples = np.random.normal(mu, short_sigma, quantity)
    min_value = -max_value
    long_samples = long_samples[(min_value <= long_samples) & (long_samples <= max_value)]
    result = np.append(long_samples, short_samples)

    # visualisation
    # count, bins, ignored = plt.hist(long_samples, 20, density=True)
    # plt.plot(bins, 1/(long_sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * long_sigma**2) ), linewidth=2, color='r')
    # plt.show()

    return result

def ray_casting(name, textured_mesh, test=False, visualize='', alpha=0):
    print("Processing: ", name)

    # camera parameters
    Fx_depth = 924.348
    Fy_depth = 924.921
    Cx_depth = 646.676
    Cy_depth = 344.145

    width=1280
    height=720

    intrinsic_matrix = o3d.core.Tensor([[Fx_depth, 0, Cx_depth],
                                       [0, Fy_depth, Cy_depth],
                                       [0, 0, 1]])    

    # Rotation angles in radians
    rotation_step = alpha
    roll = np.deg2rad(-rotation_step)  # Rotation around X-axis
    pitch = np.deg2rad(-240)  # Rotation around Y-axis
    yaw = np.deg2rad(270 - rotation_step)  # Rotation around Z-axis

    # Translation values
    tx = -.03
    ty = -.03
    tz = 1.5

    camera = Camera(Fx=Fx_depth, Fy=Fy_depth, Cx=Cx_depth, Cy=Cy_depth, width=width, height=height, intrinsic_matrix=intrinsic_matrix)
    camera.rotate(roll=roll, pitch=pitch, yaw=yaw)
    camera.translate(tx=tx, ty=ty, tz=tz)
    
    # END OF CAMERA SETTINGS

    # file_path = "/home/piotr/Desktop/ProRoc/DeepSDF/Test_PPRAI.v2/ycb_proc1/data-item-1-1/models/model_normalized.obj"
    # textured_mesh = o3d.io.read_triangle_mesh(file_path)

    preprocessed_mesh = mesh_preprocessing(textured_mesh, test=test)
    # duplicated_triangle_idx = remove_duplicated_mesh(preprocessed_mesh)
    # preprocessed_mesh.remove_triangles_by_index(duplicated_triangle_idx)

    mesh = o3d.t.geometry.TriangleMesh.from_legacy(preprocessed_mesh)
    scene = o3d.t.geometry.RaycastingScene()

    scene.add_triangles(mesh)

    mesh_center = np.mean(np.asarray(preprocessed_mesh.vertices), axis=0)
    print("MESH CENTER: ", mesh_center)
    npz_array = np.zeros((1,4))
    depth_images = {}
    itr = 0
    sub_size = 256

    # Stacking depth images
    while True:
        rays = camera.raycasting()

        # Depth image.
        ans = scene.cast_rays(rays)
        img = ans['t_hit'].numpy()
        ids = ans["primitive_ids"].numpy()

        # Change inf values to 0
        img[img == np.inf] = 0
        img = img.astype(np.float32)
        # ROI = np.where(img != 0)
        # y_min = np.min(ROI[0])
        # y_max = np.max(ROI[0])
        # x_min = np.min(ROI[1])
        # x_max = np.max(ROI[1])

        ROI_ids = ids[ids != np.max(ids)]

        if visualize.lower() == 'depth':
            img = img[206:337, 602:652]
            plt.imshow(img, cmap='gray')
            plt.title('Pionhole camera image')
            plt.show()
            # return y_min, y_max, x_min, x_max
        #save depth img
        depth_images[itr] = img

        # exit when there is no mesh to intersect with
        if np.max(img) == 0:
            break

        preprocessed_mesh.remove_triangles_by_index(list(ROI_ids))

        mesh = o3d.t.geometry.TriangleMesh.from_legacy(preprocessed_mesh)

        # Create a scene and add the triangle mesh
        mesh.compute_vertex_normals()
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh)
        
        itr += 1

    keys = list(depth_images.keys())
    print("DICT kEYS: ", keys)

    #Depth images processing
    ROI = np.where(depth_images[keys[0]] != 0)
    ROI_y = ROI[0]
    ROI_x = ROI[1]
    y_center = (np.max(ROI_y) + np.min(ROI_y)) // 2
    x_center = (np.max(ROI_x) + np.min(ROI_x)) // 2
    print("x, y center:", x_center, y_center)

    y_start = y_center - sub_size/2
    x_start = x_center - sub_size/2

    # print(len(ROI_y))
    max_dist = 0
    min_dist = 1
    mean_z = 0
    for x, y in zip(ROI_x, ROI_y):
        depth_values = []
        for depth_image in depth_images.values():
            depth_values.append(depth_image[y][x])
        last_surface_idx = np.max(np.nonzero(depth_values))
        
        if test:
            max_z = depth_values[0] + 1.5e-2  # np.max(depth_images[keys[0]])
            min_z = max_z - 1
            distance = depth_values[0]
            if distance == 0:
                continue

            #Sampling front
            samples_front = sampling(2, 0.1)
            z = depth_values[0] + samples_front
            z = z[(z <= max_z) & (z >= min_z)]
            sdf = depth_values[0] - z
            point_data = np.column_stack((np.full(sdf.shape, x),   
                                        np.full(sdf.shape, y),
                                        np.full(sdf.shape, z),
                                        sdf))
            npz_array = np.concatenate((npz_array, point_data), axis=0) 
            mean_z += np.mean(depth_values[0])

        else:
            # if last_surface_idx%2 == 0 or last_surface_idx%2 == 1:
            last_surface_dist = depth_values[last_surface_idx]
            distance = last_surface_dist - depth_values[0]
            if distance == 0:
                continue
            max_dist = max(max_dist, distance)
            min_dist = min(min_dist, distance)
            # continue

            #Sampling front
            samples_front = sampling(8, 1-distance/2)
            samples_front = samples_front[samples_front < distance/2]
            z = depth_values[0] + samples_front
            sdf = depth_values[0] - z
            point_data = np.column_stack((np.full(sdf.shape, x),   
                                        np.full(sdf.shape, y),
                                        np.full(sdf.shape, z),
                                        sdf))
            npz_array = np.concatenate((npz_array, point_data), axis=0) 

            #Sampling back
            samples_back = sampling(8, 1-distance/2)
            samples_back = samples_back[samples_back > -distance/2]
            z = last_surface_dist + samples_back
            sdf = z - last_surface_dist
            point_data = np.column_stack((np.full(sdf.shape, x),   
                                        np.full(sdf.shape, y),
                                        np.full(sdf.shape, z),
                                        sdf))
            npz_array = np.concatenate((npz_array, point_data), axis=0) 

            mean_z += (np.mean(last_surface_dist) + np.mean(depth_values[0])) / 2
    
    mean_z /= len(list(zip(ROI_x, ROI_y)))
    for i in range(sub_size):
        x = x_start
        for j in range(sub_size):
            abc = (x, y_start)
            defg = list(zip(ROI_x, ROI_y))
            # if not abc in defg:
            z = np.linspace(mean_z - 1, mean_z + 1, num=16)
            sdf = 2
            point_data = np.column_stack((np.full(z.shape, x),   
                                        np.full(z.shape, y_start),
                                        z,
                                        np.full(z.shape, sdf)))
            npz_array = np.concatenate((npz_array, point_data), axis=0) 
            x += 1
        print(y_start)
        y_start += 1
    
    print("MEAN Z, MIN DIST, MAX DIST: :", mean_z, min_dist, max_dist)
    # print(depth_images.keys())
    npz_array = np.delete(npz_array, 0, axis=0)
    
    # depth image to point cloud
    z = npz_array[:, 2]
    # print('x', np.min(npz_array[:, 0]), np.max(npz_array[:, 0]))
    # print('y', np.min(npz_array[:, 1]), np.max(npz_array[:, 1]))
    # print('z', np.min(npz_array[:, 2]), np.max(npz_array[:, 2]))

    x = (Cx_depth - npz_array[:, 0]) * z / Fx_depth  # y on image is x in real world
    y = (Cy_depth - npz_array[:, 1]) * z / Fy_depth  # x on image is y in real world

    npz_result = np.column_stack([x, y, z, npz_array[:, 3]])
    dist = np.linalg.norm(npz_array[0, :3] - npz_array[1, :3])
    print("LINALG DIST, 0 IDX Z, 1 IDX Z: ", dist, abs(npz_array[0, 3] - npz_array[1, 3]))
    print("CENTER: ", np.mean(npz_result[:, :3], axis=0))
    npz_result[:, :3] -= np.mean(npz_result[:, :3], axis=0)
    print("CENTER AFTER SUBTRACTION: ", np.mean(npz_result[:, :3], axis=0))
    
    pos_data = npz_result[npz_result[:, 3] >= 0]
    neg_data = npz_result[npz_result[:, 3] <= 0]
    print("POS DATA SHAPE, NEG DATA SHAPE: ", pos_data.shape, neg_data.shape)

    npz_data = {"pos": pos_data, "neg": neg_data}

    pcd = np.column_stack((pos_data[:, 0], pos_data[:, 1], pos_data[:, 2]))
    pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)  # set pcd_np as the point cloud points

    # pcd_o3d = save_data_for_sdf(pcd)
    # bpa_mesh = pcd_to_obj(pcd_o3d)
    destination_path = f"dataset_YCB_test/magisterka_sdf/{name.split('/')[-1]}_new/models"
    print(destination_path)
    # if not os.path.exists(destination_path):
        # os.makedirs(destination_path)
    # o3d.io.write_triangle_mesh(os.path.join(destination_path, "model_normalized.obj"), bpa_mesh, write_vertex_normals=False, write_vertex_colors=False)


    # Visualize:
    if visualize.lower() == 'cloud':
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd_o3d, origin])


    destination_filename = f"/home/piotr/Desktop/ProRoc/DeepSDF/magisterka/trening/{name.split('/')[-1]}_train.pcd"
    dest2 = f"dataset_YCB_train/depth_norm/depth_{name.split('/')[-1]}.npz"
    # o3d.io.write_point_cloud(destination_filename, pcd_o3d)
    print(f"SAVED POINT CLOUD: {destination_filename}")

    if test:
        destination_path = f"data_YCB/SdfSamples/dataset_YCB_test/{name.split('/')[0]}"
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        np.savez(os.path.join(destination_path, name.split('/')[-1]+".npz"), **npz_data)
        print(f"SAVED NPZ: {os.path.join(destination_path, name.split('/')[-1]+'.npz')}")
    else:
        destination_path = f"data_YCB/SdfSamples/dataset_YCB_train/{name.split('/')[0]}"
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        np.savez(os.path.join(destination_path, name.split('/')[-1]+".npz"), **npz_data)
        print(f"SAVED NPZ: {os.path.join(destination_path, name.split('/')[-1]+'.npz')}")

    exit(777)

    # ZAPISAĆ CHMURĘ PUNKTÓW, ALE BEZ KOLUMNY SDF. WCZYTAĆ W FUNKCJI SKRYPTU MESH.PY

if __name__ == '__main__':
    json_path = 'examples/splits/sensors_test.json'
    data = load_data(json_path)
    y_bot, x_left = 9999, 9999
    y_top, x_right = 0, 0
    for name, mesh in data.items():
        ray_casting(name, mesh, True, 'cloud')
    #     y_min, y_max, x_min, x_max = ray_casting(name, mesh, False, '')
    #     y_bot = min(y_min, y_bot)
    #     y_top = max(y_max, y_top)
    #     x_left = min(x_left, x_min)
    #     x_right = max(x_right, x_max)

    # print(y_bot, y_top, x_left, x_right)
        

    #     print(name)
    #     pcd = o3d.io.read_point_cloud(
    #         os.path.join("/home/piotr/Desktop/ProRoc/DeepSDF/magisterka/trening/", name.split('/')[-1]+"_train.pcd")
    #         )
    #     o3d.visualization.draw_geometries([pcd])
    #     decision = input("Good? [y/n]: ")
    #     if decision.lower() == 'y':
    #         good.append(name)

    # print(*good)


# do testu depth/9cec36de93bb49d3f07ead639233915e, depth/883ace957dbb32e7846564a8a219239b,
#  depth/62451f0ab130709ef7480cb1ee830fb9, depth/29b6f9c7ae76847e763c517ce709a8cc

# do treningu depth/1ef68777bfdb7d6ba7a07ee616e34cd7 depth/21239b0cafa13526cafb7c62b057a234 
# depth/216adefa94f25d7968a3710932407607 depth/22249179e88b0502846564a8a219239b depth/24feb92770933b1663995fb119e59971 
# depth/2618100a5821a4d847df6165146d5bbd depth/26a8d94392d86aa0940806ade53ef2f depth/26e6f23bf6baea05fe5c8ffd0f5eba47 
# depth/2722bec1947151b86e22e2d2f64c8cef depth/27b9f07da9217c89ef026123226f5519 depth/2976131e4621231655bf395569b6fd5 
# depth/29cad22b0fc6c9985f16c469ffeb982e depth/2bbd2b37776088354e23e9314af9ae57 depth/917fd3e68de79bf69ece21a0b0786a69 
# depth/b95559b4f2146b6a823177feb9bf5114