import open3d as o3d
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from depth_utils import *


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

def mesh_preprocessing(input_mesh, test=False):
    '''This function normalize mesh size'''
    mesh = o3d.geometry.TriangleMesh(input_mesh)
    mesh_vertices = mesh.vertices
    mesh_vertices = np.asarray(mesh_vertices)
    mesh_center = np.mean(mesh_vertices, axis=0)

    # Centering
    mesh_vertices -= mesh_center

    # Scaling
    max_distance = 0
    for vert in mesh_vertices:
        distance = magnitude(vert)
        max_distance = max(max_distance, distance)
    max_distance *= 5
    # if not test:
    mesh_vertices /= max_distance
    x_dist = abs(np.min(mesh_vertices[:, 0]))
    y_dist = abs(np.min(mesh_vertices[:, 1]))
    z_dist = abs(np.min(mesh_vertices[:, 2]))
    print(z_dist)
    # Moving - for objects lying on side add min(mesh_vertices[:, 2])
    # mesh_vertices[:, 0] += 0.35
    # mesh_vertices[:, 1] += 0.
    # mesh_vertices[:, 2] += 0.05 + z_dist


    # TU BĘDZIE AUGMENTACJA augment()
    # mesh_vertices += np.array([0.49066027, -0.02950335,  0.21953733])

    # radians = -np.pi/2
    # rotation_matrix = np.array([[np.cos(radians), -np.sin(radians), 0],
    #         [np.sin(radians), np.cos(radians), 0],
    #         [0, 0, 1]])
    # mesh.rotate(rotation_matrix, center=[0,0,0])

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
    result = np.append(long_samples, 0)

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
    # roll = np.deg2rad(0)  # Rotation around X-axis
    # pitch = np.deg2rad(-240)  # Rotation around Y-axis
    # yaw = np.deg2rad(270)  # Rotation around Z-axis

    rotation_step = alpha
    roll = np.deg2rad(-rotation_step)  # Rotation around X-axis
    pitch = np.deg2rad(-240)  # Rotation around Y-axis
    yaw = np.deg2rad(270 - rotation_step)  # Rotation around Z-axis

    # Translation values
    tx = -.03
    ty = -.03
    tz = 1.5

    # Create rotation matrices for each axis
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

    # Create the extrinsic matrix
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = [tx, ty, tz]
    
    # END OF CAMERA SETTINGS

    preprocessed_mesh = mesh_preprocessing(textured_mesh, test=test)
    # duplicated_triangle_idx = remove_duplicated_mesh(preprocessed_mesh)
    # preprocessed_mesh.remove_triangles_by_index(duplicated_triangle_idx)

    mesh = o3d.t.geometry.TriangleMesh.from_legacy(preprocessed_mesh)
    scene = o3d.t.geometry.RaycastingScene()

    scene.add_triangles(mesh)

    npz_array = np.zeros((1,4))
    depth_images = {}
    itr = 0

    sub_img_width, sub_img_height = 350, 350
    sub_img_x, sub_img_y = 0, 0
    last_distance_diff = 0.

    # Stacking depth images
    while True:
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
            width_px=width,
            height_px=height
        )

        # Matrix (height, width) of intersections
        intersect = scene.count_intersections(
            rays
        ).numpy()

        # Depth image.
        ans = scene.cast_rays(rays)
        img = ans['t_hit'].numpy()
        ids = ans["primitive_ids"].numpy()

        # Change inf values to 0
        img[img == np.inf] = 0
        img = img.astype(np.float32)

        ROI_ids = ids[ids != np.max(ids)]

        if visualize.lower() == 'depth':
            plt.imshow(img, cmap='gray')
            plt.title(name)
            plt.show()
        
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
    print(keys)
    #Depth images processing
    ROI = np.where(depth_images[keys[0]] != 0)
    ROI_y = ROI[0]
    ROI_x = ROI[1]
    # print(len(ROI_y))
    max_dist = 0
    min_dist = 1
    for x, y in zip(ROI_x, ROI_y):
        depth_values = []
        for depth_image in depth_images.values():
            depth_values.append(depth_image[y][x])
        last_surface_idx = np.max(np.nonzero(depth_values))
        
        if test:
            distance=1e-2
            #Sampling front
            samples_front = sampling(1, distance/2)
            z = depth_values[0] + samples_front
            sdf = depth_values[0] - z
            point_data = np.column_stack((np.full(sdf.shape, x),   
                                        np.full(sdf.shape, y),
                                        np.full(sdf.shape, z),
                                        sdf))
            npz_array = np.concatenate((npz_array, point_data), axis=0)
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
            samples_front = sampling(1, distance/2)
            z = depth_values[0] + samples_front
            sdf = depth_values[0] - z
            point_data = np.column_stack((np.full(sdf.shape, x),   
                                        np.full(sdf.shape, y),
                                        np.full(sdf.shape, z),
                                        sdf))
            npz_array = np.concatenate((npz_array, point_data), axis=0) 

            #Sampling back
            samples_back = sampling(1, distance/2)
            z = last_surface_dist + samples_back
            sdf = z - last_surface_dist
            point_data = np.column_stack((np.full(sdf.shape, x),   
                                        np.full(sdf.shape, y),
                                        np.full(sdf.shape, z),
                                        sdf))
            npz_array = np.concatenate((npz_array, point_data), axis=0) 
        # else:
        #     duplicated_triangle = False
        #     for i in range(last_surface_idx+1):
        #         try:
        #             distance = depth_values[i+1] - depth_values[i]
        #         except:
        #             distance = depth_values[i] - depth_values[i-1]
                    
        #         if distance == 0:
        #             duplicated_triangle = True
        #             continue
        #         elif distance < 0:
        #             distance = depth_values[i] - depth_values[i-1]

        #         samples = sampling(50, distance/2)

        #         z = depth_values[i] + samples

        #         if i%2 == 0 or duplicated_triangle:
        #             sdf = depth_values[i] - z
        #             duplicated_triangle = False
        #         else:
        #             sdf = z - depth_values[i]

        #         for j, zi in enumerate(z):
        #             if (zi < depth_values[i]) and (sdf[j] < 0):
        #                 sdf[j] *= -1
        #                 # print(f"HERE WE ARE, z: {zi}, depth: {depth_values[i]}, sdf: {sdf[j]}")
        #                 # print(f"distance: {distance}, i: {i}, whole depth: {depth_values}")

        #         point_data = np.column_stack((np.full(sdf.shape, x),   
        #                                     np.full(sdf.shape, y),
        #                                     np.full(sdf.shape, z),
        #                                     sdf))
        #         npz_array = np.concatenate((npz_array, point_data), axis=0) 
    # exit(777)
    print(min_dist, max_dist)
    # print(depth_images.keys())
    npz_array = np.delete(npz_array, 0, axis=0)
    
    # depth image to point cloud
    z = npz_array[:, 2]
    # print('x', np.min(npz_array[:, 0]), np.max(npz_array[:, 0]))
    # print('y', np.min(npz_array[:, 1]), np.max(npz_array[:, 1]))
    # print('z', np.min(npz_array[:, 2]), np.max(npz_array[:, 2]))

    x = (Cx_depth - npz_array[:, 0]) * z / Fx_depth  # y on image is x in real world
    y = (Cy_depth - npz_array[:, 1]) * z / Fy_depth  # x on image is y in real world

    # # NORMALIZATION
    # sub_img_x = (np.min(npz_array[:, 0]) + np.max(npz_array[:, 0])) // 2
    # half_width = sub_img_width // 2
    # min_x = np.min((sub_img_x - half_width - Cx_depth) * z / Fx_depth)
    # max_x = np.max((sub_img_x + half_width - Cx_depth) * z / Fx_depth)

    # sub_img_y = (np.min(npz_array[:, 1]) + np.max(npz_array[:, 1])) // 2
    # half_height = sub_img_height / 2
    # min_y = np.min((sub_img_y - half_height - Cy_depth) * z / Fy_depth)
    # max_y = np.max((sub_img_y + half_height - Cy_depth) * z / Fy_depth)

    # y_norm = y[(y > min_y) & (y < max_y)]
    # x_norm = x[(x > min_x) & (x < max_x)]

    # z_norm = (z - 1) / (2 - 1)

    npz_result = np.column_stack([x, y, z, npz_array[:, 3]])
    dist = np.linalg.norm(npz_array[0, :3] - npz_array[1, :3])
    print(dist, npz_array[0, 3], npz_array[1, 3])
    print(np.mean(npz_result[:, :3], axis=0))
    npz_result[:, :3] -= np.mean(npz_result[:, :3], axis=0)
    print(np.mean(npz_result[:, :3], axis=0))

    pos_data = npz_result[npz_result[:, 3] > 0]
    neg_data = npz_result[npz_result[:, 3] < 0]
    print(pos_data.shape, neg_data.shape)
    npz_data = {"pos": pos_data, "neg": neg_data}
    # if test:
        # np.savez(f"data_YCB/SdfSamples/dataset_YCB_test/ycb_z/{name.split('/')[-1]}.npz", **npz_data)
    # else:
        # np.savez(f"data_YCB/SdfSamples/dataset_YCB_train/depth/{name.split('/')[-1]}_{rotation_step}.npz", **npz_data)


    pcd = np.column_stack((pos_data[:, 0], pos_data[:, 1], pos_data[:, 2]))
    pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)  # set pcd_np as the point cloud points

    # Visualize:
    if visualize.lower() == 'cloud':
        # radians = -np.pi/2
        # rotation_matrix = np.array([[np.cos(radians), -np.sin(radians), 0],
        #         [np.sin(radians), np.cos(radians), 0],
        #         [0, 0, 1]])
        # textured_mesh.rotate(rotation_matrix, center=[0,0,0])
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd_o3d, origin])

    destination_filename = f"/home/piotr/Desktop/ProRoc/DeepSDF/ycb1/depth_to_pcd/{name.split('/')[-1]}.pcd"
    dest2 = f"dataset_YCB_train/depth_norm/depth_{name.split('/')[-1]}.npz"
    # o3d.io.write_point_cloud(destination_filename, pcd_o3d)
    # print(f"SAVED POINT CLOUD: {destination_filename}")
    
    # exit(777)


if __name__ == '__main__':
    # np.set_printoptions(threshold=sys.maxsize)

    json_path = 'examples/splits/YCB_depth_train.json'
    data = load_data(json_path)

    for name, mesh in data.items():
        angles = [0, 90, 180, 270]
        for angle in angles:
            ray_casting(name, mesh, False, 'cloud', alpha=angle)


