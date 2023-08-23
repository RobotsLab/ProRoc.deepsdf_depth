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
            file_path = os.path.join(folder_name, name, 'model_normalized.obj')
            textured_mesh = o3d.io.read_triangle_mesh(file_path)
            loaded_data[name] = textured_mesh

    return loaded_data

def mesh_preprocessing(input_mesh):
    '''This function normalize mesh size'''
    mesh = o3d.geometry.TriangleMesh(input_mesh)
    mesh_vertices = mesh.vertices
    mesh_vertices = np.asarray(mesh_vertices)
    mesh_center = np.mean(mesh_vertices, axis=0)

    # Centering
    # mesh_vertices -= mesh_center

    # Scaling
    max_distance = 0
    for vert in mesh_vertices:
        distance = magnitude(vert)
        max_distance = max(max_distance, distance)
    max_distance *= 5
    mesh_vertices /= max_distance

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

    print(f"All triangles: {triangles.shape[0]}",
          f"\nDuplicated triangles: {number_of_duplicates}",
          f"\nDuplication Ratio: {100*number_of_duplicates/triangles.shape[0]}%")
    
    return triangles_to_delete

def sampling(quantity):

    mu = 0
    sigma = 0.0005 # mean and standard deviation
    samples = np.random.normal(mu, sigma, quantity)

    return samples

def ray_casting(name, textured_mesh, visualize=''):
    print("Processing: ", name)

    # camera parameters
    Fx_depth = 924.348
    Fy_depth = 924.921
    Cx_depth = 646.676
    Cy_depth = 344.145
    s = Fx_depth/Fy_depth
    width=1280
    height=720

    intrinsic_matrix = o3d.core.Tensor([[Fx_depth, 0, Cx_depth],
                                       [0, Fy_depth, Cy_depth],
                                       [0, 0, 1]])    

    # Rotation angles in radians
    roll = np.deg2rad(0)  # Rotation around X-axis
    pitch = np.deg2rad(-240)  # Rotation around Y-axis
    yaw = np.deg2rad(270)  # Rotation around Z-axis

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

    preprocessed_mesh = mesh_preprocessing(textured_mesh)
    duplicated_triangle_idx = remove_duplicated_mesh(preprocessed_mesh)
    preprocessed_mesh.remove_triangles_by_index(duplicated_triangle_idx)

    mesh = o3d.t.geometry.TriangleMesh.from_legacy(preprocessed_mesh)
    scene = o3d.t.geometry.RaycastingScene()

    scene.add_triangles(mesh)

    npz_array = np.zeros((1,4))
    surfaces = {}
    first_iteration = True
    mask = np.zeros((height, width))
    itr = 0

    sub_img_width, sub_img_height = 350, 350
    sub_img_x, sub_img_y = 0, 0

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

        # print(len(np.where(intersect == 1)[0]), len(np.where(intersect == 2)[0]), len(np.where(intersect == 3)[0]), len(np.where(intersect == 4)[0]), len(np.where(intersect > 4)[0]))

        # Depth image.
        ans = scene.cast_rays(rays)
        img = ans['t_hit'].numpy()
        ids = ans["primitive_ids"].numpy()

        # Change inf values to 0
        img[img == np.inf] = 0
        img = img.astype(np.float32)

        # ploting depth image
        if first_iteration:
            odd_rays = len(np.where(intersect%2 != 0)[0])
            odd_rays_ratio = 100*odd_rays/len(np.where(intersect != 0)[0])
            print(f"Number of rays with odd amount of intersections: {odd_rays}",
                f"\nIntersection Ratio: {round(odd_rays_ratio, 4)}%")  
            if odd_rays_ratio > 2:
                print(f"Object {name.title()} has been rejected because of Intersection Ratio")
                return None
            
            intersect[intersect%2 != 0] = 0
            mask = intersect
            itr = np.max(intersect)

            if visualize.lower() == 'depth':
                plt.imshow(img, cmap='gray')
                plt.title(name)
                plt.show()

            first_iteration = False

        img[mask == 0] = 0
        ids[mask == 0] = np.max(ids)

        # exit when there is no mesh to intersect with
        if np.max(img) == 0:
            break

        # subimage
        ROI = np.where(img != 0)

        ROI_ids = ids[ids != np.max(ids)]
        ROI_ids_x = np.where(ids != np.max(ids))[1]
        ROI_ids_y = np.where(ids != np.max(ids))[0]

        # sub_dims = np.where(img != 0)
        # sub_img_x = (np.min(sub_dims[1]) + np.max(sub_dims[1])) // 2 - sub_img_width // 2
        # sub_img_y = (np.min(sub_dims[0]) + np.max(sub_dims[0])) // 2 - sub_img_height // 2
        # sub_img = np.array(img[sub_img_y:sub_img_y+sub_img_height, sub_img_x:sub_img_x+sub_img_width])
        # print('x: ', np.min(sub_dims[0]), np.max(sub_dims[0]), 'y: ', np.min(sub_dims[1]), np.max(sub_dims[1]))
        # plt.imshow(sub_img, cmap='gray')
        # plt.show()

        # normals = ans["primitive_normals"].numpy()
        
        #save depth img
        surfaces[itr] = img

        if len(surfaces.keys()) >= 1:
            recent_image = surfaces[list(surfaces.keys())[-1]]
            # first_image = surfaces[list(surfaces.keys())[0]]
            # distance_diff = img - recent_image
        
            for x, y in zip(ROI_ids_x, ROI_ids_y):
                samples = sampling(50)

                # if distance_diff[y][x] == 0:
                    # continue

                z = img[y][x] + samples
                # sdf_front = recent_image[y][x] - z
                # sdf_back = z - img[y][x]
                # sdf = np.minimum(sdf_front, sdf_back)
                sdf = z - img[y][x]
                point_data = np.column_stack((np.full(sdf.shape, x),   
                                        np.full(sdf.shape, y),
                                        np.full(sdf.shape, z),
                                        sdf))
                npz_array = np.concatenate((npz_array, point_data), axis=0)            

        preprocessed_mesh.remove_triangles_by_index(list(ROI_ids))

        mesh = o3d.t.geometry.TriangleMesh.from_legacy(preprocessed_mesh)

        # Create a scene and add the triangle mesh
        mesh.compute_vertex_normals()
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh)
        
        # o3d.visualization.draw_geometries([preprocessed_mesh])

        itr -= 1

    print(surfaces.keys())
    npz_array = np.delete(npz_array, 0, axis=0)
    
    # depth image to point cloud
    z = npz_array[:, 2]
    print('x', np.min(npz_array[:, 0]), np.max(npz_array[:, 0]))
    print('y', np.min(npz_array[:, 1]), np.max(npz_array[:, 1]))
    print('z', np.min(npz_array[:, 2]), np.max(npz_array[:, 2]))

    x = (Cx_depth - npz_array[:, 0]) * z / Fx_depth  # y on image is x in real world
    y = (Cy_depth - npz_array[:, 1]) * z / Fy_depth  # x on image is y in real world
    pcd = np.column_stack((x, y, z))
    pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)  # set pcd_np as the point cloud points

    # Visualize:
    if visualize.lower() == 'cloud':
        # radians = -np.pi/2
        # rotation_matrix = np.array([[np.cos(radians), -np.sin(radians), 0],
        #         [np.sin(radians), np.cos(radians), 0],
        #         [0, 0, 1]])
        # textured_mesh.rotate(rotation_matrix, center=[0,0,0])
        # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        o3d.visualization.draw_geometries([pcd_o3d])

    destination_filename = f"/home/piotr/Desktop/ProRoc/DeepSDF/ycb1/depth_to_pcd/{name.split('/')[-1]}.pcd"
    # o3d.io.write_point_cloud(destination_filename, pcd_o3d)
    print(f"SAVED POINT CLOUD: {destination_filename}")

    exit(7)


if __name__ == '__main__':
    # np.set_printoptions(threshold=sys.maxsize)

    json_path = 'examples/splits/YCB_depth_image_train.json' # depth_image_train.json'
    data = load_data(json_path)
    # preprocessed_files = os.listdir("/home/piotr/Desktop/ProRoc/DeepSDF/ycb1/depth_to_pcd/")
    # visualize_pcd()

    for name, mesh in data.items():
        ray_casting(name, mesh, 'cloud')
        # if 'bottle' in name and '_x0' in name: #  ('mug' in name and not '_' in name) or ('bottle' in name and '_x0' in name):   
            # print(name)  
            # radians = np.deg2rad(90)
            # rotation_matrix = np.array([[1, 0, 0],
            # [0, 0, 1],
            # [0, 1, 0]])
            # mesh.rotate(rotation_matrix, center=[0,0,0])
            # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
            # filename = name.split("/")[-1]
            # filename = filename.split("_")[0]
            # new_path = os.path.join('dataset_YCB_train/bottle_z', filename)
            # if not os.path.exists(new_path):
            #     os.mkdir(new_path)
            #     o3d.visualization.draw_geometries([mesh, origin])       
            #     # yes = input("Do you want to save it?[y/n]")
            #     # if yes.lower() == 'y':
            #     o3d.io.write_triangle_mesh(f"{os.path.join(new_path, 'model_normalized')}.obj", mesh)
