import open3d as o3d
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import sys


def load_data(json_path):
    '''Loading filenames with name of category and .npz extension from .json file.
    Afterwards it loads .obj file with particular name.'''
    loaded_data = {}
    with open(json_path) as j:
        json_data = json.load(j)
    folder_name = list(json_data.keys())[0]

    for category, file_name in json_data[folder_name].items():
        for fn in file_name:
            name = os.path.join(category, fn)
            file_path = os.path.join(folder_name, name, 'models/model_normalized.obj')
            textured_mesh = o3d.io.read_triangle_mesh(file_path)
            loaded_data[name] = textured_mesh

    return loaded_data

def magnitude(vector):
    '''Calculate length of vector of every dimension'''
    return np.sqrt(sum(pow(element, 2) for element in vector))

def vec_towards_triangle(triangle, vertices):
    '''Find vector directed towards center of triangle.'''
    vert1, vert2, vert3 = triangle[0], triangle[1], triangle[2]
    target_point = np.mean([vertices[vert1], vertices[vert2], vertices[vert3]], axis=0)

    return target_point

def mesh_preprocessing(mesh):
    '''This function normalize mesh size'''
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
    mesh_vertices /= max_distance
    # print('Max distance:', max_distance)

    mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
    print(mesh_center,np.mean(mesh.vertices, axis=0))

    return mesh

def remove_duplicated_mesh(mesh):
    '''Remove meshes with the same vertices'''
    mesh = mesh.compute_triangle_normals()
    mesh = mesh.normalize_normals()
    normals = mesh.triangle_normals
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    number_of_duplicates = 0
    number_of_votes = 21

    y_samples = np.linspace(-0.6, 0.6, number_of_votes)
    voting_points = np.zeros((1, 3))
    for sample in y_samples:
        voting_points = np.append(voting_points, np.array([0, sample, 0])[np.newaxis, :], axis=0)
    voting_points = np.delete(voting_points, 0, axis=0)

    triangles_to_delete = []
    for itr, triangle in enumerate(triangles):
        for i, duplicat in enumerate(triangles):
            if i <= itr:
                continue
            elif sorted(list(np.asarray(triangle))) == sorted(list(np.asarray(duplicat))):
                # print(f"They are the same: {itr}. {triangle} and {i}. {duplicat}")
                # print(f"Normals 1: {normals[itr]}, Normals 2: {normals[i]}")

                triangle_center = vec_towards_triangle(triangle=triangle, vertices=vertices)
                votes = 0
                for v_point in voting_points[0:, :]:
                    vector = triangle_center - v_point

                    theta1 = np.dot(vector, normals[itr])
                    theta2 = np.dot(vector, normals[i]) 

                    # zbieranie głosów za tym że normalna jest zgodna, tzn. skierowana na zewnątrz obiektu
                    if theta1 > 0:
                        votes += 1
                    
                    # print(v_point, vector, theta1, theta2)  # 1 = pokrywa się z wektorem, -1 = jest skierowany przeciwnie
                # print('itr:', itr, 'i:', i)
                # print('votes:', votes)
                if votes > number_of_votes/2:
                    triangles_to_delete.append(i)
                else:
                    triangles_to_delete.append(itr)

                # zliczanie powtarzających się meshy
                if i > itr:
                    number_of_duplicates += 1

    # # visualization
    # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(voting_points)

    # o3d.visualization.draw_geometries([mesh, origin, pcd])

    return triangles_to_delete
    

def ray_casting(name, textured_mesh):
    # print(name)
    
    preprocessed_mesh = mesh_preprocessing(textured_mesh)
    duplicated_triangle_idx = remove_duplicated_mesh(preprocessed_mesh)
    preprocessed_mesh.remove_triangles_by_index(duplicated_triangle_idx)
    
    # np.set_printoptions(threshold=sys.maxsize)

    mesh = o3d.t.geometry.TriangleMesh.from_legacy(preprocessed_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    # camera position from .xml
    # v1 = 0.666247
    # v2 = 0.344917 
    # v3 = 0.487054

    width=640
    height=480

    npz_array = np.zeros((1,4))
    surfaces = {}
    first_img = True

    while True:
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            fov_deg=60,
            center=[0.,0.,0.],
            eye=[0,1.5,1.5],
            up=[0,-1,0],
            width_px=width,
            height_px=height,
        )
        intersect = scene.count_intersections(
            rays
        ).numpy()

        # We can directly pass the rays tensor to the cast_rays function.
        ans = scene.cast_rays(rays)
        img = ans['t_hit'].numpy()

        # points out of ROI values == 0
        img[img == np.inf] = 0
        img = img.astype(np.float32)
        # print(img, img.shape)

        # ploting depth image
        if first_img:
            plt.imshow(img, cmap='gray')
            plt.title(name)
            first_img = False
        # plt.show()
        # break

        # exit when there is no mesh to intersect with
        if np.max(intersect) == 0:
            break

        #find hit triangle
        ids = ans["primitive_ids"].numpy()
        ROI_ids = ids[ids != np.max(ids)]
        ROI_ids_x = np.where(ids != np.max(ids))[0]
        ROI_ids_y = np.where(ids != np.max(ids))[1]

        # normals = ans["primitive_normals"].numpy()

        if surfaces:
            recent_image = surfaces[list(surfaces.keys())[-1]]
            distance_diff = img - recent_image

            for point in zip(ROI_ids_x, ROI_ids_y):
                x, y = point[0], point[1]

                samples = np.linspace(0, distance_diff[x][y], num=2)
                if distance_diff[x][y] == 0:
                    continue
                for sample in samples:
                    point = list(point)
                    z = img[x][y] - sample
                    sdf_front = recent_image[x][y] - z
                    sdf_back = z - img[x][y]
                    sdf = max(sdf_front, sdf_back)
                    result = np.array([x, y, z, sdf])[np.newaxis, :]

                    npz_array = np.append(npz_array, result, axis=0)

        #save depth img
        surfaces[np.max(intersect)] = img

        preprocessed_mesh.remove_triangles_by_index(list(ROI_ids))

        mesh = o3d.t.geometry.TriangleMesh.from_legacy(preprocessed_mesh)

        # Create a scene and add the triangle mesh
        mesh.compute_vertex_normals()
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh)
        
        # o3d.visualization.draw_geometries([preprocessed_mesh])

    npz_array = np.delete(npz_array, 0, axis=0)
    # print(npz_array)

    # print(npz_array[:, :3])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(npz_array[:, 0], npz_array[:, 1], npz_array[:, 2], c=npz_array[:, 3], cmap='viridis', marker='.')
    plt.title(name)
    plt.show()

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(npz_array[:, :3])
    # o3d.visualization.draw_geometries([pcd])

    exit(7)


if __name__ == '__main__':
    json_path = 'examples/splits/YCB_bottle_mug_train.json'
    data = load_data(json_path)
    for name, mesh in data.items():
        if 'bottle' in name and '_x0' in name: #  ('mug' in name and not '_' in name) or ('bottle' in name and '_x0' in name):
            ray_casting(name, mesh)
    
    




    # first = True
    # first_img = np.zeros((height, width))
    # long_dist = np.zeros((height, width))

    # mask = long_dist < img
    # long_dist[mask] = img[mask]

    # if first:
    #     first_img = img[:]
    #     first = False

    # KDTree APPROACH
    # print(first_img[first_img != 0], first_img[first_img != 0].shape)
    # print(long_dist[long_dist != 0], long_dist[long_dist != 0].shape)
    # long_dist_x = np.where(long_dist != 0)[0]
    # long_dist_y = np.where(long_dist != 0)[1]      

    # tree = KDTree(mesh_vertices)

    # for i in range(long_dist[long_dist != 0].shape[0]):
    #     x, y = long_dist_x[i], long_dist_y[i]
    #     front_img = first_img[x][y]
    #     back_img = long_dist[x][y]
    #     # print(x, y, front_img, back_img) 

    #     samples = np.linspace(front_img, back_img, num=5)
    #     for z in samples:
    #         npz_array = np.append(npz_array, np.array([x, y, z, 0])[np.newaxis, :], axis=0)
    #         distances, indices = tree.query(np.array([x, y, z]), k=1)
    #         print(distances, indices)
    #         print(x, y, z)
    #         print(mesh_vertices[indices])

    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(npz_array[:, :3])
    # o3d.io.write_point_cloud("test.pcd", pcd)
    # pcd_load = o3d.io.read_point_cloud("test.pcd")
    # o3d.visualization.draw_geometries([pcd_load])