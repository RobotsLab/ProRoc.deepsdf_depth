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
        for fn in sorted(file_name):
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

    mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
    print(mesh_center,np.mean(mesh.vertices, axis=0))

    return mesh

def remove_duplicated_mesh(mesh):
    '''Remove meshes with the same vertices'''  # może się przydać normalizacja wektorów i ważenie względem dot product
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
                # max_dot_product = 0

                for v_point in voting_points[0:, :]:
                    vector = triangle_center - v_point

                    theta1 = np.dot(vector, normals[itr])
                    theta2 = np.dot(vector, normals[i]) 
                    # max_dot_product = max(max_dot_product, abs(theta1))

                    # zbieranie głosów za tym że normalna jest zgodna, tzn. skierowana na zewnątrz obiektu
                    if theta1 > 0:
                        votes += 1
                    
                    # print(v_point, vector, theta1, theta2)  # 1 = pokrywa się z wektorem, -1 = jest skierowany przeciwnie

                # print(max_dot_product, 'itr:', itr, 'i:', i, 'votes:', votes)
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
    
def u_distribution(normal_distribution, mean):
    '''Changes normal distribution to U-shaped distribution'''
    result = np.array(normal_distribution)
    result = np.where(result < mean, mean - result, result)
    result = np.where(result > mean, 3 * mean - result, result)
    # print(result)
    return result

def sampling(start, end, quantity, linspace=True):
    if linspace:
        samples = np.linspace(start, end, num=quantity)
    else:
        mu = end/2
        sigma = mu/3 # mean and standard deviation
        samples = np.random.normal(mu, sigma, quantity)
        samples = u_distribution(samples, mu)

        # visualisation
        # count, bins, ignored = plt.hist(samples, 20, density=True)
        # plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
        # plt.show()

    return samples

def ray_casting(name, textured_mesh, visualize=''):
    print("Processing: ", name)
    
    preprocessed_mesh = mesh_preprocessing(textured_mesh)
    duplicated_triangle_idx = remove_duplicated_mesh(preprocessed_mesh)
    preprocessed_mesh.remove_triangles_by_index(duplicated_triangle_idx)
    
    # np.set_printoptions(threshold=sys.maxsize)

    mesh = o3d.t.geometry.TriangleMesh.from_legacy(preprocessed_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

# transformation from depth image to point cloud
# pinhole camera model
# what parameters i need
# parameters of azure kinect
# transformation point cloud -> depth ; depth -> point cloud
# scale x, y, z to <0; 1>
# google typical camera parameters azure kinect

    # parameters from .xml
    # extrinsic_matrix = o3d.core.Tensor([[-0.809735, 0.364392, -0.459944, 0.666247],
    #                                    [0.585098, 0.560975, -0.585635, 0.344917],
    #                                    [0.0446166, -0.743321, -0.667446, 0.487054],
    #                                    [0, 0, 0, 1]])

    # camera parameters
    Fx_depth = 924.348
    Fy_depth = 924.921
    Cx_depth = 646.676
    Cy_depth = 344.145
    width=1280
    height=720

    npz_array = np.zeros((1,4))
    surfaces = {}
    first_img = True
    angle = np.deg2rad(-180)

    intrinsic_matrix = o3d.core.Tensor([[Fx_depth, 0, Cx_depth],
                                       [0, Fy_depth, Cy_depth],
                                       [0, 0, 1]])    
    extrinsic_matrix = o3d.core.Tensor([[1, 0, 0, 0],
                                       [0, np.cos(angle), np.sin(angle), 0],
                                       [0, np.sin(angle), np.cos(angle), 5],
                                       [0, 0, 0, 1]])
    
    while True:
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix,
            extrinsic_matrix,
            width,
            height
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

        # ploting depth image
        if first_img and visualize.lower() == 'depth':
            plt.imshow(img, cmap='gray')
            plt.title(name)
            first_img = False
            plt.show()

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

                samples = sampling(0, distance_diff[x][y], 4, linspace=True)

                if distance_diff[x][y] == 0:
                    continue

                z = img[x][y] - samples
                sdf_front = recent_image[x][y] - z
                sdf_back = z - img[x][y]
                sdf = np.maximum(sdf_front, sdf_back)
                point_data = np.column_stack((np.full(sdf.shape, x),   
                                        np.full(sdf.shape, y),
                                        np.full(sdf.shape, z),
                                        sdf))
                npz_array = np.concatenate((npz_array, point_data), axis=0)            

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
    pcd = []
    for npz in npz_array:
        z = npz[2]
        x = (Cx_depth - npz[0]) * z / Fx_depth
        y = (Cy_depth - npz[1]) * z / Fy_depth
        pcd.append([x, y, z])

    pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)  # set pcd_np as the point cloud points

    # Visualize:
    if visualize.lower() == 'cloud':
        o3d.visualization.draw_geometries([pcd_o3d])
    destination_filename = f"/home/piotr/Desktop/ProRoc/DeepSDF/ycb1/depth_to_pcd/{name.split('/')[-1]}.pcd"
    o3d.io.write_point_cloud(destination_filename, pcd_o3d)
    print(f"SAVED POINT CLOUD: {destination_filename}")

    # saved_pcd = o3d.io.read_point_cloud(destination_filename)
    # o3d.visualization.draw_geometries([saved_pcd])

    # exit(7)


if __name__ == '__main__':
    json_path = 'examples/splits/YCB_depth_image_train.json'
    data = load_data(json_path)
    preprocessed_files = os.listdir("/home/piotr/Desktop/ProRoc/DeepSDF/ycb1/depth_to_pcd/")
    for name, mesh in data.items():
        if 'bottle' in name and '_x0' in name: #  ('mug' in name and not '_' in name) or ('bottle' in name and '_x0' in name):
            source_filename = name.split('/')[-1] + '.pcd'
            if not source_filename in preprocessed_files:
                ray_casting(name, mesh, 'cloud')
            else:
                print(f"{source_filename} already exists.")
    
    




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


                #     if distance_diff[x][y] == 0:
                #     continue
                # # for sample in samples:
                # z = img[x][y] - samples
                # sdf_front = recent_image[x][y] - z
                # sdf_back = z - img[x][y]
                # sdfs = np.concatenate([sdf_back[:, np.newaxis], sdf_front[:, np.newaxis]], axis=1)
                # print(sdfs.shape)
                # sdf = np.max(sdfs, axis=1)
                # print(sdf.shape)
                # result = np.concatenate([np.full((len(samples), 2), [x, y]), samples[:, np.newaxis], sdf[:, np.newaxis]], axis=1)
                # npz_array = np.concatenate([npz_array, result], axis=0)
                # print(result.shape)

                