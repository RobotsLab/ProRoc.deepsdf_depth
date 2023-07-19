import open3d as o3d
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import sys


def load_data(json_path):
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

def mesh_preprocessing(mesh):
    mesh_vertices = mesh.vertices
    mesh_vertices = np.asarray(mesh_vertices)
    mesh_center = np.mean(mesh_vertices, axis=0)
    mesh_vertices -= mesh_center
    mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
    # print(mesh_center,np.mean(mesh.vertices, axis=0))

    return mesh

def ray_casting(name, textured_mesh):
    print(name)
    
    preprocessed_mesh = mesh_preprocessing(textured_mesh)
    mesh_vertices = np.asarray(preprocessed_mesh.vertices)
    
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

    while True:
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            fov_deg=45,
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


        ROI = img[img != 0]
        ROI_x = np.where(img != 0)[0]
        ROI_y = np.where(img != 0)[1]

        # ploting depth image
        # plt.imshow(img, cmap='gray')
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

        normals = ans["primitive_normals"].numpy()
        normals_mask = (normals[:, :, 1] < 0) & (normals[:, :, 2] < 0)
        direction = np.where(normals_mask)
        # print(normals_mask, direction)

        # exit(7)
        # sampling
        # mam x, y

        distance = np.zeros((width, height))
        if surfaces:
            recent_image = surfaces[list(surfaces.keys())[-1]]
            distance_diff = img - recent_image

            for point in zip(direction[0], direction[1]):
                x, y = point[0], point[1]
                samples = np.linspace(0, distance_diff[x][y], num=25)
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
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh)
        
        # o3d.visualization.draw_geometries([preprocessed_mesh])

    npz_array = np.delete(npz_array, 0, axis=0)
    print(npz_array)

    # print(npz_array[:, :3])
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(npz_array[:, 0], npz_array[:, 1], npz_array[:, 2], c=npz_array[:, 3], cmap='viridis', marker='x')
    # # ax.scatter(mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_vertices[:, 2], c='r', marker='x')
    # plt.show()

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(npz_array[:, :3])
    # o3d.visualization.draw_geometries([pcd])

    # exit(7)


if __name__ == '__main__':
    json_path = 'examples/splits/YCB_bottle_mug_train.json'
    data = load_data(json_path)
    for name, mesh in data.items():
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