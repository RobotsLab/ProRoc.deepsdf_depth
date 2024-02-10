import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from depth.camera import Camera

def loadSDFpoints(source_path):
    dict_data = np.load(source_path)
    pos_data = dict_data[dict_data.files[0]]
    neg_data = dict_data[dict_data.files[1]]
    result = np.concatenate((pos_data, neg_data), axis=0)
    return result

def construct_image(points, name):

    # camera parameters
    Fx_depth = 924
    Fy_depth = 924
    Cx_depth = 640
    Cy_depth = 360

    width=1280
    height=720

    # Rotation angles in radians
    roll = np.deg2rad(0)  # Rotation around X-axis
    pitch = np.deg2rad(0)  # Rotation around Y-axis
    yaw = np.deg2rad(0)  # Rotation around Z-axis

    # Translation values
    tx = -.03
    ty = -.03
    tz = 1.5

    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, 3] = [tx, ty, tz]

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

    extrinsic_matrix[:3, :3] = rotation_matrix

    xyz = points[:, :3]
    xyz = np.column_stack((xyz, np.full(xyz[:, 0].shape, 1)))

    xyz = extrinsic_matrix @ xyz.T
    xyz = xyz.T

    z = xyz[:, 2]
    x = Cx_depth - Fx_depth * xyz[:, 0] / z 
    y = Cy_depth - Fy_depth * xyz[:, 1] / z

    x = np.round(x, 0)
    y = np.round(y, 0)

    image = np.column_stack((x, y, z, points[:, 3]))

    max_sdf = 2*np.sqrt(2)/256
    image = image[(image[:, 3] < max_sdf) & (image[:, 3] > -0)]
    # print(f"max sdf: {max_sdf}, shape: {image.shape}")

    object_center = list(np.mean(image, axis=0))
    # print(f"object center: {object_center}")

    image = image[(image[:, 0] < width) & (image[:, 0] > 0) & (image[:, 1] < height) & (image[:, 1] > 0) & (image[:, 2] > 1)]

    image_back = image[image[:, 2] >= object_center[2]]
    image_front = image[image[:, 2] < object_center[2]]

    image_front[:, 2] += image_front[:, 3]
    image_back[:, 2] -= image_back[:, 3]

    surface_image = np.concatenate((image_front, image_back), axis=0)
    depth_image = np.zeros((height, width))
    for i in range(surface_image.shape[0]):
        x = int(surface_image[i, 0])
        y = int(surface_image[i, 1])
        z = surface_image[i, 2]
        depth_image[y][x] = z

    mesh, pcd = depth_to_pcd(surface_image, Cx_depth, Cy_depth, Fx_depth, Fy_depth)


    return mesh, pcd

def points_stats(points):
    # Count the number of points for each unique SDF value
    sdf_values = points[:, 2]

    unique_sdf_values, counts = np.unique(sdf_values, return_counts=True)
    print(unique_sdf_values, counts)
    plt.hist(sdf_values, bins=20, edgecolor='black', alpha=0.7)

    # Create the column chart
    # plt.bar(unique_sdf_values, counts, width=0.05, align='center')

    # Customize the plot
    plt.title('SDF Value Distribution')
    plt.xlabel('SDF Value')
    plt.ylabel('Number of Points')
    plt.show()

def depth_to_pcd(depth_image, Cx, Cy, Fx, Fy):
    z = depth_image[:, 2]
    x = (Cx - depth_image[:, 0]) * z / Fx  # y on image is x in real world
    y = (Cy - depth_image[:, 1]) * z / Fy  # x on image is y in real world
    npz_result = np.column_stack([x, y, z])
    # print("npz_rezuly", npz_result)

    pcd = np.column_stack((npz_result[:, 0], npz_result[:, 1], npz_result[:, 2]))
    pcd_mean = np.mean(pcd, axis=0)
    pcd -= pcd_mean
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

    mesh = pcd_to_obj(pcd=pcd_o3d)

    return mesh, pcd_o3d

def scale_and_center(points, scale_factor=False):
    points[:, 0] -= np.mean(points[:, 0])
    points[:, 1] -= np.mean(points[:, 1]) - abs(np.min(points[:, 1]))/2
    points[:, 2] += abs(np.min(points[:, 2]))

    max_distance = 0
    for point in points:
        max_distance = max(max_distance, np.linalg.norm(point))

    if scale_factor:
        scale_multiplier = max_distance * scale_factor
        points /= scale_multiplier

    return points

def pcd_to_obj(pcd):
    # pcd.estimate_normals()
    pcd = o3d.geometry.PointCloud(pcd)
    pcd.normals = o3d.utility.Vector3dVector(pcd.points)
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))
    
    return bpa_mesh

def comparison_prepare(filename):
    mesh = o3d.io.read_triangle_mesh(filename)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    vertices = np.asarray(mesh.vertices)
    new_vertices = scale_and_center(vertices, 1)
    mesh.vertices = o3d.utility.Vector3dVector(new_vertices)  # set pcd_np as the point cloud points

    o3d.visualization.draw_geometries([mesh, origin])

    return mesh

def main():
    source_path = "examples/depth/Reconstructions/1000/Meshes/dataset_YCB_test/mug_depth/"
    # source_path = "dataset_YCB_test/magisterka_sdf/"
    filenames = os.listdir(source_path)
    print(filenames)
    for filename in filenames:

        # To jest do przygotwania danych wyjściowych z DeepSDF fo porównania
        # if filename:
        #     prepared_mesh = comparison_prepare(os.path.join(source_path, filename, "models/model_normalized.obj"))
        #     destination_filename = filename.split(".")[0]
        #     print(destination_filename)
        #     o3d.io.write_triangle_mesh(os.path.join(source_path, destination_filename + "scaled_gt.ply"), prepared_mesh, write_vertex_normals=False, write_vertex_colors=False)
        # continue

        # if filename.endswith("_new"):
        #     prepared_mesh = comparison_prepare(os.path.join(source_path, filename, "models/model_normalized.obj"))
        #     destination_filename = filename.split(".")[0]
        #     print(destination_filename)
        #     o3d.io.write_triangle_mesh(os.path.join(source_path, destination_filename + "scaled_input.ply"), prepared_mesh, write_vertex_normals=False, write_vertex_colors=False)
        # else:
        #     continue

        source_file = os.path.join(source_path, filename)
        # if not filename.endswith(".npz"):
            # continue
        points = loadSDFpoints(source_file)
        points_stats(points)
        # exit(777)
        mesh, pcd = construct_image(points, source_file)

        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        o3d.visualization.draw_geometries([pcd, mesh, origin])

        destination_filename = filename.split(".")[0]
        print(destination_filename)
        o3d.io.write_point_cloud(os.path.join(source_path, destination_filename + "scaled.pcd"), pcd)
        o3d.io.write_triangle_mesh(os.path.join(source_path, destination_filename + "scaled.ply"), mesh, write_vertex_normals=False, write_vertex_colors=False)


if __name__ == "__main__":
    main()