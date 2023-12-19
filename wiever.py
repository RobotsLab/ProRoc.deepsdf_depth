import open3d as o3d
import numpy as np

def array_to_pcd(array):
    pcd = o3d.geometry.PointCloud()  # create point cloud object
    pcd.points = o3d.utility.Vector3dVector(array[:, :3])
    return pcd

def load(path):
    if path.endswith(".npz"):
        dict_data = np.load(path)
        data = dict_data[dict_data.files[1]]
        new_data = data[data[:, 3] <= 0.1]
        print(new_data)
        return array_to_pcd(new_data)
    elif path.endswith(".pcd"):
        pcd = o3d.io.read_point_cloud(path)
        return pcd
    elif path.endswith(".obj") or path.endswith(".ply"):
        mesh = o3d.io.read_triangle_mesh(path)
        points = np.asarray(mesh.vertices)
        return array_to_pcd(points)
    
    return None

def visualize(pcd):
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    o3d.visualization.draw_geometries([origin, pcd])

if __name__ == '__main__':
    path = "examples/depth/Reconstructions/1500/Meshes/dataset_YCB_test/magisterka_depth/29b6f9c7ae76847e763c517ce709a8cc.npz"
    point_cloud = load(path)
    if point_cloud.points != 0:
        visualize(point_cloud)
    else:
        print("File is empty")