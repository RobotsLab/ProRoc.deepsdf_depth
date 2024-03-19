import open3d as o3d
import numpy as np

def array_to_pcd(array):
    pcd = o3d.geometry.PointCloud()  # create point cloud object
    pcd.points = o3d.utility.Vector3dVector(array[:, :3])
    return pcd

def load(path):
    if path.endswith(".npz"):
        dict_data = np.load(path)
        neg_data = dict_data[dict_data.files[1]]
        pos_data = dict_data[dict_data.files[0]]
        data = np.concatenate([neg_data, pos_data])
        print(data.shape)

        return array_to_pcd(data)
    elif path.endswith(".pcd"):
        pcd = o3d.io.read_point_cloud(path)
        return pcd
    elif path.endswith(".obj") or path.endswith(".ply"):
        mesh = o3d.io.read_triangle_mesh(path)
        points = np.asarray(mesh.vertices)
        return array_to_pcd(points)
    
    return None

def visualize(*pcd):
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    pcd = list(pcd)
    pcd.append(origin)
    o3d.visualization.draw_geometries(pcd)

if __name__ == '__main__':
    path = "dataset_YCB_train/DepthDeepSDF/files/untitled_1_0_inp.pcd"
    # path2 = "data_YCB/SdfSamples/dataset_YCB_train/sensors_depth/1ef68777bfdb7d6ba7a07ee616e34cd7.npz"
    point_cloud = load(path)
    # pcd2 = load(path2)
    if point_cloud.points != 0:
        visualize(point_cloud)
    else:
        print("File is empty")