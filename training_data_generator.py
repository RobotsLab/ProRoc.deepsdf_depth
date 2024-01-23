import argparse
import open3d as o3d
import matplotlib.pyplot as plt
import copy

from depth_utils import *
from depth_camera import Camera
from depth_image_generator import File as DepthFile

class File():
    def __init__(self, source_path, destination_dir):
        self.source_path = source_path
        self.destination_dir = destination_dir 
        self.name = self.get_name_()
        self.version = self.get_version_()
        self.o_c_transformation = np.zeros(6)
        self.pixels = []
        self.ds = 0

    def get_name_(self):
        tail = os.path.split(self.source_path)[1]
        return tail.split('.')[0]
    
    def get_version_(self):
        dir_files = os.listdir(self.destination_dir)
        file_number = len([x for x in dir_files if x.startswith(self.name) and x.endswith('.txt')]) + 1
        return file_number
    
    def get_camera_parameters(self, f, cx, cy):
        self.f = f
        self.cx = cx
        self.cy = cy

    def get_image_resolution(self, Ndx, Ndy):
        self.Ndx = Ndx
        self.Ndy = Ndy

    def get_outliar_factor(self, ds):
        self.ds = ds

    def get_bounding_box_coords(self, nx, ny, z):
        self.nx = nx
        self.ny = ny
        self.z = z

    def get_bounding_box_size(self, ndx, ndy, dz, dz2):
        self.ndx = ndx
        self.ndy = ndy
        self.dz = dz
        self.dz2 = dz2

    def save(self):
        with open(os.path.join(self.destination_dir, self.name + '_inp' +'.txt'), 'w') as f:
            for pixel in self.pixels:
                f.write(f"{' '.join(map(str, pixel))}\n")
                print(pixel)
                # print(pixel)
                # for p in pixel:
                    # print(p)


def load_depth_file(input_file):
    with open(input_file.source_path, "r") as file:
        input_file.o_c_transformation = np.array(file.readline().split(), dtype=np.float32)
        f, cx, cy = file.readline().split()
        input_file.get_camera_parameters(float(f), float(cx), float(cy))
        Ndx, Ndy = file.readline().split()
        input_file.get_image_resolution(int(Ndx), int(Ndy))
        input_file.ds = float(file.readline())
        nx, ny, z = file.readline().split()
        input_file.get_bounding_box_coords(int(nx), int(ny), float(z))
        ndx, ndy, dz, dz2 = file.readline().split()
        input_file.get_bounding_box_size(int(ndx), int(ndy), float(dz), float(dz2))
        pixels = file.readlines()
        input_file.pixels = [np.array(pixel.split(), dtype=np.float32) for pixel in pixels]

def generate_pcd(input_file):
    pixels = np.asarray(input_file.pixels)
    pixels = np.reshape(pixels, (input_file.ndy, input_file.ndx, -1))

    points = np.zeros((1,3))

    for image in range(pixels.shape[2]):
        img = np.zeros((input_file.Ndy, input_file.Ndx))
        img[input_file.ny:input_file.ny+input_file.ndy,input_file.nx:input_file.nx+input_file.ndx] = pixels[:, :, image]
        roi_y, roi_x = np.where(img!=0)

        # plt.imshow(img, cmap='gray')
        # plt.show()

        z = np.array(img[img!=0])
        x = (input_file.cx - roi_x) * z / input_file.f  # y on image is x in real world
        y = (input_file.cy - roi_y) * z / input_file.f  # x on image is y in real world

        points_data = np.column_stack([x, y, z])
        points = np.concatenate((points, points_data), axis=0) 

    points = np.delete(points, 0, axis=0)
    pcd = o3d.geometry.PointCloud()  # create point cloud object
    pcd.points = o3d.utility.Vector3dVector(points)  # set pcd_np as the point cloud points

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # o3d.visualization.draw_geometries([pcd, origin])
    
    return pcd

def find_sdf(input_file, pcd, point, z, index):
    point += z
    height, width = input_file.ndy, input_file.ndx

    v = (index // width) + input_file.ny
    u = (index % width) + input_file.nx

    z = point
    x = (input_file.cx - u) * z / input_file.f  # y on image is x in real world
    y = (input_file.cy - v) * z / input_file.f  # x on image is y in real world

    sampled_point = np.column_stack([x, y, z])

    object_points = np.asarray(pcd.points)
    
    from scipy.spatial import KDTree

    # find 10 nearest points
    tree = KDTree(object_points, leafsize=object_points.shape[0]+1)
    distances, ndx = tree.query([sampled_point], k=1)

    sdf = np.linalg.norm(object_points[ndx] - sampled_point)

    return sdf

def sample_points(unique_surf_distances, num_samples):
    if len(unique_surf_distances) % 2 == 0:
        samples = []
        for i, surface in enumerate(unique_surf_distances):
            distance = (surface - unique_surf_distances[0])/2
        return 1
    else:
        return None

if __name__ == '__main__':
    SOURCE_PATH = 'dataset_YCB_train/DepthDeepSDF/files/untitled_6_0.txt'
    DESTINATION_PATH = 'dataset_YCB_train/DepthDeepSDF/files'

    input_file = DepthFile(SOURCE_PATH)
    load_depth_file(input_file)

    output_file = File(SOURCE_PATH, DESTINATION_PATH)
    
    pcd = generate_pcd(input_file)
    odds = 0
    output_file.pixels = []
    for i, pixel in enumerate(input_file.pixels):
        unique = np.unique(pixel[pixel!=0])
        if unique.any() and len(unique)%2 == 0:
            rd = unique[0] - input_file.dz
            dd = [rd]
            for j, number in enumerate(unique):
                if j == 0:
                    continue
                if j%2 == 1:
                    sampled_distance = (number - unique[0])/2
                    sdf = 0
                else:
                    sampled_distance = (number - unique[0])/2
                    sdf = find_sdf(input_file, pcd, sampled_distance, unique[0], i)
                dd.append(sampled_distance)
                dd.append(sdf)
            dd = np.array(dd)
            output_file.pixels.append(dd)

        else:
            output_file.pixels.append(np.array([0.]))
        if len(unique)%2 != 0:
            odds+=1
    output_file.save()
    print(odds)