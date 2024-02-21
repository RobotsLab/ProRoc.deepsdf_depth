import argparse
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import random
import json

from depth.utils import *
from depth.camera import Camera
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

    def save(self, dictionary):
        # with open(os.path.join(self.destination_dir, self.name + '_inp' +'.txt'), 'w') as f:
        #     for pixel in self.pixels:
        #         f.write(f"{' '.join(map(str, pixel))}\n")
                # print(pixel)
                # for p in pixel:
                    # print(p)
        with open(os.path.join(self.destination_dir, self.name + '_inp' +'.json'), "w") as outfile:
            json.dump(dictionary, outfile)
        print("Saved:", os.path.join(self.destination_dir, self.name + '_inp' +'.json'))


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

def find_sdf(input_file, pcd, points, first_surface, index):
    sdf_list = []
    for point in points:
        point += first_surface
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
        sdf_list.append(sdf)

    return np.array(sdf_list)

def sample_points(start, stop, num_samples):
    problems = 0
    mu = (stop + start) / 2
    sigma = (stop - start) / 6
    samples_gauss_dist = np.random.normal(mu, sigma, num_samples)
    samples_near_surf = u_distribution(samples_gauss_dist, mu, sigma)
    # samples_near_surf = samples_near_surf[(samples_near_surf >= start) & (samples_near_surf <= stop)]
    if np.all(samples_near_surf == samples_near_surf[0]):
        samples_near_surf = np.array([mu-2*sigma, mu-sigma, mu+sigma, mu+2*sigma])
        problems += 1
    return samples_near_surf, problems

#6512 6628 1_0
if __name__ == '__main__':
    for a in range(3):
        a += 1
        for b in range(5):
            SOURCE_PATH = f'dataset_YCB_train/DepthDeepSDF/files/untitled_{a}_{b}.txt'
            DESTINATION_PATH = 'dataset_YCB_train/DepthDeepSDF/input_training_data_u_dist'

            input_file = DepthFile(SOURCE_PATH)
            load_depth_file(input_file)

            output_file = File(SOURCE_PATH, DESTINATION_PATH)
            
            pcd = generate_pcd(input_file)
            odds = 0
            nans = 0
            problems = 0
            num_samples = 4
            samples = 0
            output_file.pixels = []
            visualize_dict = {}
            for i, pixel in enumerate(input_file.pixels):
                unique = np.unique(pixel[pixel!=0])
                x = (i % input_file.ndx) + input_file.nx
                y = (i // input_file.ndx) + input_file.ny
                key = f"{x}, {y}"
                visualize_dict[key] = []
                if unique.any() and len(unique)%2 == 0:
                    first_surface = unique[0]
                    fornt_bbox_z = input_file.dz
                    back_bbox_z = input_file.dz2
                    rd = first_surface - fornt_bbox_z
                    for j, point_z in enumerate(unique):
                        pixel = [rd]
                        if j%2 == 1:
                            sampled_points, problem = sample_points(unique[j-1], point_z, num_samples)
                            dd = sampled_points - rd - fornt_bbox_z
                            sdf = np.zeros(dd.shape)
                            for d, s in zip(dd, sdf):
                                output_file.pixels.append(np.array([rd, d, s]))
                                visualize_dict[key].append([rd, d, s])
                            problems += problem
                            samples += sampled_points.shape[0]
                        elif j > 0:
                            sampled_points, problem = sample_points(unique[j-1], point_z, num_samples)
                            dd = sampled_points - rd - fornt_bbox_z
                            sdf = find_sdf(input_file, pcd, dd, first_surface, i)
                            for d, s in zip(dd, sdf):
                                output_file.pixels.append(np.array([rd, d, s])) 
                                visualize_dict[key].append([rd, d, s])
                            problems += problem
                            samples += sampled_points.shape[0]

                        if point_z == unique[-1]:
                            pixel = [rd]
                            # sampled_point = random.uniform(point_z, back_bbox_z)
                            sampled_points, problem = sample_points(point_z, back_bbox_z, num_samples)
                            dd = sampled_points - rd - fornt_bbox_z
                            sdf = find_sdf(input_file, pcd, dd, first_surface, i)
                            for d, s in zip(dd, sdf):
                                output_file.pixels.append(np.array([rd, d, s]))
                                visualize_dict[key].append([rd, d, s])
                            problems += problem
                            samples += sampled_points.shape[0]

                elif unique.any() and len(unique)%2 == 1:
                    output_file.pixels.append(np.array([np.nan]))
                    visualize_dict[key].append([np.nan])
                    odds+=1
                else:
                    output_file.pixels.append(np.array([np.nan]))
                    visualize_dict[key].append([np.nan])
                    nans+=1
            output_file.save(visualize_dict)
            print("Odds:", odds)
            print("Total:", len(output_file.pixels))
            print("Ratio:", format(odds/len(output_file.pixels), ".00%"))

            print("\nNans:", nans)
            print("Total:", len(output_file.pixels))
            print("Ratio:", format(nans/len(output_file.pixels), ".00%"), "\n")
            print("PROBLEMS", problems)
            print("Samples", samples)
            print("--------------------------------------")
            depth_image = []
            for key, value in visualize_dict.items():
                pixel = []
                for row in value:
                    x = key[0]
                    y = key[1]
                    if np.any(np.isnan(row)):
                        # pixel.append(np.array([x, y, np.nan, np.nan]))
                        break
                    else:
                        z = row[0] + row[1] + input_file.dz
                        sdf = row[2]
                        depth_image.append(np.array([x, y, z, sdf]))

            image = np.vstack(depth_image)  # tu jest coś zjebane
            print(np.min(image, axis=0), np.max(image, axis=0), image.shape)
            
            z = np.array(image[:, 2])
            x = (input_file.cx - image[:, 0]) * z / input_file.f  # y on image is x in real world
            y = (input_file.cy - image[:, 1]) * z / input_file.f  # x on image is y in real world

            points = np.column_stack((x, y, z))

            pcd = o3d.geometry.PointCloud()  # create point cloud object
            pcd.points = o3d.utility.Vector3dVector(points)  # set pcd_np as the point cloud points
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            o3d.visualization.draw_geometries([pcd, origin])
            # o3d.io.write_point_cloud(f'dataset_YCB_train/DepthDeepSDF/input_training_data_u_dist/untitled_{a}_{b}.pcd', pcd)

            exit(777)