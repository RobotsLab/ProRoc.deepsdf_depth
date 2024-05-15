import argparse
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import random
import cv2

from depth.utils import *
from depth.camera import Camera
from depth_file_generator import File as ViewsFile

POWER_FACTOR = 20
GT = True

class File():
    def __init__(self, source_path, destination_dir=''):
        self.source_path = source_path
        self.destination_dir = destination_dir 
        self.name = self.get_name_()
        if destination_dir:
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
        self.nx = int(nx)
        self.ny = int(ny)
        self.z = z

    def get_bounding_box_size(self, ndx, ndy, dz, dz2):
        self.ndx = ndx
        self.ndy = ndy
        self.dz = dz
        self.dz2 = dz2

    def save(self, view):
        if GT:
            with open(os.path.join(self.destination_dir, self.name + '_' + str(view) + f'_gt.txt'), 'w') as f:
                f.write(f"{' '.join(map(str, self.o_c_transformation))}\n")
                f.write(f'{self.f} {self.cx} {self.cy}\n')
                f.write(f'{self.Ndx} {self.Ndy}\n')
                f.write(f'{self.ds}\n')
                f.write(f'{self.nx} {self.ny} {self.z}\n')
                f.write(f'{self.ndx} {self.ndy} {self.dz} {self.dz2}\n')
                for image in self.pixels:
                    for row in image:
                        for pixel in row:
                            f.write(f"{' '.join(map(str, pixel))}\n")
            print("FILE SAVED:", os.path.join(self.destination_dir, self.name + '_' + str(view) + f'_gt.txt'))
        else:
            with open(os.path.join(self.destination_dir, self.name + '_' + str(view) + f'_a{POWER_FACTOR}.txt'), 'w') as f:
                f.write(f"{' '.join(map(str, self.o_c_transformation))}\n")
                f.write(f'{self.f} {self.cx} {self.cy}\n')
                f.write(f'{self.Ndx} {self.Ndy}\n')
                f.write(f'{self.ds}\n')
                f.write(f'{self.nx} {self.ny} {self.z}\n')
                f.write(f'{self.ndx} {self.ndy} {self.dz} {self.dz2}\n')
                for image in self.pixels:
                    for row in image:
                        for pixel in row:
                            f.write(f"{' '.join(map(str, pixel))}\n")
            print("FILE SAVED:", os.path.join(self.destination_dir, self.name + '_' + str(view) + f'_a{POWER_FACTOR}.txt'))


def set_camera(input_file, output_file):
    '''This function is used to set camera position'''
    # camera parameters
    Fx_depth = 924
    Fy_depth = 924
    Cx_depth = 640
    Cy_depth = 360

    width=1280
    height=720

    intrinsic_matrix = o3d.core.Tensor([[Fx_depth, 0, Cx_depth],
                                       [0, Fy_depth, Cy_depth],
                                       [0, 0, 1]])    

    roll = np.deg2rad(input_file.o_c_transformation[3])  # Rotation around X-axis
    pitch = np.deg2rad(input_file.o_c_transformation[4])  # Rotation around Y-axis
    yaw = np.deg2rad(input_file.o_c_transformation[5])  # Rotation around Z-axis

    # Translation values
    tx = input_file.o_c_transformation[0]
    ty = input_file.o_c_transformation[1]
    tz = input_file.o_c_transformation[2]

    output_file.get_image_resolution(width, height)
    output_file.get_camera_parameters(Fx_depth, Cx_depth, Cy_depth)

    camera = Camera(Fx=Fx_depth, Fy=Fy_depth, Cx=Cx_depth, Cy=Cy_depth, width=width, height=height, intrinsic_matrix=intrinsic_matrix)
    camera.rotate(roll=roll, pitch=pitch, yaw=yaw)
    camera.translate(tx=tx, ty=ty, tz=tz)
    
    return camera

def s_o_translation(mesh):
    mesh_vertices = np.copy(np.asarray(mesh.vertices))
    mesh_translation_vector = object_translation(mesh_vertices, True)
    return mesh_translation_vector

def s_o_rotation(mesh):
    return np.array([0.,0.,0.])

def object_translation(vertices, pos_z=False):
    vertices_mean = np.mean(vertices, axis=0)
    vertices -= vertices_mean

    if pos_z:
        min_z = np.min(vertices[:, 2])
    else:
        min_z = 0

    translation_vec = vertices_mean + np.array([0., 0., min_z])
    return translation_vec

def translate(mesh, translation):
    mesh_vertices = np.asarray(mesh.vertices)
    mesh_vertices -= translation
    mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
    return mesh

def rotate(mesh, rotation):
    R = mesh.get_rotation_matrix_from_xyz(np.radians(rotation))
    mesh.rotate(R, center=(0, 0, 0))
    return mesh

def scale(mesh, scale_factor, scale_to_unit=False):
    mesh_vertices = np.copy(np.asarray(mesh.vertices))
    input_z_dist = np.max(mesh_vertices[:, 2]) - np.min(mesh_vertices[:, 2])

    if scale_to_unit:
        max_z_dist = np.max(mesh_vertices[:, 2]) - np.min(mesh_vertices[:, 2])
        mesh_vertices /= max_z_dist
    
    mesh_vertices *= scale_factor
    output_z_dist = np.max(mesh_vertices[:, 2]) - np.min(mesh_vertices[:, 2])
    mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
    scaled_mesh = translate(mesh, object_translation(mesh_vertices, True))

    real_scale_factor = output_z_dist / input_z_dist

    return scaled_mesh, real_scale_factor

def load_generator_file(input_file):
    with open(input_file.source_path, "r") as file:
        input_file.scale = float(file.readline())
        input_file.s_o_transformation = np.array(file.readline().split(), dtype=np.float32)
        input_file.o_c_transformation = np.array(file.readline().split(), dtype=np.float32)
        frames = file.readlines()
        input_file.frames = [np.array(frame.split(), dtype=np.float32) for frame in frames]

def find_angle(v1, v2):
    c = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    angle = np.rad2deg(np.arccos(np.clip(c, -1, 1)))
    return angle

def triangle_area_3d(vertices):
    """
    Compute the area of a triangle in 3D space given its vertices.
    
    Args:
    - vertices (np.ndarray): Array of shape (3, 3) containing the vertices of the triangle.
    
    Returns:
    - float: Area of the triangle.
    """
    # Get vectors representing two sides of the triangle
    AB = vertices[1] - vertices[0]
    AC = vertices[2] - vertices[0]
    
    # Compute the cross product
    cross_product = np.cross(AB, AC)
    
    # Compute the magnitude of the cross product
    area = 0.5 * np.linalg.norm(cross_product)
    
    return area

def visualize_odds(origin_ray_vector, ray_vector, triangles_id, mesh, hits):
    # create pcd and add all points along the ray to pcd - lol we have origin point and vector
    # get vertices for certain vertices
    points_along_ray = np.array([origin_ray_vector + t * ray_vector for t in np.linspace(1, 2, 300)])
    pcd = o3d.geometry.PointCloud()  # create point cloud object
    pcd.points = o3d.utility.Vector3dVector(points_along_ray)  # set pcd_np as the point cloud points
    # Create a subset mesh containing only the triangles with IDs in `triangle_id`
    subset_mesh = o3d.geometry.TriangleMesh()
    subset_mesh.vertices = mesh.vertices
    max_area = 0
    triangles_verts = {}

    for i, triangle_id in enumerate(triangles_id):
        verts_id = mesh.triangles[triangle_id]
        mesh_vertices = np.asarray(subset_mesh.vertices)
        triangle_vertices = mesh_vertices[verts_id]
        # Convert vertices to a tuple so they can be used as dictionary keys
        verts_key = tuple(sorted(map(tuple, triangle_vertices)))
        # Append triangle ID to the list of triangles with the same vertices
        triangles_verts[verts_key] = [hits[i]]
        # max_area = max(triangle_area_3d(triangle_vertices), max_area)
    
    unique_hits = []
    for verts, hit in triangles_verts.items():
        # print("Vertices:", verts)
        # print("Triangle IDs:", hit)
        # print()
        unique_hits.append(hit)

    # if len(unique_hits) % 2 != 0:
    #     visualize_verts = []
    #     for i in range(len(unique_hits)):
    #         print("TRIANGLE:", triangles_id[i])
    #         print("VERTICES:", list(triangles_verts.keys())[i])
    #         print("HITS:", unique_hits[i])
    #         verts_id = mesh.triangles[triangles_id[i]]
    #         visualize_verts.append(verts_id)
    #     triangles_id = np.row_stack(visualize_verts)
    #     subset_mesh.triangles = o3d.utility.Vector3iVector(triangles_id)
    #     # Visualize the subset mesh
    #     o3d.visualization.draw_geometries([mesh, pcd])

    return unique_hits
    
def halo(img):
    img[img == np.inf] = 0
    img = img.astype(np.float32)
    
    # Define the kernel for dilation
    kernel = np.ones((3,3),np.uint8)  # Adjust the kernel size as needed

    # Perform dilation on the image
    img_dilated = cv2.dilate(img, kernel)

    # Calculate the difference between the dilated image and the original image
    img_diff = cv2.absdiff(img_dilated, img)
    # threshold_diff = img_diff > 1.0
    # img_diff = threshold_diff.astype(np.uint8) * 255


    # Calculate the number of new pixels that occurred after dilation
    # num_new_pixels = np.count_nonzero(img_diff)

    # # Display the dilated image and the difference image
    # cv2.imshow('Dilated Image', img_dilated)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow('Difference Image', img_diff)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # plt.imshow(img_dilated, cmap='gray')
    # plt.title('img_dilated')
    # plt.show()
    # plt.imshow(img_diff, cmap='gray')
    # plt.title('img_diff')
    # plt.show()
    # plt.imshow(img, cmap='gray')
    # plt.title('img')
    # plt.show()
    return img_diff

def stack_images(file, input_mesh, camera, view=0):
    '''SOLVED
    Gdy któryś z promieni pierwszy dotknie następnego trójkąta, ten trójkąt nie zostanie zarejestrowany przez sąsiedni promień.
    Sprawdzić ile promieni przecina trójkąt do usunięcia, jeżeli po usunięciu liczba intersekcji spadnie dla 
    większej ilości promieni niż przewidziano to nie można usunąć tego mesha.
    '''
    mesh = copy.deepcopy(input_mesh)
    mesh.compute_triangle_normals()
    triangle_normals = np.asarray(mesh.triangle_normals)
    

    scene_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(scene_mesh)
    sub_size = 175

    rays = camera.raycasting()
    np_rays = rays.numpy()

    # Depth image.
    lx = scene.list_intersections(rays)
    ans = scene.cast_rays(rays)

    t_hit = lx['t_hit'].numpy()
    ray_ids = lx['ray_ids'].numpy()
    primitive_ids = lx['primitive_ids'].numpy()
    img = ans['t_hit'].numpy()
    unique_ray_ids, counts = np.unique(ray_ids, return_counts=True)
    diff_img = halo(img)
    # Create a dictionary to store distances for each ray
    distances_dict = {ray_id: ([], [], []) for ray_id in unique_ray_ids}

    # Iterate through the data and store distances for each ray
    for i, ray_id in enumerate(ray_ids):
        distances_dict[ray_ids[i]][0].append(t_hit[i])
        distances_dict[ray_ids[i]][1].append(ray_id)
        distances_dict[ray_ids[i]][2].append(primitive_ids[i])
    
    depth_image = np.zeros((file.Ndy, file.Ndx, np.max(counts)))

    # mask = diff_img != 0
    # depth_image[mask, 0] = 255
    depth_image[:, :, 0] = diff_img[:,:]
    num_new_pixels = np.count_nonzero(depth_image[:, :, 0])
    print("New points from halo:", num_new_pixels)
    # plt.imshow(depth_image[:, :, 0], cmap='gray')
    # plt.title('depth_image with only diff image')
    # plt.show()
    min_y = file.Ndy
    min_x = file.Ndx
    max_y = 0
    max_x = 0
    min_intersection_angle = 100
    max_intersection_angle = 0
    rejected_angles = 0
    odd = 0
    sin_angles = []
    angles_plot = []
    num_hits = {}

    for key in distances_dict.keys():
        hits, ray_id, triangles_id = distances_dict[key]
        y = key // file.Ndx
        x = key % file.Ndx
        ray_vector = np_rays[y, x, -3:]  
        origin_ray_vector = np_rays[y, x, :3]  # array with ray vector starting point

        # lists for visualization
        correct_hits = []
        correct_angles_and_sin = []

        # rejection sampling for rays
        for i in range(len(hits)):
            surface_normal_vector = triangle_normals[triangles_id[i]]
            intersection_angle = find_angle(surface_normal_vector, ray_vector)  # 90 means that surface is parallel to ray
            min_intersection_angle = min(intersection_angle, min_intersection_angle)
            max_intersection_angle = max(intersection_angle, max_intersection_angle)

            # absolute value of sinus from intersection angle (ray-surface)
            sin_angle = abs(np.sin(np.deg2rad(intersection_angle)))

            # if grand truth needed
            if GT:
                sin_angle = -1

            # power_factor the higher the less aggressive rejection
            probability = np.power(random.random(), 1 / POWER_FACTOR)
            if probability >= sin_angle:
                correct_hits.append(hits[i])
                correct_angles_and_sin.append((intersection_angle, sin_angle))
            else:
                rejected_angles += 1

        unique_hits, indices_hits = np.unique(hits, axis=0, return_index=True)

        # number or intersections and quality check
        if len(hits) == len(correct_hits) and len(hits) % 2 == 0:
            for i in range(len(correct_hits)):
                depth_image[y, x, i] = hits[i]
                angles_plot.append(correct_angles_and_sin[i][0])
                sin_angles.append(correct_angles_and_sin[i][1])
        elif len(hits) == len(correct_hits) and len(unique_hits) % 2 == 0:
            for i, idx in enumerate(sorted(indices_hits)):
                depth_image[y, x, i] = hits[idx]
                # angles_plot.append(correct_angles_and_sin[i][0])
                # sin_angles.append(correct_angles_and_sin[i][1])
        elif len(hits) == len(correct_hits) and len(hits) % 2 == 1:
            # return hits if len(hits) jest parzysta to spoko, else odds += 1
            unique_hits_new = visualize_odds(origin_ray_vector, ray_vector, triangles_id, mesh, hits)
            if len(unique_hits_new) % 2 == 0:
                for i, hit in enumerate(unique_hits_new):
                    depth_image[y, x, i] = hit[0]
            else:
                odd += 1

        non_zero_hits = np.argwhere(depth_image[y, x, :])
        if len(non_zero_hits) in num_hits.keys():
            num_hits[len(non_zero_hits)] += 1
        else:
            num_hits[len(non_zero_hits)] = 0

        min_y = min(min_y, y)
        min_x = min(min_x, x)
        max_y = max(max_y, y)
        max_x = max(max_x, x)
        # print(depth_image.shape)[ 1.0606601 -0.         1.0606601]
    # plt.scatter(x=angles_plot, y=sin_angles)
    # plt.show()
    # num_new_pixels = np.count_nonzero(depth_image[:, :, 0])
    # print(num_new_pixels)
    # plt.imshow(depth_image[:, :, 0], cmap='gray')
    # plt.title('depth_image first layer')
    # plt.show()
    print(f"Angle range from {90-view} to {90+view}\nSamples removed due to angle: {rejected_angles} Minimum angle: {min_intersection_angle} Maximum angle: {max_intersection_angle}")
    print("Odds:", odd)
    print("Length dict: ", len(distances_dict))
    odds_ratio = 100 * odd / (odd + len(distances_dict))
    print("Odds ratio:", round(odds_ratio, 2), "%")
    for key, value in num_hits.items():
        print(f"Intersected triangles: {key}, Intersections: {value}")
    if odds_ratio >= 0.5:
        print("DISMISSED!\n")
        return None

    min_z = np.min(depth_image[depth_image!=0])
    max_z = np.max(depth_image[depth_image!=0])
    x_center = (min_x+max_x)//2
    y_center = (min_y+max_y)//2
    file.get_bounding_box_coords(x_center-(sub_size/2), y_center-(sub_size/2), min_z)
    # file.get_bounding_box_size(x_max-x_min, y_max-y_min, min_z-0.1, min_z+0.4)
    file.get_bounding_box_size(sub_size, sub_size, min_z-0.1, min_z+0.4)

    return depth_image[file.ny:file.ny+file.ndy, file.nx:file.nx+file.ndx, :]

if __name__ == '__main__':
    categories = ['mug', 'laptop']
    for category in categories:
        names_txt = [name for name in os.listdir(f'dataset_YCB_train/DepthDeepSDF/files/{category}') if not '_' in name]
        saved_files = 0
        for current_iteration, name_txt in enumerate(names_txt):
            SOURCE_PATH = os.path.join(f'dataset_YCB_train/DepthDeepSDF/files/{category}', name_txt)
            MESH_PATH =  os.path.join(f'ShapeNetCore/{category}', name_txt.split('.')[0], 'models/model_normalized.obj')
            DESTINATION_PATH = f'dataset_YCB_train/DepthDeepSDF/files/{category}'

            input_file = ViewsFile(SOURCE_PATH)
            load_generator_file(input_file)

            output_file = File(SOURCE_PATH, DESTINATION_PATH)
            output_file.o_c_transformation = input_file.o_c_transformation

            input_mesh = load_file(MESH_PATH)
            input_mesh = rotate(input_mesh, np.array([90, 0, 0]))
            centered_mesh = translate(input_mesh, input_file.s_o_transformation[:3])
            scaled_mesh, _ = scale(centered_mesh, input_file.scale)
            print()
            print("===================================================")
            print("SOURCE PATH", SOURCE_PATH)
            for view, frame in enumerate(input_file.frames):
                if name_txt.split('.')[0] + f'_{view}_a{POWER_FACTOR}.txt' in os.listdir(f'dataset_YCB_train/DepthDeepSDF/files/{category}'):
                    continue
                # if not name_txt.split('.')[0] + f'_{view}_gt.txt' in os.listdir(f'dataset_YCB_train/DepthDeepSDF/files/{category}'):
                #     continue
                print("VIEW:", view)
                scaled_mesh = translate(scaled_mesh, frame[:3])
                scaled_mesh = rotate(scaled_mesh, frame[3:])

                mesh = o3d.t.geometry.TriangleMesh.from_legacy(scaled_mesh)
                scene = o3d.t.geometry.RaycastingScene()
                scene.add_triangles(mesh)

                camera = set_camera(input_file, output_file)
                rays = camera.raycasting()
                
                # Depth image.
                ans = scene.cast_rays(rays)
                img = ans['t_hit'].numpy()
                ids = ans["primitive_ids"].numpy()

                img[img == np.inf] = 0
                img = img.astype(np.float32)

                # plt.imshow(img, cmap='gray')
                # plt.title('Input camera image')
                # plt.show()

                depth_image = stack_images(output_file, scaled_mesh, camera)
                print(f"CURRENT ITERATION: {current_iteration} OUT OF {len(names_txt)}")
                if depth_image is None:
                    print("DEPTH IMAGE IS NONE", name_txt)
                    break
                elif view == 9:
                    saved_files += 1
                print(f"SAVED FILES: {saved_files}\n")
                
                # plt.imshow(depth_image[:,:,0], cmap='gray')
                # plt.title('Output camera image')
                # plt.show()
                plt.imshow(img, cmap='gray')
                plt.title('Input camera image')
                # plt.show()
                if GT:
                    plt.savefig(os.path.join(output_file.destination_dir, output_file.name + '_' + str(view) + f'_gt.png'))
                else:
                    plt.savefig(os.path.join(output_file.destination_dir, output_file.name + '_' + str(view) + f'_a{POWER_FACTOR}.png'))
                scaled_mesh = translate(scaled_mesh, -frame[:3])
                scaled_mesh = rotate(scaled_mesh, -frame[3:])

                output_file.pixels.append(depth_image)
                output_file.save(view)
                output_file.pixels.clear()
            if saved_files == 10:
                break
            # w tym programie musi być liczenie ścianek na promieniu, jeżeli:
            # unique(intersection) % 2 = 0 promień zostaje
            # unique(intersection) % 2 = 1 promień jest odrzucany
            # 100 * odrzucone promienie / wszystkie promienie > 5 widok jest odrzucony
            
            # na majówke
            
            # 1. zrobić visual debigung
            # 1.1 pokazać konkretny promień z trójkątami które są przecinane
            # 1.2 pokazać trójkąty na promieniu - umieścić na promieniu gęstą ilość punktów żeby zwizualizować promień
            # 2. wygenerować bardzo gęstą chmurę punktów z normalnymi

            # aureola
            # telefon, aprat, miska, kubek, butelka, laptop - przedmioty biurkowe
            # zmierzyć metryki

            # bowls
            # 1b4d7803a3298f8477bdcb8816a3fac9_0_gt
            # 12ddb18397a816c8948bef6886fb4ac_0_gt
            # 292d2dda9923752f3e275dc4ab785b9f_0_gt
            # 11547e8d8f143557525b133235812833_0_gt
            # 260545503087dc5186810055962d0a91_0_gt