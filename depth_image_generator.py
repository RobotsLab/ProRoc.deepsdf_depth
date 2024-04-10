import argparse
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import random

from depth.utils import *
from depth.camera import Camera
from depth_file_generator import File as ViewsFile

POWER_FACTOR = 10

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
    normals = ans['primitive_normals'].numpy()
    unique_ray_ids, counts = np.unique(ray_ids, return_counts=True)

    # Create a dictionary to store distances for each ray
    distances_dict = {ray_id: ([], [], []) for ray_id in unique_ray_ids}

    # Iterate through the data and store distances for each ray
    for i, ray_id in enumerate(ray_ids):
        distances_dict[ray_ids[i]][0].append(t_hit[i])
        distances_dict[ray_ids[i]][1].append(ray_id)
        distances_dict[ray_ids[i]][2].append(primitive_ids[i])
    
    depth_image = np.zeros((file.Ndy, file.Ndx, np.max(counts)))

    gt = True
    min_y = file.Ndy
    min_x = file.Ndx
    max_y = 0
    max_x = 0
    min_angle = 100
    max_angle = 0
    rejected_angles = 0
    odd = 0
    sin_angles = []
    angles_plot = []

    for key in distances_dict.keys():
        hits, ray_id, triangle_id = distances_dict[key]
        y = key // file.Ndx
        x = key % file.Ndx
        depth_values = np.asarray(hits)

        correct_hits = []
        correct_angles_and_sin = []
        for i in range(len(hits)):
            ray_vector = np_rays[y, x, -3:]
            normal_vector = triangle_normals[triangle_id[i]]
            angle = find_angle(normal_vector, ray_vector)
            min_angle = min(angle, min_angle)
            max_angle = max(angle, max_angle)

            # abs sinus z kąta angle
            sin_angle = abs(np.sin(np.deg2rad(angle)))

            # power_factor the higher the less aggressive rejection
            probability = np.power(random.random(), 1 / POWER_FACTOR)
            if gt:
                sin_angle = -1

            if probability >= sin_angle:
                correct_hits.append(hits[i])
                correct_angles_and_sin.append((angle, sin_angle))
            else:
                rejected_angles += 1

        if len(hits) == len(correct_hits) and len(hits) % 2 == 0:
            for i in range(len(correct_hits)):
                depth_image[y, x, i] = hits[i]
                angles_plot.append(correct_angles_and_sin[i][0])
                sin_angles.append(correct_angles_and_sin[i][1])
        elif len(hits) % 2 == 1:
            odd += 1


        min_y = min(min_y, y)
        min_x = min(min_x, x)
        max_y = max(max_y, y)
        max_x = max(max_x, x)
        # print(depth_image.shape)
    # plt.scatter(x=angles_plot, y=sin_angles)
    # plt.show()

    print(f"Angle range from {90-view} to {90+view}\nSamples removed due to angle: {rejected_angles} Minimum angle: {min_angle} Maximum angle: {max_angle}")
    print("Odds:", odd)
    print("Length dict: ", len(distances_dict))
    odds_ratio = 100 * odd / (odd + len(distances_dict))
    print("Odds ratio:", round(odds_ratio, 2), "%")

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
    SOURCE_PATH = 'dataset_YCB_train/DepthDeepSDF/files/untitled_1.txt'
    MESH_PATH = 'dataset_YCB_train/DepthDeepSDF/1a1c0a8d4bad82169f0594e65f756cf5/models/untitled.ply'
    DESTINATION_PATH = 'dataset_YCB_train/DepthDeepSDF/files/'

    input_file = ViewsFile(SOURCE_PATH)
    load_generator_file(input_file)

    output_file = File(SOURCE_PATH, DESTINATION_PATH)
    output_file.o_c_transformation = input_file.o_c_transformation

    input_mesh = load_file(MESH_PATH)
    centered_mesh = translate(input_mesh, input_file.s_o_transformation[:3])
    scaled_mesh, _ = scale(centered_mesh, input_file.scale)


    for view, frame in enumerate(input_file.frames):
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
        # plt.title('Pionhole camera image')
        # plt.show()
        depth_image = stack_images(output_file, scaled_mesh, camera)
        if depth_image is None:
            continue
        # plt.imshow(depth_image[:,:,0], cmap='gray')
        # plt.title('Pionhole camera image')
        # plt.show()

        scaled_mesh = translate(scaled_mesh, -frame[:3])
        scaled_mesh = rotate(scaled_mesh, -frame[3:])

        output_file.pixels.append(depth_image)
        output_file.save(view)
        output_file.pixels.clear()
        exit(777)
        # w tym programie musi być liczenie ścianek na promieniu, jeżeli:
        # unique(intersection) % 2 = 0 promień zostaje
        # unique(intersection) % 2 = 1 promień jest odrzucany
        # 100 * odrzucone promienie / wszystkie promienie > 5 widok jest odrzucony
        