import argparse
import open3d as o3d
import matplotlib.pyplot as plt
import random

from depth.utils import *
from depth.camera import Camera

class File():
    def __init__(self, source_path, destination_dir=''):
        self.source_path = source_path
        self.destination_dir = destination_dir 
        self.name = self.get_name_()
        self.scale = 0
        self.s_o_transformation = np.zeros(6)
        self.o_c_transformation = np.zeros(6)
        self.frames = []

        if destination_dir:
            self.version = self.get_version_()

    def get_name_(self):
        head = os.path.split(self.source_path)[0]
        return head.split('/')[-2]
    
    def get_version_(self):
        dir_files = os.listdir(self.destination_dir)
        file_number = len([x for x in dir_files if x.startswith(self.name) and x.endswith('.txt')]) + 1
        return file_number
    
    def save(self):
        with open(os.path.join(self.destination_dir, self.name + '.txt'), 'w') as f:
            f.write(f'{self.scale}\n')
            f.write(f"{' '.join(map(str, self.s_o_transformation))}\n")
            f.write(f"{' '.join(map(str, self.o_c_transformation))}\n")
            for frame in self.frames:
                f.write(f"{' '.join(map(str, frame))}\n")
        print(f"Saved: {os.path.join(self.destination_dir, self.name + '.txt')}")


def set_camera(file):
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

    # Rotation angles in radians
    rotation_step = 0
    roll = np.deg2rad(-rotation_step)  # Rotation around X-axis
    pitch = np.deg2rad(135)  # Rotation around Y-axis
    yaw = np.deg2rad(270 - rotation_step)  # Rotation around Z-axis

    # Translation values
    tx = .0
    ty = .0
    tz = 1.5

    file.o_c_transformation = np.array([tx, ty, tz, np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)])

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
    max_idx = 0
    max_dist = 0
    for i in range(3):
        input_max_dist = np.max(mesh_vertices[:, i]) - np.min(mesh_vertices[:, i])
        max_dist = max(max_dist, input_max_dist)
        if max_dist == input_max_dist:
            max_idx = i

    if scale_to_unit:
        max_z_dist = np.max(mesh_vertices[:, max_idx]) - np.min(mesh_vertices[:, max_idx])
        mesh_vertices /= max_z_dist
    
    mesh_vertices *= scale_factor
    output_z_dist = np.max(mesh_vertices[:, max_idx]) - np.min(mesh_vertices[:, max_idx])
    mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
    scaled_mesh = translate(mesh, object_translation(mesh_vertices, True))

    real_scale_factor = output_z_dist / max_dist
    
    print(f"Max dist catured along axis: {max_idx}")

    return scaled_mesh, real_scale_factor

def create_directory(directory):
    # Check if the directory already exists
    if not os.path.exists(directory):
        # Create the directory if it doesn't exist
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully.")
    else:
        print(f"Directory '{directory}' already exists.")

if __name__ == '__main__':
    category = 'telephone'
    rotate_object = True
    names = os.listdir(f'ShapeNetCore/{category}')

    # print(list_dir)
    # exit(777)
    # names = [
    #     'dataset_YCB_train/DepthDeepSDF/4a6ba57aa2b47dfade1831cbcbd278d4/models/untitled.ply',
    #     'dataset_YCB_train/DepthDeepSDF/4d4fc73864844dad1ceb7b8cc3792fd/models/untitled.ply',
    #     'dataset_YCB_train/DepthDeepSDF/5c326273d61272ad4b6e06cda31f9bc6/models/untitled.ply',
    #     'dataset_YCB_train/DepthDeepSDF/1a1c0a8d4bad82169f0594e65f756cf5/models/untitled.ply',
    #     'dataset_YCB_train/DepthDeepSDF/1a97f3c83016abca21d0de04f408950f/models/untitled.ply',
    #     'dataset_YCB_train/DepthDeepSDF/1c9f9e25c654cbca3c71bf3f4dd78475/models/untitled.ply'
    # ]
    for name in names:
        SOURCE_PATH = os.path.join(f'ShapeNetCore/{category}', name, 'models/model_normalized.obj')
        DESTINATION_PATH = f'dataset_YCB_train/DepthDeepSDF/files/{category}'
        create_directory(DESTINATION_PATH)
        generated_file = File(SOURCE_PATH, DESTINATION_PATH)

        input_mesh = load_file(SOURCE_PATH)
        input_mesh = rotate(input_mesh, np.array([180, 0, 0]))

        s_o_vector = np.concatenate([s_o_translation(input_mesh), s_o_rotation(input_mesh)], axis=0)
        generated_file.s_o_transformation = s_o_vector

        centered_mesh = translate(input_mesh, s_o_vector[:3])
        scale_factor = 0.2
        scaled_mesh, generated_file.scale = scale(centered_mesh, scale_factor, True)
        translations_and_rotations = []

        if rotate_object:
            random_value = random.choice([0, 90, 180, 270])
        else:
            random_value = 0
        for i in range(10):
            translations_and_rotations.append(np.array([0.,0.,0.,0 ,0,i + random_value]))


        for i in translations_and_rotations:
            scaled_mesh = translate(scaled_mesh, i[:3])
            scaled_mesh = rotate(scaled_mesh, i[3:])

            mesh = o3d.t.geometry.TriangleMesh.from_legacy(scaled_mesh)
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(mesh)

            camera = set_camera(generated_file)
            rays = camera.raycasting()
            print(generated_file.name, generated_file.version, generated_file.scale, generated_file.o_c_transformation, generated_file.s_o_transformation)
            
            # Depth image.
            ans = scene.cast_rays(rays)
            img = ans['t_hit'].numpy()

            img[img == np.inf] = 0
            img = img.astype(np.float32)

            # plt.imshow(img, cmap='gray')
            # plt.title('Pionhole camera image')
            # plt.show()

            scaled_mesh = translate(scaled_mesh, -i[:3])
            scaled_mesh = rotate(scaled_mesh, -i[3:])

            generated_file.frames.append(i)
        # exit(777)
        generated_file.save()