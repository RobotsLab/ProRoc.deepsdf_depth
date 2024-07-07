import argparse
import open3d as o3d
import matplotlib.pyplot as plt
import random
import json

from depth.utils import *
from depth.camera import Camera

class ViewFile():
    def __init__(self, name):
        self.name = name
        self.scale = 0
        self.s_o_transformation = np.zeros(6)
        self.o_c_transformation = np.zeros(6)
        self.frames = []
    
    def save(self, destination_path, step):
        data = {
            'scale': self.scale,
            's_o_transformation': self.s_o_transformation.tolist(),
            'o_c_transformation': self.o_c_transformation.tolist(),
            'frames': [frame.tolist() for frame in self.frames]
        }
        with open(os.path.join(destination_path, self.name + f'_{step}.json'), 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved: {os.path.join(destination_path, self.name + f'_{step}.json')}")
        print(data)

    def load(self, source_path):
        with open(source_path, 'r') as f:
            data = json.load(f)
            self.scale = data['scale']
            self.s_o_transformation = np.array(data['s_o_transformation'])
            self.o_c_transformation = np.array(data['o_c_transformation'])
            self.frames = [np.array(frame) for frame in data['frames']]
        print(f"Loaded: {source_path}")
        print(data)

def set_camera(position, rotation):
    '''This function is used to set camera position. Rotation must be in radians.'''
    camera = Camera()
    camera.rotate(*rotation)
    camera.translate(*position)
    return camera

def get_s_o_translation(mesh):
    mesh_vertices = np.copy(np.asarray(mesh.vertices))
    mesh_center = np.mean(mesh_vertices, axis=0)
    mesh_vertices -= mesh_center
    min_z = np.min(mesh_vertices[:, 2])
    mesh_translation_vector = - mesh_center - [0, 0, min_z]

    return mesh_translation_vector

def get_s_o_rotation(mesh):
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
    mesh_vertices += translation
    mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
    return mesh

def rotate(mesh, rotation):
    R = mesh.get_rotation_matrix_from_xyz(np.radians(rotation))
    mesh.rotate(R, center=(0, 0, 0))
    return mesh

def scale(mesh, scale_factor):
    mesh_vertices = np.copy(np.asarray(mesh.vertices))    
    max_idx = 0
    max_dist = 0
    for i in range(3):
        input_max_dist = np.max(mesh_vertices[:, i]) - np.min(mesh_vertices[:, i])
        max_dist = max(max_dist, input_max_dist)
        if max_dist == input_max_dist:
            max_idx = i

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
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully.")
    else:
        print(f"Directory '{directory}' already exists.")

if __name__ == '__main__':
    categories = ['bottle', 'bowl', 'laptop', 'mug', 'can', 'jar']
    random_rotation = False
    experiment_name = 'new_exp_1'

    for category in categories:
        names = os.listdir(f'ShapeNetCore/{category}')
        refused_names = []
        generated_files = [gf.split('_')[0] for gf in os.listdir(f'examples/{experiment_name}/data/{category}') if gf.endswith('.json')]
        if len(generated_files) >= 10:
            continue
        accepted_names = len(generated_files)
        with open(f'examples/{experiment_name}/data/{category}/refused_names.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                generated_files.append(line.rstrip('\n'))

        for name in names:
            if name in generated_files:
                continue
            SOURCE_PATH = os.path.join(f'ShapeNetCore/{category}', name, 'models/model_normalized.obj')
            DESTINATION_PATH = f'examples/{experiment_name}/data/{category}'
            create_directory(DESTINATION_PATH)
            generated_file = ViewFile(name)

            input_mesh = load_file(SOURCE_PATH)
            input_mesh = rotate(input_mesh, np.array([90, 0, 0]))

            s_o_translation = get_s_o_translation(input_mesh)
            s_o_rotation = get_s_o_rotation(input_mesh)
            s_o_vector = np.concatenate([s_o_translation, s_o_rotation], axis=0)
            
            generated_file.s_o_transformation = s_o_vector

            centered_mesh = translate(input_mesh, s_o_vector[:3])
            scale_factor = 0.2
            scaled_mesh, real_scale_factor = scale(centered_mesh, scale_factor)
            generated_file.scale = scale_factor
            translations_and_rotations = []
            
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            o3d.visualization.draw_geometries([scaled_mesh, origin])
            
            user_opinion = input("Save the file? [y/n]")
            if user_opinion.lower() == 'n':
                refused_names.append(name)
                continue
            else:
                accepted_names += 1
            
            if random_rotation:
                random_value = random.choice([0, 90, 180, 270])
            else:
                random_value = 0

            step = 30

            for i in range(10):
                translations_and_rotations.append(np.array([0.,0.,0.,0 ,0,i*step + random_value]))

            for i in translations_and_rotations:
                scaled_mesh = translate(scaled_mesh, i[:3])
                scaled_mesh = rotate(scaled_mesh, i[3:])

                mesh = o3d.t.geometry.TriangleMesh.from_legacy(scaled_mesh)
                scene = o3d.t.geometry.RaycastingScene()
                scene.add_triangles(mesh)

                rotation_step = 0
                camera_position = [0., 0., 1.5]
                camera_rotation_degrees = [-rotation_step, 135, 270 - rotation_step]
                camera_rotation_radians = [np.deg2rad(angle) for angle in camera_rotation_degrees]
                camera = set_camera(camera_position, camera_rotation_radians)
                camera_position.extend(camera_rotation_degrees)
                generated_file.o_c_transformation = np.array(camera_position)

                rays = camera.raycasting()
                print(generated_file.name, generated_file.scale, generated_file.o_c_transformation, generated_file.s_o_transformation)
                
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

            generated_file.save(f'examples/{experiment_name}/data/{category}', step)
            generated_file.load(os.path.join(f'examples/{experiment_name}/data/{category}', generated_file.name + f'_{step}.json'))
            with open(f'examples/{experiment_name}/data/{category}/refused_names.txt', 'a') as f:
                for line in refused_names:
                    f.write(f"{line}\n")
            if accepted_names == 10:
                break
