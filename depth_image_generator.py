import argparse
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import random
import cv2
import json

from depth.utils import *
from depth.camera import Camera
from depth_file_generator import ViewFile, translate, scale, rotate, set_camera

POWER_FACTOR = 25
GT = False

class DepthImageFile():
    def __init__(self, name):
        self.name = name
        self.o_c_transformation = np.zeros(6)
        self.pixels = {}
        self.ds = 0
    
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

    def save(self, destination_dir, rotation_step, rejection_angle, view):
        pixels_serializable = {f"{key[0]},{key[1]}": list(value) for key, value in self.pixels.items()}
        data = {
            'o_c_transformation': self.o_c_transformation.tolist(),
            'f': self.f,
            'cx': self.cx,
            'cy': self.cy,
            'Ndx': self.Ndx,
            'Ndy': self.Ndy,
            'ds': self.ds,
            'nx': self.nx,
            'ny': self.ny,
            'z': self.z,
            'ndx': self.ndx,
            'ndy': self.ndy,
            'dz': self.dz,
            'dz2': self.dz2,
            'pixels': pixels_serializable
                }
        
        file_name = f"{self.name}_{rotation_step}_a{rejection_angle}_view{view}.json"
        file_path = os.path.join(destination_dir, file_name)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        print("FILE SAVED:", file_path)

    def load(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        self.o_c_transformation = np.array(data['o_c_transformation'])
        self.f = data['f']
        self.cx = data['cx']
        self.cy = data['cy']
        self.Ndx = data['Ndx']
        self.Ndy = data['Ndy']
        self.ds = data['ds']
        self.nx = data['nx']
        self.ny = data['ny']
        self.z = data['z']
        self.ndx = data['ndx']
        self.ndy = data['ndy']
        self.dz = data['dz']
        self.dz2 = data['dz2']
        self.pixels = {(int(k.split(',')[0]), int(k.split(',')[1])): np.array(v) for k, v in data['pixels'].items()}

        print(f"FILE LOADED: {file_path}")

    def visualize_as_point_cloud(self):
        points = []
        for (x, y), depth_values in self.pixels.items():
            for z in depth_values:
                if z > 0:  # Ignore points with zero depth
                    X = (x - self.cx) * z / self.f
                    Y = (y - self.cy) * z / self.f
                    points.append([X, Y, z])
        
        if not points:
            print("No valid points to display.")
            return

        points = np.array(points)
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(points)
        
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([self.point_cloud, origin])
        # exit(777)

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

    mask = diff_img > 1
    depth_image[mask, 0] = diff_img[mask]
    # depth_image[:, :, 0] = diff_img[:,:]
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

    # Convert depth image to dictionary format
    depth_image_dict = {}
    for y in range(depth_image.shape[0]):
        for x in range(depth_image.shape[1]):
            if depth_image[y, x, 0] != 0:  # Assuming 0 means no depth information
                depth_image_dict[(x, y)] = depth_image[y, x, :]

    file.pixels = depth_image_dict

    return depth_image[file.ny:file.ny+file.ndy, file.nx:file.nx+file.ndx, :]

if __name__ == '__main__':
    categories = ['mug']  # ['bottle', 'bowl', 'mug']
    experiment_name = 'new_exp_4'
    with open(f'examples/{experiment_name}/data/dataset_config.json', 'r') as json_file:
        config = json.load(json_file)
    
    for category in categories:
        generated_files = [gf.split('.')[0] for gf in os.listdir(f'examples/{experiment_name}/data/training_data/{category}') if gf.endswith('.json')]
        names_json = [name for name in os.listdir(f'examples/{experiment_name}/data/{category}') if name.endswith(f"_{config['rotation_step']}.json")]

        for current_iteration, name_json in enumerate(names_json):
            name = name_json.split('_')[0]
            SOURCE_PATH = os.path.join(f'examples/{experiment_name}/data/{category}', name_json)
            MESH_PATH =  os.path.join(f'ShapeNetCore/{category}', name, 'models/model_normalized.obj')

            view_file = ViewFile(name)
            view_file.load(SOURCE_PATH)

            depth_image_file = DepthImageFile(name)
            depth_image_file.o_c_transformation = view_file.o_c_transformation

            input_mesh = load_file(MESH_PATH)
            input_mesh = rotate(input_mesh, np.array([90, 0, 0]))
            centered_mesh = translate(input_mesh, view_file.s_o_transformation[:3])
            scaled_mesh, _ = scale(centered_mesh, view_file.scale)

            print()
            print("===================================================")
            print("SOURCE PATH", SOURCE_PATH)
            for view, frame in enumerate(view_file.frames):
                if not name + f"_{config['rotation_step']}_a{POWER_FACTOR}_view{view}" in generated_files:
                    print("File was processed earlier")
                    continue
                scaled_mesh = translate(scaled_mesh, frame[:3])
                scaled_mesh = rotate(scaled_mesh, frame[3:])

                mesh = o3d.t.geometry.TriangleMesh.from_legacy(scaled_mesh)
                scene = o3d.t.geometry.RaycastingScene()
                scene.add_triangles(mesh)

                camera_position = view_file.o_c_transformation[:3]
                camera_orientation = [np.deg2rad(angle) for angle in view_file.o_c_transformation[3:]]
                camera = set_camera(camera_position, camera_orientation)  # position, orientation in radians
                rays = camera.raycasting()
                depth_image_file.get_camera_parameters(camera.Fx, camera.Cx, camera.Cy)
                depth_image_file.get_image_resolution(camera.width, camera.height)
                
                # Depth image.
                ans = scene.cast_rays(rays)
                img = ans['t_hit'].numpy()
                ids = ans["primitive_ids"].numpy()

                img[img == np.inf] = 0
                img = img.astype(np.float32)

                # plt.imshow(img, cmap='gray')
                # plt.title('Input camera image')
                # plt.show()

                depth_image = stack_images(depth_image_file, scaled_mesh, camera)

                print(f"CURRENT ITERATION: {current_iteration} OUT OF {len(names_json)}")

                if depth_image is None:
                    scaled_mesh = translate(scaled_mesh, -frame[:3])
                    scaled_mesh = rotate(scaled_mesh, -frame[3:])
                    depth_image_file.pixels.clear()
                    print("DEPTH IMAGE IS NONE", name_json)
                    continue
                
                # plt.imshow(depth_image[:,:,0], cmap='gray')
                # plt.title('Output camera image')
                # plt.show()
                # plt.imshow(img, cmap='gray')
                # plt.title('Input camera image')
                # plt.show()
                # if GT:
                #     plt.savefig(os.path.join(output_file.destination_dir, output_file.name + '_' + str(view) + f'_gt.png'))
                # else:
                #     plt.savefig(os.path.join(output_file.destination_dir, output_file.name + '_' + str(view) + f'_a{POWER_FACTOR}.png'))
                scaled_mesh = translate(scaled_mesh, -frame[:3])
                scaled_mesh = rotate(scaled_mesh, -frame[3:])
                print(depth_image.shape, type(depth_image))
                depth_image_file.visualize_as_point_cloud()
                file_name = f"{depth_image_file.name}_{config['rotation_step']}_a{POWER_FACTOR}_view{view}.json"
                destination_path = f'examples/{experiment_name}/data/training_data/{category}'
                file_path = os.path.join(destination_path, file_name)
                depth_file_view = DepthImageFile(name)
                depth_file_view.load(file_path)
                depth_file_view.visualize_as_point_cloud()
                # depth_image_file.save(f'examples/{experiment_name}/data/training_data/{category}', config['rotation_step'], POWER_FACTOR, view)
                depth_image_file.pixels.clear()

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

            # zrobić dot product w rejection sampling do powierzchni - bardziej nieliniowa funkcja odrzucenia   !!!!

            # wygenerować mesh - kernel density? w 3D z Nearest Neighbour Search
            # idąc po promieniu dla każdego piksela akumuluję wartość dla punktów znajdujących się w promieniu r
            # ważyć je exp^{wartość/odległość} i zrobić wartość krytyczną (próg) dla której będzie powierzchnia

            # zwiększyć aureolę do 10 pikseli
            # 1. Wrzucić wynik do Marching Cubes. Jeżeli SDF jest poniżej jakiejś wartości progowej to dany voxel jest zajęty.
            # przesunąć do 0,0,0 
            # przeskalować
            # do MC
            
            # sprawdzić czy dane generowane są równolegle
            # alternatywa idziemy po promieniu i szukamy zmiany znaku wartości sdf - tam powstaje punkt należący do trójkąta

            # nasycenie nieliniowe kolorów do wizualizacji
            # dane do json
            # odwrócić wartości sdf - na zewnątrz 0 wewnątrz dodatnie wartości
            # ASAP wysłać dane
            # badanie na 1 obiekcie, 50 widoków

            # try new parameters in meshlab
            # 1. apply default params for edge cases
            # 2. try grid search for these cases to improve quality

            # Check directions for development
            # 1. increase number of objects up to 6, keep one similar object and one different for testing set
            # 2. increase number of views;  change increment in degrees: 5, 10, 15
            # 3. mesh reconstruction - check meaning of the params
            # 4. compare to deepsdf - include metrics, save point clouds at any step from deepsdf

            # 5. train with 6 objects - increase/keep value of increment
                # "10f6e09036350e92b3f21f1137c3c347_9_a25_k150_inp_test",

            # 6. przygotować tabele z obiketami i widokami
            # 7. w deepsdf skalowanie do 0.2
            # 8. wysłać pcd i ply z meshlab
            # 2/3 widoków do treningu i 1/3 widoków do testu
            # 6 obiektów - 4 do treningu 2 do testu