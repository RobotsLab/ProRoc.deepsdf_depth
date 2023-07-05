import open3d as o3d
import matplotlib.pyplot as plt
import json
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import random
from scipy.spatial import KDTree


def raycasting(path, name):
    print(path)
    textured_mesh = o3d.io.read_triangle_mesh(path)
    cube = o3d.t.geometry.TriangleMesh.from_legacy(textured_mesh)
    # Create a scene and add the triangle mesh
    scene = o3d.t.geometry.RaycastingScene()
    try:
        cube_id = scene.add_triangles(cube)
    except:
        pass
    v1 = 0.666247
    v2 = 0.344917 
    v3 = 0.487054
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=60,
        center=[0.,0.,0.],
        eye=[0,1.5,1.5],
        up=[0,-1,0],
        width_px=640,
        height_px=480,
    )
    # We can directly pass the rays tensor to the cast_rays function.
    ans = scene.cast_rays(rays)
    img = ans['t_hit'].numpy()
    ROI = img[img != np.inf]
    # points out of ROI values == 0
    img[img == np.inf] = 0
    img = img.astype(np.float32)
    print(np.max(ROI))
    print(img)
    # np.savez(f'dataset_YCB_train/depth/depth_{name}.npz', img)
    # img2 = np.load(f'dataset_YCB_train/depth/depth_{name}.npz')
    # img2 = img2[img2.files[0]]
    plt.imshow(img, cmap='gray')
    plt.show()
    # print(img2[img2.files[0]])

    #zapisać jako ply
    # hit = ans['t_hit'].isfinite()
    # points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))
    # pcd = o3d.t.geometry.PointCloud(points)
    # o3d.t.io.write_point_cloud(f"raycasted_samples/rc_60_{name}.pcd", pcd)

    # exit(5)

def save_to_json():
    path = "/home/piotr/Desktop/ProRoc/DeepSDF/dataset_YCB_train"
    dirs = os.listdir(path)
    test_set = []
    test_dict = {"dataset_YCB_train" : {}}

    for d in dirs:
        if d != 'bottle' and d != 'mug':
            continue
        category_dir = '/'.join((path, d))
        instance_name = sorted(os.listdir(category_dir))

        for name in instance_name:
            filename = os.path.splitext(name)[0]
            # if not '_' in filename:
            print(filename)
            test_set.append(filename)
        test_dict["dataset_YCB_train"][f'{d}'] = test_set
        test_set = []
        print('----------------------')

    # print(train_dict)
    print('----------------------')
    # print(test_dict)

    # Serializing json
    # json_train = json.dumps(train_dict, indent=4)
    json_test = json.dumps(test_dict, indent=4)
    
    # Writing to sample.json
    # with open("/home/piotr/Desktop/ProRoc/DeepSDF/examples/splits/PPRAI_train.json", "w") as outfile:
        # outfile.write(json_train)
    with open("/home/piotr/Desktop/ProRoc/DeepSDF/examples/splits/YCB_bottle_mug_train.json", "w") as outfile:
        outfile.write(json_test)

def load_to_rc():
    # Opening JSON file
    with open('examples/splits/YCB_bottle_mug_train.json', 'r') as openfile:

        # Reading from json file
        json_object = json.load(openfile)

    keyList = [key for key in json_object['dataset_YCB_train'].keys()]
    for k in keyList:
        for obj in json_object['dataset_YCB_train'][k]:
            # if obj.endswith('_x0y0z0'):
            raycasting('/'.join(['dataset_YCB_train', k, obj, 'models/model_normalized.obj']), obj)

def pcd_to_npz():
    npz = np.load('/home/piotr/Desktop/ProRoc/DeepSDF/data_PPRAI/SdfSamples/Dataset_PPRAI/bottle/2bbd2b37776088354e23e9314af9ae57.npz')
    d = dict(zip(("{}".format(k) for k in npz), (npz[k] for k in npz)))
    pos_data = d['pos'][:, 0:3]
    neg_data = d['neg'][:, 0:3]
    neg_pcd = o3d.geometry.PointCloud()
    pos_pcd = o3d.geometry.PointCloud()
    neg_pcd.points = o3d.utility.Vector3dVector(neg_data)
    pos_pcd.points = o3d.utility.Vector3dVector(pos_data)
    o3d.visualization.draw_geometries([pos_pcd])
    print(min(np.asarray(pos_pcd.points)[:, 0]),max(np.asarray(pos_pcd.points)[:, 0]))
    # o3d.io.write_point_cloud("/home/piotr/Desktop/ProRoc/DeepSDF/others/1c47c52b9f57f348e44d45504b29e834.ply", neg_pcd)

def pcd_to_obj_ycb():
    source_path = '/home/piotr/Desktop/ProRoc/DeepSDF/ycb1/objects-proc3'
    pcd_list = sorted([x for x in os.listdir(source_path) if x.endswith('.pcd')])

    for i, name in enumerate(pcd_list):
        pcd = o3d.io.read_point_cloud(source_path + '/' + pcd_list[i])
        print(source_path + '/' + pcd_list[i])
        points = np.asarray(pcd.points)
        points = points[~np.isnan(points)]
        points = np.reshape(points, (-1, 3))
        pcd.points = o3d.utility.Vector3dVector(points)

        tree = ET.parse(source_path + '/data.xml')
        # Get the root element
        root = tree.getroot()

        # Loop through each item element
        for item in root.iter('item'):
            # Get the prefix attribute
            prefix = item.attrib['prefix']
            start = name.find('-item-')  # find the starting index of the prefix
            end = name.find('.pcd')  # find the ending index of the prefix
            item_name = name[start:end]  # extract the prefix using slicing
            
            # Check if the prefix equals a specific value
            if prefix == item_name:
                # Get the values of v1, v2, and v3
                v1 = float(item.find('extrinsic').attrib['v1'])
                v2 = float(item.find('extrinsic').attrib['v2'])
                v3 = float(item.find('extrinsic').attrib['v3'])
                
                # Print the values
                print(f"v1={v1}, v2={v2}, v3={v3}")
                break

        # # Define rotation angle (in radians)
        # angle = -np.pi/2

        # # Define rotation matrix
        # rotation_matrix = np.eye(3)
        # rotation_matrix[1:3,1:3] = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        # pcd.estimate_normals()
        pcd.normals = o3d.utility.Vector3dVector(np.array([2*v1,2*v2-0.7,2*v3]) - pcd.points)
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)

        # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
        # exit(6)

        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist
        bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))

        o3d.visualization.draw_geometries([bpa_mesh, origin])
        
        bpa_mesh.triangle_normals = o3d.utility.Vector3dVector([])

        # Rotate point cloud
        # bpa_mesh.rotate(rotation_matrix, center=[0,0,0])

        destination_path = f"/home/piotr/Desktop/ProRoc/DeepSDF/Test_PPRAI.v2/ycb/{name.split('.')[0]}/models"
        # if not os.path.exists(destination_path):
            # os.makedirs(destination_path)
        # print(np.mean(bpa_mesh.vertices, axis=0))
        # o3d.io.write_triangle_mesh(destination_path + '/model_normalized.obj', bpa_mesh, write_vertex_normals=False, write_vertex_colors=False)
        # print(destination_path)
        # if i == 5:
        # exit(6)

def pcd_to_obj():
    source_path = '/home/piotr/Desktop/ProRoc/DeepSDF/new_shapeNetGraspable_30instances_DexNet_Test'
    classess = os.listdir(source_path)

    for c in classess:
        class_path = source_path + '/' + c
        pcd_list = sorted([x for x in os.listdir(class_path) if x.endswith('.pcd') and x.startswith('cloudG')])
        names_list = sorted([x for x in os.listdir(class_path) if x.endswith('.dat')])
        print(c)

        for i in range(len(pcd_list)):
            with open(class_path + '/' + names_list[i]) as f:
                line = f.readline()
            instance_name = line.split()[1]
            with open(class_path + '/camPoses.txt') as f:
                lines = f.readXlines()
            x = float(lines[i].split()[7])
            y = float(lines[i].split()[11])
            z = float(lines[i].split()[15])

            pcd = o3d.io.read_point_cloud(class_path + '/' + pcd_list[i])

            # Define rotation angle (in radians)
            angle = -np.pi/2

            # Define rotation axis (in this case, the z-axis)
            axis = np.array([1, 0, 0])

            # Define rotation matrix
            rotation_matrix = np.eye(3)
            rotation_matrix[1:3,1:3] = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

            pcd.estimate_normals()
            pcd.normals = o3d.utility.Vector3dVector(np.array([x,y,z]) - pcd.points)

            # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
            
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 3 * avg_dist
            bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))

            print(x,y,z)

            # o3d.visualization.draw_geometries([bpa_mesh])
            
            bpa_mesh.triangle_normals = o3d.utility.Vector3dVector([])

            # Rotate point cloud
            bpa_mesh.rotate(rotation_matrix, center=[0,0,0])


            name_itr = 0

            if os.path.exists(f"/home/piotr/Desktop/ProRoc/DeepSDF/Test_PPRAI.v2/{c.lower()}"):
                obj_list = os.listdir(f"/home/piotr/Desktop/ProRoc/DeepSDF/Test_PPRAI.v2/{c.lower()}")
                for o in obj_list:
                    if instance_name == o[:-2]:
                        name_itr += 1

            new_name = instance_name + '_' + str(name_itr)

            destination_path = f"/home/piotr/Desktop/ProRoc/DeepSDF/Test_PPRAI.v2/{c.lower()}/{new_name}/models"
            if not os.path.exists(destination_path):
                os.makedirs(destination_path)
            
            o3d.io.write_triangle_mesh(destination_path + '/model_normalized.obj', bpa_mesh, write_vertex_normals=False, write_vertex_colors=False)
            print(destination_path)

            exit(6)

#  1. (filename: str, mesh: open3d.cuda.pybind.geometry.TriangleMesh, write_ascii: bool = False, compressed: bool = False, write_vertex_normals: bool = True,
#  write_vertex_colors: bool = True, write_triangle_uvs: bool = True, print_progress: bool = False) -> bool


def npz_overview():
    source_path = '/home/piotr/Desktop/ProRoc/DeepSDF/data_YCB/SdfSamples/dataset_YCB_test'  #  /home/piotr/Desktop/ProRoc/DeepSDF/data_PPRAI/SdfSamples/Dataset_PPRAI
    classess = os.listdir(source_path)
    list_by_samples = []
    class_neg_value = []
    
    for c in classess:
        print(c)
        avoid = ['ycb']
        if not c in avoid:
            continue
        class_path = os.path.join(source_path, c)
        # names_list = sorted([x for x in os.listdir(class_path) if '_' in x])
        names_list = sorted(os.listdir(class_path))
        print(names_list)
        # add th for laptop and mug
        for n in names_list:
            npz_file = np.load(os.path.join(class_path, n))
            print(c, n)

            pos_data = npz_file[npz_file.files[0]]
            neg_data = npz_file[npz_file.files[1]]
            # Extract x, y, z, and value from the data
            pos_x = pos_data[:, 0]
            pos_y = pos_data[:, 1]
            pos_z = pos_data[:, 2]
            pos_value = pos_data[:, 3]

            neg_x = neg_data[:, 0]
            neg_y = neg_data[:, 1]
            neg_z = neg_data[:, 2]
            neg_value = neg_data[:, 3]

            # # quantiles
            # Q1 = np.quantile(neg_value, 0.25)
            # Q3 = np.quantile(neg_value, 0.75)
            # IRQ = abs(Q3 - Q1)

            # # filtering mask
            # filter_mask = np.all([[Q3 + IRQ * 1.5 >= neg_value], [neg_value >= Q1 - IRQ * 1.5]], axis=0)
            # filter_mask = filter_mask.reshape(-1)
            # newarr = neg_value[filter_mask]

            # # class_neg_value.append(newarr.tolist())
            # for n in neg_value.tolist():
            #     class_neg_value.append(n)

            # continue

            print('mean: ', np.mean(neg_value))
            print('std_dev: ', np.std(neg_value))
            print('min: ', min(neg_value))
            print('max: ', max(neg_value))
            print('----------------')
            source = os.path.join(c, n)
            samples = len(pos_data) + len(neg_data)
            list_by_samples.append((source, samples))

            # neg_mean = np.mean(neg_data[:, 3])
            # neg_std = np.std(neg_data[:, 3])

            threshold = 0.031232145173622754
            sdfs_filtered = [sdfs for sdfs in neg_data if abs(sdfs[3]) <= threshold]
            sdfs_filtered = np.asarray(sdfs_filtered)
            print('mean: ', np.mean(sdfs_filtered[:, 3]))
            print('std_dev: ', np.std(sdfs_filtered[:, 3]))
            print('min: ', min(sdfs_filtered[:, 3]))
            print('max: ', max(sdfs_filtered[:, 3]))
            print('----------------')
            print('----------------')

            data = {"pos": pos_data, "neg": sdfs_filtered}
            np.savez(os.path.join("/home/piotr/Desktop/ProRoc/DeepSDF/data_YCB/SdfSamples/dataset_YCB_test/ycb_mean", n), **data)
            # Create the figure
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plot the positive data
            # ax.scatter(pos_x, pos_y, pos_z, c=pos_value, cmap='viridis', marker='o')

            # Plot the negative data
            # ax.scatter(x, y, z, c=sdf, cmap='viridis', marker='x')
            ax.scatter(neg_x, neg_y, neg_z, c=neg_value, cmap='viridis', marker='x')


            # Set the labels for the x, y, and z axes
            ax.set
            plt.show()
    # Q1 = np.quantile(class_neg_value, 0.25)
    # Q3 = np.quantile(class_neg_value, 0.75)
    # IRQ = abs(Q3 - Q1)
    # print('Q1:', Q1, 'Q3:', Q3, 'IRQ:', IRQ, 'median:', np.median(class_neg_value), 'mean:', np.mean(class_neg_value))
    # plt.boxplot(class_neg_value)
    # plt.title(f'all')
    # plt.show()
    # list_by_samples.sort(key=lambda a: a[1])
    # print(list_by_samples, len(list_by_samples), '/', 22*len(classess))

def calculate_distance(vertice, center=''):
    if center:
        vertice -= center
        return np.sqrt(sum([np.power(vertice[0], 2), np.power(vertice[1], 2), np.power(vertice[2], 2)]))
    else:
        return np.sqrt(sum([np.power(vertice[0], 2), np.power(vertice[1], 2), np.power(vertice[2], 2)]))

def rescaling():
    obj_path = "/home/piotr/Desktop/ProRoc/DeepSDF/dataset_YCB_test/ycb/"  # /data-item-1-1/models/model_normalized.obj"
    ply_path = '/home/piotr/Desktop/ProRoc/DeepSDF/examples/YCB/Reconstructions/2000/Meshes/dataset_YCB_test_ytop/ycb_median/'
    # ply_path = '/home/piotr/Desktop/ProRoc/DeepSDF/ycb1/model2/Reconstructions_noth/2000/Meshes/Dataset_PPRAI/ycb/data-item-1-1.ply'
    
    objects = sorted(os.listdir(obj_path))
    names = ['data-item-2-1', 'data-item-3-1', 'data-item-7-2', 'data-item-8-2', 'data-item-9-2']
    for obj_name in objects:
        obj = o3d.io.read_triangle_mesh(os.path.join(obj_path, obj_name, "models/model_normalized.obj"))
        obj_vertices = np.asarray(obj.vertices)
        print(obj_name)

        obj_center = obj_vertices.mean(axis=0)
        print("Obj file center:", obj_center)
        obj_maxDistance = 0
        for v in obj_vertices:
            distance = calculate_distance(np.copy(v), center=list(obj_center))
            obj_maxDistance = max(obj_maxDistance, distance)

        buffer = 1.03
        obj_maxDistance *= buffer
        print("Obj file max distance:", obj_maxDistance)

        # rotate it -90 by x axis
        ply = o3d.io.read_triangle_mesh(ply_path + obj_name + ".ply")
        ply_vertices = np.asarray(ply.vertices)
        if not np.any(ply_vertices):
            continue

        radians = np.pi/2
        x_rotation_matrix = np.array([[1, 0, 0],
                [0, np.cos(radians), -np.sin(radians)],
                [0, np.sin(radians), np.cos(radians)]])
        ply.rotate(x_rotation_matrix, center=[0,0,0])

        ply_center = ply_vertices.mean(axis=0)
        print("Original ply file center:", ply_center)
        # ply_vertices -= ply_center

        ply_vertices *= obj_maxDistance
        # ply_vertices += ply_center
        # ply_vertices -= obj_center

        print("Translated ply file center:", ply_vertices.mean(0))
        # exit(8)
        destination_path = "/home/piotr/Desktop/ProRoc/DeepSDF/examples/YCB/Reconstructions/2000/Meshes/dataset_YCB_test_ytop/translated3_ycb_median"

        itr = 1
        result = np.array((ply_vertices.shape))
        best_score = 1.
        while itr < 1000:

            tree = KDTree(ply_vertices)
            # Query the KDTree to find the farthest points
            distances, indices = tree.query(obj_vertices)
            # indices = np.flip(indices, axis=0)
            closest_points = ply_vertices[indices]
            # Compute the squared differences between array1 and the closest points
            squared_diff = np.square(obj_vertices - closest_points)
            mean_diff = np.mean(obj_vertices - closest_points, axis=0)
            directions_diff = mean_diff / abs(mean_diff)
            # print(directions_diff)

            mse = np.mean(squared_diff, axis=0)
            rmse = np.sqrt(mse)

            # print("Mean Square Error:", mse, rmse)
            
            ply_vertices += rmse * directions_diff

            # print(f"ply {ply_vertices.mean(axis=0)}, obj {obj_vertices.mean(axis=0)}")
            ply.vertices = o3d.utility.Vector3dVector(ply_vertices)

            if np.sum(rmse) < best_score:
                result = np.copy(ply_vertices)
                print(itr)
                itr = 0
                best_score = np.sum(rmse)
                print("Mean Square Error:", mse, rmse)


            previous_mse = mse
            previous_rmse = rmse
            itr += 1

        ply.vertices = o3d.utility.Vector3dVector(result)

        # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

        # vis = o3d.visualization.Visualizer()
        # visualize = list()
        # visualize.append(obj)
        # visualize.append(ply)
        # visualize.append(origin)
        # for x in visualize:
        #     vis.add_geometry(x, reset_bounding_box=False)

        # o3d.visualization.draw_geometries(visualize, mesh_show_back_face=True)
        # vis.clear_geometries()
        # vis.destroy_window()
        # save_it = input("Do you want to save it?[y/n] ")
        # if save_it.lower() == 'y':
        o3d.io.write_triangle_mesh(f"{os.path.join(destination_path, obj_name)}.ply", ply)
        # exit(190)


def mesh_normalization(mesh, origin, obj_path, filename):
    txt = input('Do you want to rotate it? [y/n]')

    if txt == 'y':
        # Compute the center of the mesh
        center = mesh.get_center()

        # Translate the mesh to the origin
        mesh.translate(-center)

        # Scale the mesh to fit within a 1x1x1 bounding box
        max_coord = max(mesh.get_max_bound() - mesh.get_min_bound())
        mesh.scale(1 / max_coord, center=(0, 0, 0))

        # Define rotation angle (in radians)
        angle = -np.pi/2

        # Define rotation matrix
        x_rotation_matrix = np.eye(3)
        x_rotation_matrix[1:3,1:3] = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        y_rotation_matrix = np.array([[np.cos(angle), 0, np.sin(angle)],
                        [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]])
        mesh.rotate(x_rotation_matrix, center=[0,0,0])

        # Compute the center of the mesh
        center = mesh.get_center()

        # Translate the mesh to the origin
        mesh.translate(-center)

        # Set view control parameters
        o3d.visualization.draw_geometries([mesh, origin])

        txt2 = input('Do you want to save it? [y/n]')

        if txt2 == 'y':
            o3d.io.write_triangle_mesh(os.path.join(obj_path, filename, 'models/model_normalized.obj'), mesh, write_vertex_normals=False, write_vertex_colors=False)


def rotate_object():
    obj_path = "/home/piotr/Desktop/ProRoc/DeepSDF/dataset_YCB_train/mug/"
    dest_path = "/home/piotr/Desktop/ProRoc/DeepSDF/dataset_YCB_train/mug/"
    obj_filename = [name for name in sorted(os.listdir(obj_path)) if not '_' in name]
    for filename in obj_filename:
        
        for files in os.listdir(os.path.join(obj_path, filename, "models")):
            if files.endswith('model_normalized.obj'):
                source_path = os.path.join(obj_path, filename, 'models', files)
        
        print(source_path)
        
        mesh = o3d.io.read_triangle_mesh(source_path)
        mesh.compute_vertex_normals()
        
        # Create an origin point
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        
        x_rads = [0, random.choice([-90, 90])]
        # y_rads = [0, 60, 120, 180, 240, 300]
        y_rads = [0, random.choice([60, 120, 180, 240, 300])]

        # for x_rad in x_rads:
        #     x_rotation_matrix = np.array([[1, 0, 0],
        #                     [0, np.cos(x_rad), -np.sin(x_rad)],
        #                     [0, np.sin(x_rad), np.cos(x_rad)]])

        #     for y_rad in y_rads:
        #         if x_rad == 0 and y_rad == 0:
        #             continue
        #         woking_mesh = o3d.geometry.TriangleMesh(mesh)
        #         woking_mesh.rotate(x_rotation_matrix, center=[0,0,0])
        #         y_rotation_matrix = np.array([[np.cos(y_rad), 0, np.sin(y_rad)],
        #                     [0, 1, 0],
        #                     [-np.sin(y_rad), 0, np.cos(y_rad)]])
        #         woking_mesh.rotate(y_rotation_matrix, center=[0,0,0])

        #         destination_path = os.path.join(dest_path, filename + f'_x{x_rad}y{y_rad}z0','models')
        #         os.makedirs(destination_path, exist_ok=True)
        #         print(destination_path)
        #         if os.path.exists(destination_path + '/model_normalized.obj'):
        #             continue
        #         else:
        #             o3d.io.write_triangle_mesh(
        #                 destination_path + '/model_normalized.obj',
        #                 woking_mesh, write_vertex_normals=False,
        #                 write_vertex_colors=False)

        # FOR MANUAL USAGE
        while True:
            woking_mesh = o3d.geometry.TriangleMesh(mesh)
            o3d.visualization.draw_geometries([woking_mesh, origin])
            print(filename)
            print("X-axis: Red, Y-axis: Green, Z-axis: Blue")
            key = input("Type 'n' for next or 'q' for quit: ")
            if key == 'q':
                exit(777)
            elif key == 'n':
                break
            axis = input('Axis of rotation: ')
            angle = input('Degrees: ')
            angles = angle.split(' ')
            try:
                x_rad = np.radians(int(angles[0]))
                y_rad = np.radians(int(angles[1]))
                z_rad = np.radians(int(angles[2]))
            except IndexError:
                print(f'List is shorter than 3. List length is {len(angles)}')
                z_rad = 0

            if 'x' in axis.lower():
                rotation_matrix = np.array([[1, 0, 0],
                            [0, np.cos(x_rad), -np.sin(x_rad)],
                            [0, np.sin(x_rad), np.cos(x_rad)]])
                woking_mesh.rotate(rotation_matrix, center=[0,0,0])

            if 'y' in axis.lower():
                rotation_matrix = np.array([[np.cos(y_rad), 0, np.sin(y_rad)],
                            [0, 1, 0],
                            [-np.sin(y_rad), 0, np.cos(y_rad)]])
                woking_mesh.rotate(rotation_matrix, center=[0,0,0])

            if 'z' in axis.lower():
                rotation_matrix = np.array([[np.cos(z_rad), -np.sin(z_rad), 0],
                            [np.sin(z_rad), np.cos(z_rad), 0],
                            [0, 0, 1]])                
                woking_mesh.rotate(rotation_matrix, center=[0,0,0])

            o3d.visualization.draw_geometries([woking_mesh, origin])

            save = input("Do you want to save this mesh? [y/n]")

            if save.lower() == 'y':
                destination_path = os.path.join(obj_path, filename + f'_x{angles[0]}y{angles[1]}z{angles[2]}','models')
                os.makedirs(destination_path, exist_ok=True)
                print(destination_path)
                o3d.io.write_triangle_mesh(
                    destination_path + '/model_normalized.obj',
                    woking_mesh, write_vertex_normals=False,
                    write_vertex_colors=False)


def main():
    # pcd_to_npz()
    load_to_rc()
    # pcd_to_obj()
    # npz_overview()
    # save_to_json()
    # pcd_to_obj_ycb()
    # rescaling()
    # rotate_object()


if __name__ == "__main__":
    main()