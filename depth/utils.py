import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt

import logging
import torch


def load_file(path: str):
    if path.endswith(".npz"):
        dict_data = np.load(path)
        return dict_data
    elif path.endswith(".pcd"):
        pcd = o3d.io.read_point_cloud(path)
        return pcd
    elif path.endswith(".obj") or path.endswith(".ply"):
        mesh = o3d.io.read_triangle_mesh(path)
        return mesh
    
def u_distribution(normal_distribution, mean, sigma):
    '''Changes normal distribution to U-shaped distribution'''
    normal_distribution -= mean
    min_value = np.min(normal_distribution)
    max_value = np.max(normal_distribution)
    normal_distribution[normal_distribution > 0] += mean - max_value
    normal_distribution[normal_distribution < 0] += mean + abs(min_value)
    normal_distribution[normal_distribution == 0] += mean

    # count, bins, ignored = plt.hist(normal_distribution, 30, density=True)

    # plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mean)**2 / (2 * sigma**2) ), linewidth=2, color='r')

    # plt.show()

    return normal_distribution

def vec_towards_triangle(triangle, vertices):
    '''Find vector directed towards center of triangle.'''
    vert1, vert2, vert3 = triangle[0], triangle[1], triangle[2]
    target_point = np.mean([vertices[vert1], vertices[vert2], vertices[vert3]], axis=0)

    return target_point

def magnitude(vector):
    '''Calculate length of vector of every dimension'''
    return np.sqrt(sum(pow(element, 2) for element in vector))

def visualize_pcd():
    source_path = '/home/piotr/Desktop/ProRoc/DeepSDF/ycb1/depth_to_pcd/'
    pcd_list = sorted([x for x in os.listdir(source_path) if x.endswith('.pcd')])

    for i, name in enumerate(pcd_list):
        pcd = o3d.io.read_point_cloud(source_path + '/' + pcd_list[i])
        print(source_path + '/' + pcd_list[i])
        points = np.asarray(pcd.points)
        points = points[~np.isnan(points)]
        points = np.reshape(points, (-1, 3))
        print(np.mean(points, axis=0))
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd])

def add_common_args(arg_parser):
    arg_parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        action="store_true",
        help="If set, debugging messages will be printed",
    )
    arg_parser.add_argument(
        "--quiet",
        "-q",
        dest="quiet",
        default=False,
        action="store_true",
        help="If set, only warnings will be printed",
    )
    arg_parser.add_argument(
        "--log",
        dest="logfile",
        default=None,
        help="If set, the log will be saved using the specified filename.",
    )


def configure_logging(args):
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter("DeepSdf - %(levelname)s - %(message)s")
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    if args.logfile is not None:
        file_logger_handler = logging.FileHandler(args.logfile)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)


def decode_sdf(decoder, latent_vector, queries):
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
    else:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], 1)

    sdf = decoder(inputs)

    return sdf
