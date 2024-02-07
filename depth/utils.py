import open3d as o3d
import numpy as np
import os

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
    
def u_distribution(normal_distribution, mean):
    '''Changes normal distribution to U-shaped distribution'''
    result = np.array(normal_distribution)
    result = np.where(result < mean, mean - result, result)
    result = np.where(result > mean, 3 * mean - result, result)
    # print(result)
    return result

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
