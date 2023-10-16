import open3d as o3d
import numpy as np
import os
from depth_preprocessing import *

    
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