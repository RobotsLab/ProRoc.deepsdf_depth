#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import numpy as np
import plyfile
from skimage import measure
import time
import torch
import open3d as o3d
import random

import deep_sdf.utils


def create_mesh(
    decoder, latent_vec, filename, N=256, max_batch=32 ** 2, offset=None, scale=None, input_data=None
):
    start = time.time()
    ply_filename = filename
    print(filename)
    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())

    samples = prepare_samples(input_data)


    num_samples = N ** 2

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:2].cuda()

        samples[head : min(head + max_batch, num_samples), 2] = (
            deep_sdf.utils.decode_sdf(decoder, latent_vec, sample_subset)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 2]
    print("SDF VALUES:", sdf_values)
    # tutaj wywołać funkcję
    save_sdf_samples(
        samples=samples,
        filename=filename
    )

    closest_cube_root = int(np.floor(samples.shape[0] ** (1/3)))
    print("CLOSEST CUBE ROOT", closest_cube_root)
    diff_to_remove = samples.shape[0] - closest_cube_root ** 3
    print("DIFF TO REMOVE", diff_to_remove)
    rows = np.array(random.sample(range(samples.shape[0]-diff_to_remove), diff_to_remove))
    print(np.unique(rows).shape)
    print("Rows", rows, rows.shape)
    print(samples, samples.shape, type(samples))
    samples = np.delete(samples, rows, 0)
    # samples = original_sampling(N, overall_index, voxel_size, voxel_origin)
    print(samples, samples.shape, type(samples))
    sdf_values = samples[:, 2]

    sdf_values = sdf_values.reshape(closest_cube_root, closest_cube_root, closest_cube_root)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()
    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    # result = measure._marching_cubes_lewiner(...)
    verts, faces, normals, values = measure.marching_cubes(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale

    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )

def save_sdf_samples(
        samples, 
        filename
):
    samples_numpy = np.asarray(samples)
    data = {"pos": samples_numpy[samples_numpy[:, 2] > 0], "neg" : samples_numpy[samples_numpy[:, 2] < 0]}
    np.savez(filename + ".npz", **data)
    print(f"NPZ file saved in {filename}")

def prepare_samples(data):
    unique_rd, inverse_indices = np.unique(data[:, 0], return_inverse=True)
    unique_rd = unique_rd[inverse_indices]

    number_of_samples = 10
    repeated_values = np.repeat(unique_rd, number_of_samples)

    # Create the second column with linspace values
    second_column_values = np.hstack([np.linspace(0, 0.5, number_of_samples) for _ in range(len(unique_rd))])

    # Reshape the arrays to have a single column
    first_column = repeated_values.reshape(-1, 1)
    second_column = second_column_values.reshape(-1, 1)
    sdf = np.zeros(first_column.shape)

    # Concatenate the columns horizontally to create the final array
    result_array = np.hstack((first_column, second_column, sdf))    
    result_tensor = torch.from_numpy(result_array)
    result_tensor = result_tensor.to(torch.float32)
    
    return result_tensor

def original_sampling(N, overall_index, voxel_size, voxel_origin):

    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    return samples

    