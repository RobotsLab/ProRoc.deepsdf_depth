#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import json

import deep_sdf.workspace as ws
# import workspace as ws

def get_instance_filenames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".json"
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                npzfiles += [instance_filename]
    return npzfiles


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    # print(mesh_filenames[0])
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename):
    # with open(filename, "r") as file:
    #     lines = file.readlines()

    # # Remove "nan" values from the lines
    # cleaned_lines = [line.strip() for line in lines if "nan" not in line]

    # # Convert the cleaned lines to a NumPy array
    # data_array = np.loadtxt(cleaned_lines)
    print(filename)
    with open(filename, 'r') as f:
        input_file = json.load(f)

    object_image = []

    min_u = 534
    max_u = 711
    min_v = 232
    max_v = 419
    min_dd = 0.10000000000000009
    max_dd = 0.30145583152771005
    min_rd = -0.30101666450500497
    max_rd = 0.3999999999999999
    min_sdf = 0
    max_sdf = 0.2930116653442383

    # Your updated code with scaling to [0, 1] and filtering negative rd values
    for key, value in input_file.items():
        for row in value:
            if np.any(np.isnan(row)):
                continue  # Skip rows with NaN values
            else:
                u = int(key.split(', ')[0])
                v = int(key.split(', ')[1])
                
                dd = row[0]
                rd = row[1]
                sdf = row[2]
                
                # # Skip rows where rd is negative
                # if rd < 0:
                #     continue

                # Normalize u, v, dd, rd, and sdf to [0, 1]
                u_normalized = (u - min_u) / (max_u - min_u)
                v_normalized = (v - min_v) / (max_v - min_v)
                dd_normalized = (dd - min_dd) / (max_dd - min_dd)
                rd_normalized = (rd - min_rd) / (max_rd - min_rd)
                sdf_normalized = (sdf - min_sdf) / (max_sdf - min_sdf)

                # Append the normalized values to the object_image
                object_image.append(np.array([u_normalized, v_normalized, dd_normalized, rd_normalized, sdf_normalized]))


    data_array = np.vstack(object_image)
    samples = torch.from_numpy(data_array)

    return samples


def unpack_sdf_samples(filename, subsample=None):
    try:
        # with open(filename, "r") as file:
        #     lines = file.readlines()

        # # Remove "nan" values from the lines
        # cleaned_lines = [line.strip() for line in lines if "nan" not in line]

        # # Convert the cleaned lines to a NumPy array
        # data_array = np.loadtxt(cleaned_lines)
        # print(data_array)
        with open(filename, 'r') as f:
            input_file = json.load(f)
        object_image = []

        min_u = 534
        max_u = 711
        min_v = 232
        max_v = 419
        min_dd = 0.10000000000000009
        max_dd = 0.30145583152771005
        min_rd = -0.30101666450500497
        max_rd = 0.3999999999999999
        min_sdf = 0
        max_sdf = 0.2930116653442383

        # Your updated code with scaling to [0, 1] and filtering negative rd values
        for key, value in input_file.items():
            for row in value:
                if np.any(np.isnan(row)):
                    continue  # Skip rows with NaN values
                else:
                    u = int(key.split(', ')[0])
                    v = int(key.split(', ')[1])
                    
                    dd = row[0]
                    rd = row[1]
                    sdf = row[2]
                    
                    # # Skip rows where rd is negative
                    # if rd < 0:
                    #     continue

                    # Normalize u, v, dd, rd, and sdf to [0, 1]
                    u_normalized = (u - min_u) / (max_u - min_u)
                    v_normalized = (v - min_v) / (max_v - min_v)
                    dd_normalized = (dd - min_dd) / (max_dd - min_dd)
                    rd_normalized = (rd - min_rd) / (max_rd - min_rd)
                    sdf_normalized = (sdf - min_sdf) / (max_sdf - min_sdf)

                    # Append the normalized values to the object_image
                    object_image.append(np.array([u_normalized, v_normalized, dd_normalized, rd_normalized, sdf_normalized]))


        data_array = np.vstack(object_image)
        samples = torch.from_numpy(data_array)
        # if data_array[data_array < 0].any():
        #     print(filename, 'TUTAJ')
        if subsample:
            random_indices = (torch.rand(subsample) * data_array.shape[0]).long()
            samples = torch.index_select(samples, 0, random_indices)
        # print("First 10 rows before shuffling:")
        # print(samples[:10])
        # permutation for exp8
        # permutation = torch.randperm(samples.size(0))
        # samples = samples[permutation]
        # print("\nFirst 10 rows after shuffling:")
        # print(samples[:10])
        return samples

    except:
        print(f'{filename} not found')


def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data

    # split the sample into half
    half = int(subsample / 2)
    data_size = data.shape[0]
    data_start_ind = random.randint(0, data_size - half)
    samples = data[data_start_ind : (data_start_ind + half)]

    return samples


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        load_ram=False,
        print_filename=False,
        num_files=1000000,
    ):
        self.subsample = subsample

        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram

        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                    ]
                )

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx]
        )
        if self.load_ram:
            return (
                unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample),
                idx,
            )
        else:
            return unpack_sdf_samples(filename, self.subsample), idx
        

class RayDataset(torch.utils.data.Dataset):
    def __init__(self, data_source, split):
        self.data = []
        self.load_data(data_source, split)

    def load_data(self, data_source, split):
        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)
        for file in self.npyfiles:
            with open(os.path.join(self.data_source, ws.sdf_samples_subdir, file), 'r') as f:
                data = json.load(f)
                for key, points in data.items():
                    for point in points:
                        rd, dd, sdf = point
                        self.data.append((rd, dd, sdf))
        data_array = np.vstack(self.data)
        self.samples = torch.from_numpy(data_array)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.samples[idx], idx

