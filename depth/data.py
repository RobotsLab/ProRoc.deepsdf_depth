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

    for key, value in input_file.items():
        for row in value:
            if np.any(np.isnan(row)):
                break
            else:
                dd = row[0]
                rd = row[1]
                sdf = row[2]
                object_image.append(np.array([dd, rd, sdf]))

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

        for key, value in input_file.items():
            for row in value:
                if np.any(np.isnan(row)):
                    break
                else:
                    dd = row[0]
                    rd = row[1]
                    sdf = row[2]
                    object_image.append(np.array([dd, rd, sdf]))
        data_array = np.vstack(object_image)
        samples = torch.from_numpy(data_array)
        # if data_array[data_array < 0].any():
        #     print(filename, 'TUTAJ')
        random_indices = (torch.rand(subsample) * data_array.shape[0]).long()
        samples = torch.index_select(samples, 0, random_indices)
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
