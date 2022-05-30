# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Shard Descriptor template.

It is recommended to perform tensor manipulations using numpy.
"""

import torch
import random
from torch.utils.data import Dataset
import os
import glob
import numpy as np
import pandas as pd

import constants
from utils import load_image, uniform_temporal_subsample

from sklearn.model_selection import train_test_split

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor


class LocalMRIDataset(ShardDataset):
    def __init__(self, paths, targets, transform=None, data_folder=''):
        self.paths = paths
        self.targets = targets
        self.transform = transform
        self.data_folder = data_folder

    def __len__(self):
        return len(self.paths)

    def read_video(self, vid_paths):
        video = [load_image(path) for path in vid_paths]
        if self.transform:
            seed = random.randint(0, 99999)
            for i in range(len(video)):
                random.seed(seed)
                video[i] = self.transform(image=video[i].astype(np.float32))["image"]

        video = [torch.tensor(frame, dtype=torch.float32) for frame in video]
        if len(video) == 0:
            video = torch.zeros(constants.N_FRAMES, constants.IMAGE_SIZE, constants.IMAGE_SIZE)
        else:
            video = torch.stack(video)  # T * C * H * W
        return video

    def __getitem__(self, index):
        _id = self.paths[index]
        patient_path = f"{self.data_folder}/{str(_id).zfill(5)}/"
        channels = []
        for t in ["FLAIR", "T1w", "T1wCE", "T2w"]:
            t_paths = sorted(
                glob.glob(os.path.join(patient_path, t, "*")),
                key=lambda x: int(x[:-4].split("-")[-1]),
            )
            num_samples = constants.N_FRAMES
            if len(t_paths) < num_samples:
                in_frames_path = t_paths
            else:
                in_frames_path = uniform_temporal_subsample(t_paths, num_samples)

            channel = self.read_video(in_frames_path)
            if channel.shape[0] == 0:
                print("1 channel empty")
                channel = torch.zeros(num_samples, constants.IMAGE_SIZE, constants.IMAGE_SIZE)
            channels.append(channel)

        channels = torch.stack(channels).transpose(0, 1)

        y = torch.tensor(self.targets[index], dtype=torch.float)
        return {"X": channels.float(), "y": y}


class LocalMRIShardDescriptor(ShardDescriptor):
    """Shard descriptor subclass."""

    def __init__(self, data_folder, labels_file) -> None:
        """
        Initialize local Shard Descriptor.

        Parameters are arbitrary, set up the ShardDescriptor-related part
        of the envoy_config.yaml as you need.
        """
        super().__init__()

        self.data_folder = data_folder
        self.labels_file = labels_file
        self.transform = None
        self.labels = None
        self.splitted_labels = {}

    def split_dataset(self, test_size=0.2):
        self.__load_labels()
        _0, _1, y_train, y_test = train_test_split(np.zeros(len(self.labels)), self.labels, test_size=test_size, random_state=42)
        self.splitted_labels = {
            'train': y_train,
            'val': y_test
        }

    def set_transform_params(self, transform):
        self.transform = transform


    def get_dataset(self, dataset_type='train'):
        if self.splitted_labels[dataset_type] is not None:
            return LocalMRIDataset(
                paths=self.splitted_labels[dataset_type]['BraTS21ID'].values,
                targets=self.splitted_labels[dataset_type]['MGMT_value'].values,
                transform=self.transform
            )
        return None
    
    def __load_labels(self):
        self.labels = pd.read_csv(f'{self.data_folder}/{self.labels_file}')

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['4', '256', '256', '3']

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['1']

    @property
    def dataset_description(self) -> str:
        """Return the shard dataset description."""
        return (f'Local MRI Shard Descriptor is working.')
