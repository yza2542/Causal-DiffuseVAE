# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software;
# you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import os
import math
import random
import torch.nn as nn
from torch.utils import data
import argparse
import numpy as np
from torchvision import transforms
from PIL import Image
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets.age import AgeDataset
import torchvision.transforms as T

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def get_celeba_data(dataset_dir, batch_size, type="train", shuf=False):
    n = 20000
    dataset = AgeDataset(dataset_dir)
    print(f"Length of dataset: {len(dataset)}")

    train_dataset = torch.utils.data.Subset(dataset, list(range(0, int(len(dataset) * 0.7))))
    val_dataset = torch.utils.data.Subset(dataset, list(range(int(len(dataset) * 0.7), int(len(dataset) * 0.85))))
    test_dataset = torch.utils.data.Subset(dataset, list(range(int(len(dataset) * 0.85), len(dataset))))

    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
