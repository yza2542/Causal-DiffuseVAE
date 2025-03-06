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
from datasets.Synthetic import Synthetic, Synthetic_DDPM

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def get_synthetic_data(dataset_dir, batch_size, dataset="train"):
	dataset = Synthetic(dataset_dir, "train")

	train_dataset = torch.utils.data.Subset(dataset, list(range(0, int(len(dataset) * 0.8))))
	val_dataset = torch.utils.data.Subset(dataset, list(range(int(len(dataset) * 0.8), len(dataset))))
	test_dataset = Synthetic(dataset_dir, "test")

	train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


