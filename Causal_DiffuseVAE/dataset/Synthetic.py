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

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


class Synthetic(data.Dataset):
	def __init__(self, root, dataset="train"):
		root = root + "/" + dataset

		imgs = os.listdir(root)

		self.dataset = dataset

		self.imgs = [os.path.join(root, k) for k in imgs]
		self.imglabel = [list(map(int, k[:-4].split("_")[1:])) for k in imgs]
		# print(self.imglabel)
		self.transforms = transforms.Compose([transforms.ToTensor()])

	def __getitem__(self, idx):
		# print(idx)
		img_path = self.imgs[idx]

		label = torch.from_numpy(np.asarray(self.imglabel[idx]))
		# print(len(label))
		pil_img = Image.open(img_path)
		array = np.asarray(pil_img)
		array1 = np.asarray(label)
		label = torch.from_numpy(array1)
		# data = torch.from_numpy(array)
		if self.transforms:
			data = self.transforms(pil_img)
		else:
			pil_img = np.asarray(pil_img).reshape(96, 96, 4)
			data = torch.from_numpy(pil_img)

		return data, label.float()

	def __len__(self):
		return len(self.imgs)

class Synthetic_DDPM(data.Dataset):
	def __init__(self, root, dataset="train"):
		root = root + "/" + dataset

		imgs = os.listdir(root)

		self.dataset = dataset

		self.imgs = [os.path.join(root, k) for k in imgs]
		self.imglabel = [list(map(int, k[:-4].split("_")[1:])) for k in imgs]
		# print(self.imglabel)
		self.transforms = transforms.Compose([transforms.ToTensor()])

	def __getitem__(self, idx):
		# print(idx)
		img_path = self.imgs[idx]

		label = torch.from_numpy(np.asarray(self.imglabel[idx]))
		# print(len(label))
		pil_img = Image.open(img_path)
		pil_img = Image.alpha_composite(Image.new('RGBA', pil_img.size, (255, 255, 255, 255)), pil_img).convert('RGB')
		array = np.asarray(pil_img)
		array1 = np.asarray(label)
		label = torch.from_numpy(array1)
		# data = torch.from_numpy(array)
		if self.transforms:
			data = self.transforms(pil_img)

		return data, label.float()

	def __len__(self):
		return len(self.imgs)