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


class ShadowDataset(data.Dataset):
	def __init__(self, root):
		if not os.path.isdir(root):
			raise ValueError(f"The specified root: {root} does not exist")
		self.root = root
		imgs = [k for k in os.listdir(root) if k.endswith(('.png', '.jpg', '.jpeg'))]
		self.imgs = [os.path.join(root, k) for k in imgs]
		self.imglabel = [
			[float(value) for value in k[:-4].split("_") if value.replace(".", "", 1).isdigit()]
			for k in imgs
		]

		# Ensure consistent label lengths
		max_label_length = max(len(label) for label in self.imglabel)
		self.imglabel = [
			label + [0.0] * (max_label_length - len(label))  # Pad shorter labels with zeros
			for label in self.imglabel
		]

		self.transforms = transforms.Compose([
			transforms.Resize((128, 128)),
			transforms.ToTensor()
		])

	def __getitem__(self, idx):
		img_path = self.imgs[idx]
		label = torch.tensor(self.imglabel[idx], dtype=torch.float)
		pil_img = Image.open(img_path).convert('RGB')
		data = self.transforms(pil_img)
		return data, label

	def __len__(self):
		return len(self.imgs)
