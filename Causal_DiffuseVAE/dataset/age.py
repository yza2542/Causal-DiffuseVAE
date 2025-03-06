import os

import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

class AgeDataset(Dataset):
    def __init__(self, root):
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")
        self.root = root

        # age, gender, bald, beard
        attr_data = np.loadtxt('/home/2527823Y/DiffuseVAE-main/main/datasets/list_attr_celeba_filtered_sub_bald.txt', usecols=(0, 40, 21, 5, 25), dtype=str, comments='#')
        self.filenames = attr_data[:, 0]
        self.labels = attr_data[:, 1:].astype(np.float32)

        #    self.images.append(os.path.join(self.root, img))
        self.images = [os.path.join(self.root, fname) for fname in self.filenames if
                       os.path.exists(os.path.join(self.root, fname))]
        valid_indices = [i for i, fname in enumerate(self.filenames) if os.path.exists(os.path.join(self.root, fname))]
        self.labels = self.labels[valid_indices]
        self.transforms = transforms.Compose([
            transforms.CenterCrop(128),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        img_path = self.images[idx]
        pil_img = Image.open(img_path)

        label = torch.from_numpy(self.labels[idx])
        #label = torch.from_numpy(np.asarray(self.imglabel[idx]))
        #print(len(label))
        #array = np.asarray(pil_img)
        #array1 = np.asarray(label)
        #label = torch.from_numpy(array1)
        # data = torch.from_numpy(array)
        if self.transforms:
            data = self.transforms(pil_img)


        #if self.transforms:
        #    data = self.transforms(pil_img)
        #else:
        # pil_img = np.asarray(pil_img)
        # print('pil_img', pil_img.size)
        #pil_img = pil_img.reshape(96, 96, 4)
        # data = self.transforms(pil_img)
        #data = torch.from_numpy(data)
        # print(label)
        return data, label.float()

        #if self.norm:
            #img = (np.asarray(img).astype(np.float) / 127.5) - 1.0
        #else:
            #img = np.asarray(img).astype(np.float) / 255.0
        #print('data', data.shape)

    def __len__(self):
        return len(self.images)