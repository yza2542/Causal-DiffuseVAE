import argparse
import torch.utils.data as data
import numpy as np
import torch
import sys
import random
sys.path.append('../')
from causalVAE import CausalVAE
from util_1 import get_celeba_data

def load_dataset(config):
    print('Loading datasets...')

    data_name = 'smile'
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = \
        get_celeba_data(config.training.dataset_dir, 64, type='train')

    print(f'Length of training set: {len(train_dataset)}')
    print(f'Length of val set: {len(val_dataset)}')
    print(f'Length of test set: {len(test_dataset)}')

    datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }

    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

    return datasets, dataloaders, data_name

