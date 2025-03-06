import os

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.latent import UncondLatentDataset
from SCMvae import CausalVAE
from util import configure_device, get_dataset, save_as_images, get_celeba_data

def load_dataset(dataset_dir):
    print('Loading datasets...')

    data_name = 'smile'
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = \
        get_celeba_data(dataset_dir, 64, type='train')

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


@click.group()
def cli():
    pass


def compare_samples(gen, refined, save_path=None, figsize=(6, 3)):
    # Plot all the quantities
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ax[0].imshow(gen.permute(1, 2, 0))
    ax[0].set_title("VAE Reconstruction")
    ax[0].axis("off")

    ax[1].imshow(refined.permute(1, 2, 0))
    ax[1].set_title("Refined Image")
    ax[1].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)


def plot_interpolations(interpolations, save_path=None, figsize=(10, 5)):
    N = len(interpolations)
    # Plot all the quantities
    fig, ax = plt.subplots(nrows=1, ncols=N, figsize=figsize)

    for i, inter in enumerate(interpolations):
        ax[i].imshow(inter.permute(1, 2, 0))
        ax[i].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)


def compare_interpolations(
    interpolations_1, interpolations_2, save_path=None, figsize=(10, 2)
):
    assert len(interpolations_1) == len(interpolations_2)
    N = len(interpolations_1)
    # Plot all the quantities
    fig, ax = plt.subplots(nrows=2, ncols=N, figsize=figsize)

    for i, (inter_1, inter_2) in enumerate(zip(interpolations_1, interpolations_2)):
        ax[0, i].imshow(inter_1.permute(1, 2, 0))
        ax[0, i].axis("off")

        ax[1, i].imshow(inter_2.permute(1, 2, 0))
        ax[1, i].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)


# TODO: Upgrade the commands in this script to use hydra config
# and support Multi-GPU inference
@cli.command()
@click.argument("chkpt-path")
@click.argument("root")
@click.option("--device", default="gpu:1")
@click.option("--dataset", default="celebamaskhq")
@click.option("--image-size", default=128)
@click.option("--num-samples", default=-1)
@click.option("--save-path", default=os.getcwd())
@click.option("--write-mode", default="image", type=click.Choice(["numpy", "image"]))
def reconstruct(
    chkpt_path,
    root,
    device="gpu:1",
    dataset="smile",
    image_size=128,
    num_samples=-1,
    save_path=os.getcwd(),
    write_mode="image",
):
    dev = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    if num_samples == 0:
        raise ValueError(f"`--num-samples` can take value=-1 or > 0")


    dataset_dir = '/home/2527823Y/DiffuseVAE-main/main/datasets/Smile/'
    # Dataset
    datasets, data_loaders, _ = load_dataset(dataset_dir)
    # Dataset
    # dataset = get_dataset(dataset, root, image_size, norm=False, flip=False)

    # Loader
    # loader = DataLoader(
    #     dataset,
    #    16,
    #     num_workers=4,
    #    pin_memory=True,
    #    shuffle=False,
    #    drop_last=False,
    # )
    loader = data_loaders['test']
    cvae = CausalVAE.load_from_checkpoint(chkpt_path, input_res=image_size).to(dev)
    cvae.eval()

    sample_list = []
    img_list = []
    count = 0
    for _, batch in tqdm(enumerate(loader)):
        batch_img = batch[0].to(dev)
        # print(batch_img.size())
        batch_label = batch[1]
        with torch.no_grad():
            recons = cvae.forward_recons(batch_img, batch_label)
            # print(recons.size())

        if count + recons.size(0) >= num_samples and num_samples != -1:
            img_list.append(batch[:num_samples, :, :, :].cpu())
            sample_list.append(recons[:num_samples, :, :, :].cpu())
            break

        # Not transferring to CPU leads to memory overflow in GPU!
        sample_list.append(recons.cpu())
        img_list.append(batch[0].cpu())
        count += recons.size(0)

    cat_img = torch.cat(img_list, dim=0)
    cat_img = cat_img.reshape(-1, 3, 128, 128)
    cat_sample = torch.cat(sample_list, dim=0)
    cat_sample = cat_sample.reshape(-1, 3, 128, 128)

    # Save the image and reconstructions as numpy arrays
    os.makedirs(save_path, exist_ok=True)

    if write_mode == "image":
        save_as_images(
            cat_sample,
            file_name=os.path.join(save_path, "vae"),
            denorm=False,
        )
        save_as_images(
            cat_img,
            file_name=os.path.join(save_path, "orig"),
            denorm=False,
        )
    else:
        np.save(os.path.join(save_path, "images.npy"), cat_img.numpy())
        np.save(os.path.join(save_path, "recons.npy"), cat_sample.numpy())


@cli.command()
@click.argument("z-dim", type=int)
@click.argument("chkpt-path")
@click.option("--seed", default=0, type=int)
@click.option("--device", default="gpu:1")
@click.option("--image-size", default=128)
@click.option("--num-samples", default=-1)
@click.option("--save-path", default=os.getcwd())
@click.option("--write-mode", default="image", type=click.Choice(["numpy", "image"]))
def sample(
    z_dim,
    chkpt_path,
    seed=0,
    device="gpu:1",
    image_size=128,
    num_samples=1,
    save_path=os.getcwd(),
    write_mode="image",
):
    seed_everything(seed)
    dev, _ = configure_device(device)
    if num_samples <= 0:
        raise ValueError(f"`--num-samples` can take values > 0")

    dataset = UncondLatentDataset((num_samples, z_dim, 1, 1))

    # Loader
    loader = DataLoader(
        dataset,
        16,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    cvae = CausalVAE.load_from_checkpoint(chkpt_path, input_res=image_size).to(dev)
    cvae.eval()

    sample_list = []
    count = 0
    for _, batch in tqdm(enumerate(loader)):
        batch = batch.to(dev)
        with torch.no_grad():
            recons = cvae.forward(batch)

        if count + recons.size(0) >= num_samples and num_samples != -1:
            sample_list.append(recons[:num_samples, :, :, :].cpu())
            break

        # Not transferring to CPU leads to memory overflow in GPU!
        sample_list.append(recons.cpu())
        count += recons.size(0)

    cat_sample = torch.cat(sample_list, dim=0)

    # Save the image and reconstructions as numpy arrays
    os.makedirs(save_path, exist_ok=True)

    if write_mode == "image":
        save_as_images(
            cat_sample,
            file_name=os.path.join(save_path, "vae"),
            denorm=False,
        )
    else:
        np.save(os.path.join(save_path, "recons.npy"), cat_sample.numpy())


if __name__ == "__main__":
    cli()
