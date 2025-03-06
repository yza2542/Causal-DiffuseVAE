import logging
import os
import torch
import argparse
import hydra
import pytorch_lightning as pl
import torchvision.transforms as T
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from pathlib import Path
# from models.causalvae import causalVAE
from SCMvae_shadow import CausalVAE
from util import configure_device, get_dataset
# from datasets_shadow import get_shadow_data
from datasets_shadow import get_shadow_data_RGBA
import numpy as np

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def matrix_poly(matrix, d):
    x = torch.eye(d).to(device)+ torch.div(matrix.to(device), d).to(device)
    return torch.matrix_power(x, d)


def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A


@hydra.main(config_path="configs")
def load_dataset(config):
    print('Loading datasets...')

    data_name = 'shadow'
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = \
        get_shadow_data_RGBA(config.training.dataset_dir, config.training.batch_size)

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


logger = logging.getLogger(__name__)


@hydra.main(config_path="configs")
def train(config):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    # get the config
    config = config.dataset.vae
    logger.info(OmegaConf.to_yaml(config))

    # set seed
    seed_everything(config.training.seed, workers=True)

    # dataset
    datasets, data_loaders, data_name = load_dataset(config)

    # loader
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']

    # model
    cvae = CausalVAE
    c_vae = CausalVAE().to(device)

    # checkpoint
    restore_path = config.training.restore_path
    checkpoint_path = config.training.restore_path
    results_dir = config.training.results_dir
    chkpt_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename=f"vae_celeba_128-{config.training.chkpt_prefix}"
                 + "-{epoch:02d}-{train_loss:.4f}",
        every_n_epochs=5,
        save_top_k=-1,
        save_on_train_epoch_end=True,
    )


    train_kwargs = {}
    device = config.training.device
    if device.startswith("gpu"):
        _, devs = configure_device(device)
        train_kwargs["gpus"] = devs

        # Disable find_unused_parameters when using DDP training for performance reasons
        from pytorch_lightning.plugins import DDPPlugin

        train_kwargs["plugins"] = DDPPlugin(find_unused_parameters=False)
    elif device == "tpu":
        train_kwargs["tpu_cores"] = 8

    # save name
    save_name = "cdiffvae"

    trainer = pl.Trainer(default_root_dir=os.path.join(checkpoint_path, save_name),
                         gpus=1,
                         max_epochs=config.training.epochs,
                         callbacks=chkpt_callback)

    #trainer.fit(cvae, train_dataloader=train_loader, val_dataloaders=val_loader)
    Path(f"{trainer.logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)
    Path(f"{trainer.logger.log_dir}/Real").mkdir(exist_ok=True, parents=True)

    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None


    # CHECK IF LOADING
    pretrained_file = os.path.join(checkpoint_path, save_name + '.ckpt')
    if os.path.isfile(pretrained_file):
        print(f'Found pretrained model to load at {pretrained_file}, loading...')
        model = cvae.load_from_checkpoint(pretrained_file)
    else:
        pl.seed_everything(42)
        model = cvae(model_name=save_name)
        trainer.fit(model, train_loader, val_loader)
        current_epoch = trainer.current_epoch
        if trainer.current_epoch == 99:
            c_vae.save_latent_representations(dataloader=train_loader, epoch=current_epoch)
        model = cvae.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)  # load best model

    return model

if __name__ == "__main__":
    train()

