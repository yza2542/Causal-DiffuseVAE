import logging
import os
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
from SCMvae_synthetic import CausalVAE
# from SCMvae_Mnist import CausalVAE
from util import configure_device, get_dataset
from datasets_synthetic import get_synthetic_data
# from datasets_mnist import get_mnist_data


@hydra.main(config_path="configs")
def load_dataset(config):
    print('Loading datasets...')

    data_name = 'smile'
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = \
        get_synthetic_data(config.evaluation.dataset_dir, config.evaluation.batch_size, dataset='test')

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
    # get the config
    config = config.dataset.vae
    # logger.info(OmegaConf.to_yaml(config))
    image_size = config.data.image_size
    # set seed
    seed_everything(config.evaluation.seed, workers=True)

    # dataset
    datasets, data_loaders, data_name = load_dataset(config)

    # loader
    test_loader = data_loaders['test']


    # model
    cvae = CausalVAE

    # checkpoint
    restore_path = config.evaluation.restore_path
    checkpoint_path = config.evaluation.restore_path
    results_dir = config.evaluation.results_dir
    chkpt_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename=f"vae-{config.evaluation.chkpt_prefix}"
                 + "-{epoch:02d}-{train_loss:.4f}",
        every_n_epochs=config.evaluation.chkpt_interval,
        save_on_train_epoch_end=True,
    )


    # save name
    save_name = "cdiffvae"

    trainer = pl.Trainer(default_root_dir=os.path.join(checkpoint_path, save_name),
                         gpus=1,
                         max_epochs=config.evaluation.epochs,
                         callbacks=chkpt_callback)

    Path(f"{trainer.logger.log_dir}/Reconstructions_Inference").mkdir(exist_ok=True, parents=True)
    Path(f"{trainer.logger.log_dir}/Interventions_Inference").mkdir(exist_ok=True, parents=True)
    Path(f"{trainer.logger.log_dir}/Real_Inference").mkdir(exist_ok=True, parents=True)

    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None


    # CHECK IF LOADING
    pretrained_file = config.evaluation.chkpt_path
    if os.path.isfile(pretrained_file):
        print(f'Found pretrained model to load at {pretrained_file}, loading...')
        model = CausalVAE.load_from_checkpoint(pretrained_file)
    else:
        pl.seed_everything(42)
        model = CausalVAE.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)  # load best model

    trainer.test(model, test_loader, verbose=False)

if __name__ == "__main__":
    train()