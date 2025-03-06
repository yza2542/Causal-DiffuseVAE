# Helper script to generate reconstructions from a conditional DDPM model
# Add project directory to sys.path
import os
import sys

p = os.path.join(os.path.abspath("."), "main")
sys.path.insert(1, p)

import copy

import hydra
import pytorch_lightning as pl
import torch
from models.callbacks import ImageWriter
from models.diffusion import DDPM, DDPMv2, DDPMWrapper, SuperResModel
from SCMvae_shadow import CausalVAE
# from SCMvae_synthetic import CausalVAE
# from SCMvae import CausalVAE
# from SCMvae_Mnist import CausalVAE
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from util import configure_device, get_dataset
from datasets_shadow import get_shadow_data_RGBA
# from datasets_synthetic import get_synthetic_data
# from datasets_mnist import get_mnist_data
# from datasets_age import get_celeba_data
# from datasets_smile import get_celeba_data
from datasets import CIFAR10Dataset
import numpy as np


def load_dataset(dataset_dir):
    print('Loading datasets...')

    data_name = 'smile'
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = \
        get_shadow_data_RGBA(dataset_dir, 1)
    #train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = \
    #    get_synthetic_data(1)

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

def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]


@hydra.main(config_path="configs")
def generate_inference(config):
    config_ddpm = config.dataset.ddpm
    config_vae = config.dataset.vae
    seed_everything(config_ddpm.evaluation.seed, workers=True)

    batch_size = config_ddpm.evaluation.batch_size
    n_steps = config_ddpm.evaluation.n_steps
    n_samples = config_ddpm.evaluation.n_samples
    image_size = config_ddpm.data.image_size
    ddpm_latent_path = config_ddpm.data.ddpm_latent_path
    ddpm_latents = torch.load(ddpm_latent_path) if ddpm_latent_path != "" else None

    # Load pretrained VAE
    vae = CausalVAE.load_from_checkpoint(
        config_vae.evaluation.chkpt_path,
        input_res=image_size,
    )
    vae.eval()

    # Load pretrained wrapper
    attn_resolutions = __parse_str(config_ddpm.model.attn_resolutions)
    dim_mults = __parse_str(config_ddpm.model.dim_mults)
    decoder = SuperResModel(
        in_channels=config_ddpm.data.n_channels,
        model_channels=config_ddpm.model.dim,
        out_channels=3,
        num_res_blocks=config_ddpm.model.n_residual,
        attention_resolutions=attn_resolutions,
        channel_mult=dim_mults,
        use_checkpoint=False,
        dropout=config_ddpm.model.dropout,
        num_heads=config_ddpm.model.n_heads,
        z_dim=config_ddpm.evaluation.z_dim,
        use_scale_shift_norm=config_ddpm.evaluation.z_cond,
        use_z=config_ddpm.evaluation.z_cond,
    )

    ema_decoder = copy.deepcopy(decoder)
    decoder.eval()
    ema_decoder.eval()

    ddpm_cls = DDPMv2 if config_ddpm.evaluation.type == "form2" else DDPM
    online_ddpm = ddpm_cls(
        decoder,
        beta_1=config_ddpm.model.beta1,
        beta_2=config_ddpm.model.beta2,
        T=config_ddpm.model.n_timesteps,
        var_type=config_ddpm.evaluation.variance,
    )
    target_ddpm = ddpm_cls(
        ema_decoder,
        beta_1=config_ddpm.model.beta1,
        beta_2=config_ddpm.model.beta2,
        T=config_ddpm.model.n_timesteps,
        var_type=config_ddpm.evaluation.variance,
    )

    ddpm_wrapper = DDPMWrapper.load_from_checkpoint(
        config_ddpm.evaluation.chkpt_path,
        online_network=online_ddpm,
        target_network=target_ddpm,
        vae=vae,
        conditional=True,
        pred_steps=n_steps,
        eval_mode="recons",
        resample_strategy=config_ddpm.evaluation.resample_strategy,
        skip_strategy=config_ddpm.evaluation.skip_strategy,
        sample_method=config_ddpm.evaluation.sample_method,
        sample_from=config_ddpm.evaluation.sample_from,
        data_norm=config_ddpm.data.norm,
        temp=config_ddpm.evaluation.temp,
        guidance_weight=config_ddpm.evaluation.guidance_weight,
        z_cond=config_ddpm.evaluation.z_cond,
        mask=config_ddpm.evaluation.mask,
        value=config_ddpm.evaluation.value,
        ddpm_latents=ddpm_latents,
        strict=True,
    )

    # Dataset
    # root = config_ddpm.data.root
    # d_type = config_ddpm.data.name
    # image_size = config_ddpm.data.image_size
    # dataset = get_dataset(
    #     d_type,
    #     root,
    #    image_size,
    #    norm=config_ddpm.data.norm,
    #    flip=config_ddpm.data.hflip,
    #    subsample_size=n_samples,
    #)

    # dataset_dir = '/home/2527823Y/DiffuseVAE-main/main/datasets/smile_original/'
    # dataset_dir = '/home/2527823Y/DiffuseVAE-main/main/datasets/Bald'
    # dataset_dir = '/home/2527823Y/DiffuseVAE-main/main/datasets/flow_noise/'
    # dataset_dir = '/home/2527823Y/DiffuseVAE-main/main/datasets/pendulum/'
    # dataset_dir = '/home/2527823Y/DiffuseVAE-main/main/datasets/Shadow_RGBA'
    dataset_dir = '/home/2527823Y/DiffuseVAE-main/main/datasets/Shadow_Do'
    datasets, data_loaders, _ = load_dataset(dataset_dir)
    # datasets, data_loaders, _ = load_dataset()

    # Setup devices
    test_kwargs = {}
    loader_kws = {}
    device = config_ddpm.evaluation.device
    if device.startswith("gpu"):
        _, devs = configure_device(device)
        test_kwargs["gpus"] = devs

        # Disable find_unused_parameters when using DDP training for performance reasons
        loader_kws["persistent_workers"] = True
    elif device == "tpu":
        test_kwargs["tpu_cores"] = 8

    # Predict loader
    #val_loader = DataLoader(
    #    dataset,
    #    batch_size=batch_size,
    #    drop_last=False,
    #    pin_memory=True,
    #    shuffle=False,
    #    num_workers=config_ddpm.evaluation.workers,
    #    **loader_kws,
    #)
    loader = data_loaders['test']

    #rep_train = np.empty((4385, 64))
    #y_train = np.empty((4385, 4))
    #batch_idx = 0
    #while batch_idx < 274:
    #    batch, cond = next(loader)
    #    A = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.float32).to(batch.device)
    #    z, _, _, _ = vae.encode(batch)
    #    z = z.reshape(-1, 512)
    #    rep_train[batch_idx * z.shape[0]:(batch_idx * z.shape[0]) + z.shape[0], :] = z.cpu().detach().numpy()
    #    y_train[batch_idx * z.shape[0]:(batch_idx * z.shape[0]) + cond["c"].shape[0], :] = cond["c"].cpu().detach().numpy()


    # Predict trainer
    write_callback = ImageWriter(
        config_ddpm.evaluation.save_path,
        "batch",
        n_steps=n_steps,
        eval_mode="recons",
        conditional=True,
        sample_prefix=config_ddpm.evaluation.sample_prefix,
        save_mode=config_ddpm.evaluation.save_mode,
        save_vae=config_ddpm.evaluation.save_vae,
        is_norm=config_ddpm.data.norm,
    )

    test_kwargs["callbacks"] = [write_callback]
    test_kwargs["default_root_dir"] = config_ddpm.evaluation.save_path
    trainer = pl.Trainer(**test_kwargs)
    trainer.predict(ddpm_wrapper, loader)


if __name__ == "__main__":
    generate_inference()
