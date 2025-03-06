import os
from typing import Sequence, Union

import numpy as np
import torchvision.utils as vutils
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import Tensor
from torch.nn import Module
from util import save_as_images, save_as_np

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def batch_rgba_to_rgb(rgba_batch, background=(1, 1, 1)):
    """
    Convert a batch of RGBA tensors to a batch of RGB tensors by blending the alpha channel with a background color.

    Parameters:
    rgba_batch (torch.Tensor): Tensor of shape (B, 4, H, W) where B is the batch size.
    background (tuple): RGB values of the background color (default is white).

    Returns:
    torch.Tensor: Tensor of shape (B, 3, H, W) representing a batch of RGB images.
    """
    assert rgba_batch.shape[1] == 4, "Input tensor must have 4 channels (RGBA)"

    # Extract the RGB and Alpha channels
    rgb = rgba_batch[:, :3, :, :]  # Shape: (B, 3, H, W)
    alpha = rgba_batch[:, 3, :, :].unsqueeze(1)  # Shape: (B, 1, H, W)

    # Create a background tensor
    bg_tensor = torch.tensor(background, device=rgba_batch.device).view(1, 3, 1, 1)

    # Blend the RGB channels with the background using the alpha channel
    rgb_image = rgb * alpha + bg_tensor * (1 - alpha)

    return rgb_image


class EMAWeightUpdate(Callback):
    """EMA weight update
    Your model should have:
        - ``self.online_network``
        - ``self.target_network``
    Updates the target_network params using an exponential moving average update rule weighted by tau.
    BYOL claims this keeps the online_network from collapsing.
    .. note:: Automatically increases tau from ``initial_tau`` to 1.0 with every training step
    Example::
        # model must have 2 attributes
        model = Model()
        model.online_network = ...
        model.target_network = ...
        trainer = Trainer(callbacks=[EMAWeightUpdate()])
    """

    def __init__(self, tau: float = 0.9999):
        """
        Args:
            tau: EMA decay rate
        """
        super().__init__()
        self.tau = tau

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # get networks
        online_net = pl_module.online_network.decoder
        target_net = pl_module.target_network.decoder

        # update weights
        self.update_weights(online_net, target_net)

    def update_weights(
        self, online_net: Union[Module, Tensor], target_net: Union[Module, Tensor]
    ) -> None:
        # apply MA weight update
        with torch.no_grad():
            for targ, src in zip(target_net.parameters(), online_net.parameters()):
                targ.mul_(self.tau).add_(src, alpha=1 - self.tau)


class ImageWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir,
        write_interval,
        compare=False,
        n_steps=None,
        eval_mode="sample",
        conditional=True,
        sample_prefix="",
        save_vae=False,
        save_mode="image",
        is_norm=True,
    ):
        super().__init__(write_interval)
        assert eval_mode in ["sample", "recons"]
        self.output_dir = output_dir
        self.compare = compare
        self.n_steps = 1000 if n_steps is None else n_steps
        self.eval_mode = eval_mode
        self.conditional = conditional
        self.sample_prefix = sample_prefix
        self.save_vae = save_vae
        self.is_norm = is_norm
        self.save_fn = save_as_images if save_mode == "image" else save_as_np

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        rank = pl_module.global_rank
        if self.conditional:
            ddpm_samples_dict, vae_samples = prediction
            # x, classifier_label, causal_label = batch
            x, label = batch
            print(label)
            # x = x.reshape(-1, 1, 28, 28)

            if self.save_vae:
                vae_samples = vae_samples.cpu()
                # print('vae', vae_samples[:, 0, :, :])
                vae_save_path = os.path.join(self.output_dir, "vae")
                real_save_path = os.path.join(self.output_dir, "real")
                os.makedirs(vae_save_path, exist_ok=True)
                os.makedirs(real_save_path, exist_ok=True)
                # self.save_fn(
                #    vae_samples,
                #    file_name=os.path.join(
                #        vae_save_path,
                #        f"output_vae_{self.sample_prefix}_{rank}_{batch_idx}",
                #    ),
                #    denorm=self.is_norm,
                #)
                #print("vae", vae_samples[:, 3, :, :])
                x = batch_rgba_to_rgb(x)
                vutils.save_image(vae_samples,
                                  os.path.join(vae_save_path,
                                               f"output_vae_{self.sample_prefix}_{rank}_{batch_idx}.png"),
                                  normalize=True)


                vutils.save_image(x,
                                  os.path.join(real_save_path,
                                               f"output_real_{self.sample_prefix}_{rank}_{batch_idx}.png"))
        else:
            ddpm_samples_dict = prediction

        # Write output images
        # NOTE: We need to use gpu rank during saving to prevent
        # processes from overwriting images
        for k, ddpm_samples in ddpm_samples_dict.items():
            ddpm_samples = ddpm_samples.cpu()
            # print(ddpm_samples.size())
            # Setup dirs
            base_save_path = os.path.join(self.output_dir, k)
            img_save_path = os.path.join(base_save_path, "images")
            os.makedirs(img_save_path, exist_ok=True)
            # print("ddpm", ddpm_samples[:, 0, :, :])
            # Save
            #self.save_fn(
            #    ddpm_samples,
            #    file_name=os.path.join(
            #        img_save_path, f"output_{self.sample_prefix }_{rank}_{batch_idx}"
            #    ),
            #    denorm=self.is_norm,
            #)
            vutils.save_image(ddpm_samples,
                              os.path.join(img_save_path, f"output_{self.sample_prefix }_{rank}_{batch_idx}.png"))





        # FIXME: This is currently broken. Separate this from the core logic
        # into a new function. Uncomment when ready!
        # if self.compare:
        #     # Save comparisons
        #     (_, img_samples), _ = batch
        #     img_samples = normalize(img_samples).cpu()
        #     iter_ = vae_samples if self.eval_mode == "sample" else img_samples
        #     for idx, (ddpm_pred, pred) in enumerate(zip(ddpm_samples, iter_)):
        #         samples = {
        #             "VAE" if self.eval_mode == "sample" else "Original": pred,
        #             "DDPM": ddpm_pred,
        #         }
        #         compare_samples(
        #             samples,
        #             save_path=os.path.join(
        #                 self.comp_save_path, f"compare_form1_{rank}_{idx}.png"
        #             ),
        #         )
