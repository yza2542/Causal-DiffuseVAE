import pytorch_lightning as pl
import math
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import autograd, nn, optim
from torch.nn import Linear
import mask
import util as ut
import torchvision.utils as vutils
from encoder_decoder import Encoder, Decoder_DAG
from pytorch_lightning.callbacks import LearningRateMonitor
from modules import CosineWarmupScheduler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from minepy import MINE

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def compute_mic_tic(X, Y):
    """Computes MIC and TIC between each latent variable in X and each causal factor in Y"""
    mine = MINE(alpha=0.6, c=15)

    # 🔥 Ensure X and Y have the same number of samples (rows)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Shape mismatch: X has {X.shape[0]} samples, Y has {Y.shape[0]} samples.")

    mic_scores = np.zeros((X.shape[1], Y.shape[1]))  # Store MIC for each (latent, causal factor)
    tic_scores = np.zeros((X.shape[1], Y.shape[1]))  # Store TIC for each (latent, causal factor)

    for j in range(Y.shape[1]):  # Iterate over each causal factor
        Y_col = Y[:, j]  # Extract 1D column from Y (Do NOT flatten here)

        for i in range(X.shape[1]):  # Iterate over each latent variable
            X_col = X[:, i]  # Extract 1D column from X

            # 🔥 Ensure X_col and Y_col have the same length
            if X_col.shape[0] != Y_col.shape[0]:
                raise ValueError(f"Shape mismatch: X_col has {X_col.shape[0]} samples, Y_col has {Y_col.shape[0]} samples.")

            mine.compute_score(X_col, Y_col)
            mic_scores[i, j] = mine.mic()
            tic_scores[i, j] = mine.tic()

    return mic_scores, tic_scores

def calculate_mae(predicted_values, true_values):
    # Ensure both arrays are numpy arrays
    predicted_values = torch.tensor(predicted_values).to(device)
    true_values = torch.tensor(true_values).to(device)

    # Calculate the absolute errors
    absolute_errors = torch.abs(predicted_values - true_values).to(device)

    # Calculate the mean of the absolute errors
    mae = torch.mean(absolute_errors).to(device)

    return mae

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

class CausalVAE(pl.LightningModule):
    def __init__(self, model_name='cvae', z_dim=512, z1_dim=4, z2_dim=128, inference=False, alpha=0.1, beta=1, lambda_v=1e-3, lr=1e-4, use_causal_prior=True, warmup=100, max_iters=100, **kwargs):
        super().__init__()
        self.name = model_name
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.channel = 4
        # self.scale = np.array([[0,44],[100,40],[6.5, 3.5],[10,5]])
        self.scale = np.array([[20, 15], [2, 2], [59.5, 26.5], [10.5, 4.5]])
        self.encoder = Encoder(self.z_dim)
        self.decoder = Decoder_DAG(self.z_dim, self.z1_dim, self.z2_dim)
        self.inference = inference
        self.warmup = warmup
        self.max_iters = max_iters
        self.dag = mask.DagLayer(self.z1_dim, self.z1_dim, i=self.inference)
        # self.cause = nn.CausalLayer(self.z_dim, self.z1_dim, self.z2_dim)
        self.mask_z = mask.MaskLayer(self.z_dim)
        self.alpha = alpha
        self.beta = beta
        self.lambda_v = lambda_v
        self.lr = lr

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def _get_loss(self, x, label, mode="train"):
        """Calculate loss"""

        x_hat, z_sample, z_masked, z_m, z_v, label = self.forward(x, label)

        # RECONSTRUCTION LOSS
        rec = ut.log_bernoulli_with_logits(x, x_hat.reshape(x.size()))
        rec = -torch.mean(rec)

        # print(rec)

        # PRIORS
        p_m, p_v = torch.zeros(z_m.size()), torch.ones(z_m.size())
        cp_m, cp_v = ut.structural_condition_prior(self.scale, label, self.z2_dim, self.dag.A)
        # cp_m, cp_v = ut.condition_prior(self.scale, label, self.z2_dim)


        # print('cpm', cp_m)
        # print('cpv', cp_v)
        cp_v = torch.ones([z_m.size()[0], self.z1_dim, self.z2_dim]).to(device)
        #print('cpm', cp_m.shape)
        #print('cpv', cp_v.shape)
        cp_z = ut.conditional_sample_gaussian(cp_m.to(device), cp_v.to(device))

        # KL-DIVERGENCE BETWEEN DISTRIBUTION FROM ENCODER AND THE ISOTROPIC GAUSSIAN PRIOR
        kl = torch.zeros(1).to(device)

        # RESHAPE
        z_m = z_m.view(-1, self.z_dim).to(device)
        z_v = z_v.view(-1, self.z_dim).to(device)
        p_m = p_m.view(-1, self.z_dim).to(device)
        p_v = p_v.view(-1, self.z_dim).to(device)

        kl = self.alpha * ut.kl_normal(z_m, z_v, p_m, p_v)

        for i in range(self.z1_dim):
            kl = kl + self.beta * ut.kl_normal(z_masked[:, i, :].to(device), cp_v[:, i, :].to(device), cp_m[:, i, :].to(device), cp_v[:, i, :].to(device))

        kl = torch.mean(kl)
        neg_elbo = rec + kl

        # Logging
        self.log(f'{mode}_kld', kl)
        self.log(f'{mode}_rec_loss_t1', rec)
        self.log(f'{mode}_neg_elbo', neg_elbo)

        return neg_elbo, kl, rec, x_hat.reshape(x.size()), z_sample, cp_m

    def encode(self, x, label=None, mask=None, value=None):
        # q_m, q_v
        z_m, z_v = self.encoder.encode(x)
        # print('zm0', z_m.shape)
        #print('zv0', z_v.shape)
        z_m = z_m.reshape([z_m.size()[0], self.z1_dim, self.z2_dim])  # RESHAPE TO (BATCH, 4, 4)
        z_v = z_v.reshape([z_m.size()[0], self.z1_dim, self.z2_dim])  # RESHAPE TO (BATCH, 4, 4)
        #print('zmt', z_m.shape)
        #print('zvt', z_v.shape)
        z_temp = torch.clone(z_m)
        # print(z_temp.size())
        # NO DAG LEARNING SO GO STRAIGHT TO MASKING
        for i in range(4):
            if i == 0 or i == 1:
                z_temp[:, i, :] = z_m[:, i, :]
            else:
                # N, 4, 128
                z_temp[:, i, :] = self.mask_z.g(
                    self.dag.mask_z(z_temp, i).reshape([z_m.size()[0], self.z_dim])).reshape(
                    [z_m.size()[0], self.z2_dim]).to(device)

            if mask == i:
                z_temp[:, mask, :], z_v[:, mask, :] = self.perform_intervention(z_temp, z_v, mask, label, value)

        z_masked = z_temp
        # print('z_temp', z_temp.size())
        # FINAL "CAUSAL" REPRESENTATION
        z_sample = ut.conditional_sample_gaussian(z_masked, z_v * self.lambda_v)

        return z_sample, z_masked, z_m, z_v

    def perform_intervention(self, z_temp, z_v, mask, label, value):
        # print('pre', z_temp)
        label[:, mask] = value
        cp_m, cp_v = ut.condition_prior(self.scale, label, self.z2_dim)
        z_temp[:, mask, :] = cp_m[:, mask, :].to(device)
        # print('fin', z_temp)
        z_v[:, mask, :] = torch.abs(cp_v[:, mask, :])
        # z_v[:, mask, :] = cp_v[:, mask, :]

        return z_temp[:, mask, :], z_v[:, mask, :]


    def compute_sigmoid_given(self, z):
        logits = self.decoder.decode(z)
        return torch.sigmoid(logits)

    def forward(self, x, label, mask=None, value=None):
        z_sample, z_masked, z_m, z_v = self.encode(x, label=label, mask=mask, value=value)
        # print(z_sample.size())

        x_hat, _, _, _, _ = self.decoder.decode_sep(
            z_sample.reshape([z_sample.size()[0], self.z_dim]), label.to(device))
        #x_hat = self.decoder.decode_sep(
        #    z_sample.reshape([z_sample.size()[0], self.z_dim]), label.to(device))

        # print(x_hat.size())
        return x_hat, z_sample, z_masked, z_m, z_v, label

    def forward_recons(self, x, label, mask=None, value=None):
        # For generating reconstructions during inference
        z_sample, z_masked, z_m, z_v = self.encode(x, label=label, mask=mask, value=value)
        decoder_out, _, _, _, _ = self.decoder.decode_sep(
            z_sample.reshape([z_sample.size()[0], self.z_dim]), label.to(device))
        # decoder_out = torch.sigmoid(decoder_out)
        return decoder_out

    def training_step(self, batch, batch_idx):
        X, label = batch
        neg_elbo, kl, rec, x_hat, z_sample, cp_m = self._get_loss(X, label, mode="train")

        self.log('neg_ELBO', neg_elbo)
        self.log('kld', kl)
        self.log('reconstruction_err', rec)

        values = {
            'loss': neg_elbo,
            'kl': kl,
            'rec': rec,
            'x_hat': x_hat,
            'z_sample': z_sample,
            'cp_m': cp_m,
            'x': X
        }
        # print('X', X.size())
        return values


    def validation_step(self, batch, batch_idx):
        X, label = batch
        neg_elbo, kl, rec, x_hat, z_sample, cp_m = self._get_loss(X, label, mode="val")

        self.log('neg_ELBO', neg_elbo)
        self.log('kld', kl)
        self.log('reconstruction_err', rec)

    def test_step(self, batch, batch_idx, mask=3, value=11):
        X, label = batch
        # print('label', label)
        #print('X', X)
        # print(X.shape)
        x_hat, z_sample, z_masked, z_m, z_v, label = self.forward(X, label, mask=mask, value=value)
        x_hat = x_hat.reshape(-1, 4, 96, 96)
        mae = calculate_mae(x_hat, X)
        print("MSE of VAE", mae.data)
        x_hat = batch_rgba_to_rgb(x_hat)
        #print('x', x_hat)
        # print(mask)
        vutils.save_image(X,
                          os.path.join(self.logger.log_dir,
                                       "Real_Inference",
                                       f"real_{self.logger.name}_Epoch_{self.current_epoch}_{batch_idx}.png"),
                          normalize=True,
                          nrow=16)

        if mask == None:
            # print(1)
            vutils.save_image(x_hat.data,
                              os.path.join(self.logger.log_dir,
                                           "Reconstructions_Inference",
                                           f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              range=(0,1),
                              nrow=12)
        else:
            # print(0)
            # print(x_hat.shape)

            vutils.save_image(x_hat.data,
                              os.path.join(self.logger.log_dir,
                                           "Interventions_Inference",
                                           f"recons_{self.logger.name}_Epoch_{self.current_epoch}_{batch_idx}.png"),
                              nrow=16)

        z_sample = z_sample.detach().cpu().numpy()
        print(z_sample)
        label = label.detach().cpu().numpy()
        z_sample = z_sample.reshape(z_sample.shape[0], -1)
        # 🔥 Ensure shape consistency
        if z_sample.shape[0] != label.shape[0]:
            raise ValueError(f"Mismatch: z_sample has {z_sample.shape[0]} samples, label has {label.shape[0]} samples.")

        # Compute MIC & TIC
        mic_scores, tic_scores = compute_mic_tic(z_sample, label)
        print(f"MIC Scores:\n{mic_scores}")
        print(f"TIC Scores:\n{tic_scores}")

    def training_epoch_end(self, outputs):
        choice = random.choice(outputs)
        data = choice['x']
        # data = data.reshape(-1, 3, 128, 128)
        output_sample = choice['x_hat']
        output_sample = output_sample.reshape(-1, 4, 96, 96)
        # print(output_sample.size())
        # output_sample = output_sample * 0.5 + 0.5

        array1 = self.dag.A.detach().cpu()
        array_A = array1.numpy()
        plt.imshow(array_A, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title(f"Matrix Visualization at Epoch {self.current_epoch}")
        A_name = (f"A at Epoch {self.current_epoch}")
        plt.savefig('/home/2527823Y/DiffuseVAE-main/main/datasets/causal_result/A/{}'.format(A_name))
        plt.close()

        vutils.save_image(data,
                          os.path.join(self.logger.log_dir,
                                       "Real",
                                       f"real_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)
        vutils.save_image(output_sample.data,
                          os.path.join(self.logger.log_dir,
                                       "Reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          range=(0,1),
                          nrow=12)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        #lr_scheduler = CosineWarmupScheduler(optimizer,
        #                                     warmup=[200*self.warmup, 2*self.warmup, 2*self.warmup],
        #                                     offset=[10000, 0, 0],
        #                                     max_iters=self.max_iters)
        return [optimizer]#, [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def get_callbacks(exmp_inputs=None, dataset=None, **kwargs):
        lr_callback = LearningRateMonitor('step')
        return [lr_callback]


if __name__ == "__main__":
    #enc_block_config_str = "128x1,128d2,128t64,64x3,64d2,64t32,32x3,32d2,32t16,16x7,16d2,16t8,8x3,8d2,8t4,4x3,4d4,4t1,1x2"
    #enc_channel_config_str = "128:64,64:64,32:128,16:128,8:256,4:512,1:1024"

    #dec_block_config_str = "1x1,1u4,1t4,4x2,4u2,4t8,8x2,8u2,8t16,16x6,16u2,16t32,32x2,32u2,32t64,64x2,64u2,64t128,128x1"
    #dec_channel_config_str = "128:64,64:64,32:128,16:128,8:256,4:512,1:1024"

    cvae = CausalVAE
    sample = torch.randn(1, 3, 64, 64)
    out = cvae.training_step(sample, 0)
    print(cvae)
    print(out.shape)
