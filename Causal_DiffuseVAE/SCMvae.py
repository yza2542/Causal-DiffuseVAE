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
from conv_encoder_decoder_celeba import ConvEncoder, ConvDec
from pytorch_lightning.callbacks import LearningRateMonitor
from modules import CosineWarmupScheduler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")




class DeterministicWarmup(object):
    """
    Linear deterministic warm-up as described in
    [S?nderby 2016].
    """
    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1/n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t
        return self.t

def convert_to_np(obj):
    obj = obj.permute(0, 2, 3, 1).contiguous()
    obj = obj.detach().cpu().numpy()

    obj_list = []
    for _, out in enumerate(obj):
        obj_list.append(out)
    return obj_list

def matrix_poly(matrix, d, device):
    x = torch.eye(d).to(device)+ torch.div(matrix.to(device), d).to(device)
    return torch.matrix_power(x, d)

def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A

def dag_right_linear(input, weight, bias=None):
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        ret = torch.addmm(bias, input, weight.t())
    else:
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        ret = output
    return ret


def dag_left_linear(input, weight, bias=None):
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        ret = torch.addmm(bias, input, weight.t())
    else:
        output = weight.matmul(input)
        if bias is not None:
            output += bias
        ret = output
    return ret


class CausalVAE(pl.LightningModule):
    def __init__(self, model_name='cvae', z_dim=512, z1_dim=4, z2_dim=128, inference=False, alpha=0.1, beta=1, lambda_v=1e-5, lr=1e-4, use_causal_prior=True, warmup=100, max_iters=100, **kwargs):
        super().__init__()
        self.name = model_name
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.channel = 4
        self.scale = np.array([[0, 1], [0, 1], [0, 1], [0, 1]])
        self.encoder = ConvEncoder(self.z_dim)
        self.decoder = ConvDec(self.z_dim)
        self.inference = inference
        self.warmup = warmup
        self.max_iters = max_iters
        self.dag = mask.DagLayer(self.z1_dim, self.z1_dim, i=self.inference)
        # self.cause = nn.CausalLayer(self.z_dim, self.z1_dim, self.z2_dim)
        self.mask_u = mask.MaskLayer(self.z1_dim, z1_dim=4)
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
        neg_elbo = rec + 0.1*kl

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

    def test_step(self, batch, batch_idx, mask=0, value=-1):
        X, label = batch
        print('label', label)
        #print('X', X)
        # print(X.shape)
        x_hat, z_sample, z_masked, z_m, z_v, label = self.forward(X, label, mask=mask, value=value)
        x_hat = x_hat.reshape(X.size())
        #print('x', x_hat)
        # print(mask)
        vutils.save_image(X,
                          os.path.join(self.logger.log_dir,
                                       "Real_Inference",
                                       f"real_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=16)

        if mask == None:
            # print(1)
            vutils.save_image(x_hat.data,
                              os.path.join(self.logger.log_dir,
                                           "Reconstructions_Inference",
                                           f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        else:
            # print(0)
            # print(x_hat.shape)
            vutils.save_image(x_hat.data,
                              os.path.join(self.logger.log_dir,
                                           "Interventions_Inference",
                                           f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=16)

    def training_epoch_end(self, outputs):
        choice = random.choice(outputs)
        data = choice['x']
        # data = data.reshape(-1, 3, 128, 128)
        output_sample = choice['x_hat']
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
                          normalize=True,
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
