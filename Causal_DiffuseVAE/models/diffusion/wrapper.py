import pytorch_lightning as pl
import torch
import torch.nn as nn
from models.diffusion.spaced_diff import SpacedDiffusion
from models.diffusion.spaced_diff_form2 import SpacedDiffusionForm2
from models.diffusion.ddpm_form2 import DDPMv2
from util import space_timesteps


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


device = device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
class DDPMWrapper(pl.LightningModule):
    def __init__(
        self,
        online_network,
        target_network,
        vae,
        lr=2e-5,
        cfd_rate=0.0,
        n_anneal_steps=0,
        loss="l1",
        grad_clip_val=1.0,
        sample_from="target",
        resample_strategy="spaced",
        skip_strategy="uniform",
        sample_method="ddpm",
        conditional=True,
        eval_mode="sample",
        pred_steps=None,
        pred_checkpoints=[],
        temp=1.0,
        guidance_weight=0.0,
        z_cond=False,
        ddpm_latents=None,
        mask=None,
        value=None,
    ):
        super().__init__()
        assert loss in ["l1", "l2"]
        assert eval_mode in ["sample", "recons"]
        assert resample_strategy in ["truncated", "spaced"]
        assert sample_method in ["ddpm", "ddim"]
        assert skip_strategy in ["uniform", "quad"]

        self.z_cond = z_cond
        self.online_network = online_network.to('cuda:0')
        self.target_network = target_network.to('cuda:1')
        self.vae = vae
        self.cfd_rate = cfd_rate

        # Training arguments
        self.criterion = nn.MSELoss(reduction="mean") if loss == "l2" else nn.L1Loss()
        self.lr = lr
        self.grad_clip_val = grad_clip_val
        self.n_anneal_steps = n_anneal_steps

        # Evaluation arguments
        self.sample_from = sample_from
        self.conditional = conditional
        self.sample_method = sample_method
        self.resample_strategy = resample_strategy
        self.skip_strategy = skip_strategy
        self.eval_mode = eval_mode
        self.pred_steps = self.online_network.T if pred_steps is None else pred_steps
        self.pred_checkpoints = pred_checkpoints
        self.temp = temp
        self.guidance_weight = guidance_weight
        self.ddpm_latents = ddpm_latents
        self.mask = mask
        self.value = value
        # Disable automatic optimization
        self.automatic_optimization = False

        # Spaced Diffusion (for spaced re-sampling)
        self.spaced_diffusion = None

    def forward(
        self,
        x,
        cond=None,
        z=None,
        n_steps=None,
        ddpm_latents=None,
        checkpoints=[],
    ):
        sample_nw = (
            self.target_network if self.sample_from == "target" else self.online_network
        )
        spaced_nw = (
            SpacedDiffusionForm2
            if isinstance(self.online_network, DDPMv2)
            else SpacedDiffusion
        )
        # For spaced resampling
        if self.resample_strategy == "spaced":
            num_steps = n_steps if n_steps is not None else self.online_network.T
            indices = space_timesteps(sample_nw.T, num_steps, type=self.skip_strategy)
            if self.spaced_diffusion is None:
                self.spaced_diffusion = spaced_nw(sample_nw, indices).to(x.device)

            if self.sample_method == "ddim":
                return self.spaced_diffusion.ddim_sample(
                    x,
                    cond=cond,
                    z_vae=z,
                    guidance_weight=self.guidance_weight,
                    checkpoints=checkpoints,
                )
            return self.spaced_diffusion(
                x,
                cond=cond,
                z_vae=z,
                guidance_weight=self.guidance_weight,
                checkpoints=checkpoints,
                ddpm_latents=ddpm_latents,
            )

        # For truncated resampling
        if self.sample_method == "ddim":
            raise ValueError("DDIM is only supported for spaced sampling")
        return sample_nw.sample(
            x,
            cond=cond,
            z_vae=z,
            n_steps=n_steps,
            guidance_weight=self.guidance_weight,
            checkpoints=checkpoints,
            ddpm_latents=ddpm_latents,
        )

    def training_step(self, batch, batch_idx):
        # Optimizers
        optim = self.optimizers()
        lr_sched = self.lr_schedulers()

        cond = None
        z = None
        if self.conditional:
            x, label = batch
            # x, classifier_label, causal_label = batch
            # print(x.shape)
            # x = x.reshape(-1, 1, 28, 28)
            with torch.no_grad():
                # z_sample, z_masked, mu, logvar, z_class = self.vae.encode(x, causal_label)
                z_sample, z_masked, mu, logvar = self.vae.encode(x, label)
                # print(z.shape)
                z = z_sample
                # print(z.size())
                cond, _, _, _, _ = self.vae.decoder.decode_sep(z_sample.reshape([z_sample.size()[0], 384]), label.to(self.device))
                # cond = torch.sigmoid(cond)
                # print('cond', cond)
                cond = cond.reshape(x.size())
                cond = batch_rgba_to_rgb(cond)
                cond = 2 * cond - 1

                # cond = cond.view(16, 8, 1, 1)

            # Set the conditioning signal based on clf-free guidance rate
            if torch.rand(1)[0] < self.cfd_rate:
                cond = torch.zeros_like(x)
                z = torch.zeros_like(z)
        else:
            x, label = batch

        x = batch_rgba_to_rgb(x)
        # Sample timepoints
        t = torch.randint(
            0, self.online_network.T, size=(x.size(0),), device=self.device
        )
        # Sample noise
        eps = torch.randn_like(x)



        # Predict noise
        eps_pred = self.online_network(
            x, eps, t, low_res=cond, z=z.squeeze() if self.z_cond else None
        )

        # Compute loss
        loss = self.criterion(eps, eps_pred)

        # Clip gradients and Optimize
        optim.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(
            self.online_network.decoder.parameters(), self.grad_clip_val
        )
        optim.step()

        # Scheduler step
        lr_sched.step()
        self.log("loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if not self.conditional:
            if self.guidance_weight != 0.0:
                raise ValueError(
                    "Guidance weight cannot be non-zero when using unconditional DDPM"
                )
            x_t = batch
            return self(
                x_t,
                cond=None,
                z=None,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
                ddpm_latents=None,
            )

        if self.eval_mode == "sample":
            x_t, z = batch
            recons = self.vae(z)
            # recons = 2 * recons - 1

            # Initial temperature scaling
            x_t = x_t * self.temp

            # Formulation-2 initial latent
            if isinstance(self.online_network, DDPMv2):
                x_t = recons + self.temp * torch.randn_like(recons)
        else:
            if self.mask is None:
                # x, classifier_label, causal_label = batch
                x, label = batch
                recons = self.vae.forward_recons(x, label)
                recons = recons.reshape(x.size())
                recons = 2 * recons - 1

                # DDPM encoder
                x_t = self.online_network.compute_noisy_input(
                    x,
                    torch.randn_like(x),
                    torch.tensor(
                        [self.online_network.T - 1] * x.size(0), device=x.device
                    ),
                )

                if isinstance(self.online_network, DDPMv2):
                    x_t += recons
            else:
                # x, classifier_label, causal_label = batch
                # recons = self.vae.forward_recons(x, causal_label, self.mask, self.value)
                img, label = batch
                recons = self.vae.forward_recons(img, label, self.mask, self.value)
                # print(img.size())
                # print(recons.size())
                recons = recons.reshape(img.size())
                # recons = recons.reshape(-1, 4, 96, 96)
                # recons = recons.reshape(-1, 1, 28, 28)
                recons = batch_rgba_to_rgb(recons)
                recons = 2 * recons - 1

                img = batch_rgba_to_rgb(img)
                # img = x.reshape(-1, 1, 28, 28)
                # DDPM encoder
                x_t = self.online_network.compute_noisy_input(
                    img,
                    torch.randn_like(img),
                    torch.tensor(
                        [self.online_network.T - 1] * img.size(0), device=img.device
                    ),
                )
                # print(x_t.size())

                if isinstance(self.online_network, DDPMv2):
                    x_t += recons

        return (
            self(
                x_t,
                cond=recons,
                z=z.squeeze() if self.z_cond else None,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
                ddpm_latents=self.ddpm_latents,
            ),
            recons,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.online_network.decoder.parameters(), lr=self.lr
        )

        # Define the LR scheduler (As in Ho et al.)
        if self.n_anneal_steps == 0:
            lr_lambda = lambda step: 1.0
        else:
            lr_lambda = lambda step: min(step / self.n_anneal_steps, 1.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "strict": False,
            },
        }
