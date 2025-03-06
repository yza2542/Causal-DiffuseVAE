import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd, nn, optim
import util as ut


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim, in_channels=3, out_dim=None):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),            # 64x64x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),                     # 32x32x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),                     # 16x16x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),                    # 8x8x128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 4, 2, 1),                   # 4x4x128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 1),                      # 1x1x256
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 1)
        )# 1x1x512

        modules = []

        hidden_dims = [32, 64, 128, 256, 512, 512, 512]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2, inplace=True))
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

    def encode(self, x):
        z = self.conv(x)
        z = z.view(-1, 512)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(z)
        var = self.fc_var(z)
        var = F.softplus(var) + 1e-8

        return mu, var



class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, out_channels=3, out_dim=None):
        super().__init__()
        self.latent_dim = latent_dim

        self.convT = nn.Sequential(
            nn.Conv2d(latent_dim, 512, 1),  # 1x1x512
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 4),  # 4x4x256
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8x8x128
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),  # 16x16x128
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 32x32x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # 64x64x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 128x128x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, out_channels, 1),  # 128x128x3
        )

        modules = []

        hidden_dims = [32, 64, 128, 256, 512, 512, 512]

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True))
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=4,
                               stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dims[-1], out_channels=out_channels,
                      kernel_size=4, padding=1),
            nn.Tanh())

    def decode(self, z):
        z = z.view(-1, self.latent_dim, 1, 1)
        z = self.convT(z)
        return z


class ConvDec(nn.Module):
    def __init__(self, latent_dim, out_dim=None):
        super().__init__()
        self.concept = 4
        self.z_dim = latent_dim
        self.z1_dim = self.z_dim // self.concept
        self.net1 = ConvDecoder(self.z1_dim)
        self.net2 = ConvDecoder(self.z1_dim)
        self.net3 = ConvDecoder(self.z1_dim)
        self.net4 = ConvDecoder(self.z1_dim)
        self.net5 = nn.Sequential(
            nn.Linear(16, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024)
        )
        self.net6 = nn.Sequential(
            nn.Conv2d(16, 128, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, 4, 2, 1)
        )

    def decode_sep(self, z, u=None, y=None):
        z = z.view(-1, self.concept * self.z1_dim)  # 16x64

        zy = z if y is None else torch.cat((z, y), dim=1)
        # print(zy.shape)
        zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim // self.concept, dim=1)  # each is 16x16
        rx1 = self.net1.decode(zy1)
        # print(f"Hi: {rx1.size()}")
        rx2 = self.net2.decode(zy2)
        rx3 = self.net3.decode(zy3)
        rx4 = self.net4.decode(zy4)
        # z = torch.cat((rx1, rx2, rx3, rx4), dim=0)
        z = (rx1+rx2+rx3+rx4)/4
        # print(z.shape)
        # sys.exit(0)
        return z, z, z, z, z



