import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd, nn, optim
import util as ut


# class ConvEncoder(nn.Module):
#     def __init__(self, latent_dim, in_channels=1):  # MNIST has 1 input channel
#         super().__init__()
#         self.latent_dim = latent_dim
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Output: 32x28x28
#             nn.ReLU(),
#             nn.MaxPool2d(2),  # Output: 32x14x14
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: 64x14x14
#             nn.ReLU(),
#             nn.MaxPool2d(2),  # Output: 64x7x7
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Output: 128x7x7
#             nn.ReLU()
#         )
#
#         # Fully connected layers for mu and var
#         self.fc_mu = nn.Linear(128 * 7 * 7, latent_dim)
#         self.fc_var = nn.Linear(128 * 7 * 7, latent_dim)
#
#     def encode(self, x):
#         x = x.reshape(-1, 1, 28, 28)
#         z = self.conv(x)
#         z = z.view(z.size(0), -1)  # Flatten [batch_size, 512]
#
#         mu = self.fc_mu(z)
#         var = self.fc_var(z)
#         var = F.softplus(var) + 1e-8  # Ensure positive variance
#
#         return z, mu, var


# class ConvEncoder(nn.Module):
#     def __init__(self, latent_dim, in_channels=1):  # MNIST has 1 input channel
#         super().__init__()
#         self.latent_dim = latent_dim
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, 32, 4, 2, 1),           # 28x28 -> 14x14x32
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(32, 64, 4, 2, 1),                    # 14x14 -> 7x7x64
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, 4, 1),                      # 7x7 -> 4x4x128
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 256, 4, 1),                     # 4x4 -> 1x1x256
#         )
#
#         # Fully connected layers for mu and var
#         self.fc_mu = nn.Linear(256, latent_dim)
#         self.fc_var = nn.Linear(256, latent_dim)
#
#     def encode(self, x):
#         x = x.reshape(-1, 1, 28, 28)
#         z = self.conv(x)
#         z = z.view(z.size(0), -1)  # Flatten [batch_size, 512]
#
#         mu = self.fc_mu(z)
#         var = self.fc_var(z)
#         var = F.softplus(var) + 1e-8  # Ensure positive variance
#
#         return z, mu, var


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim, in_channels=1):  # MNIST has 1 input channel
        super().__init__()
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),           # 28x28 -> 14x14x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),                    # 14x14 -> 7x7x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 1),                      # 7x7 -> 4x4x128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 1),                     # 4x4 -> 1x1x256
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 1)                         # 1x1x256 -> 1x1x512
        )

        # Fully connected layers for mu and var
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_var = nn.Linear(512, latent_dim)

    def encode(self, x):
        x = x.reshape(-1, 1, 28, 28)
        z = self.conv(x)
        z = z.view(z.size(0), -1)  # Flatten [batch_size, 512]

        mu = self.fc_mu(z)
        var = self.fc_var(z)
        var = F.softplus(var) + 1e-8  # Ensure positive variance

        return z, mu, var

# class ConvEncoder(nn.Module):
#     def __init__(self, latent_dim, in_channels=1):  # MNIST has 1 input channel
#         super().__init__()
#         self.latent_dim = latent_dim
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, 32, 4, 2, 1),           # 28x28 -> 14x14x32
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(32, 64, 4, 2, 1),                    # 14x14 -> 7x7x64
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, 4, 1),                      # 7x7 -> 4x4x128
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 256, 4, 1),                     # 4x4 -> 1x1x256
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(256, 512, 1),                        # 1x1x256 -> 1x1x512
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(512, 1024, 1)                        # 1x1x512 -> 1x1x1024
#         )
#
#         # Fully connected layers for mu and var
#         self.fc_mu = nn.Linear(1024, latent_dim)
#         self.fc_var = nn.Linear(1024, latent_dim)
#
#     def encode(self, x):
#         x = x.reshape(-1, 1, 28, 28)
#         z = self.conv(x)
#         z = z.view(z.size(0), -1)  # Flatten [batch_size, 1024]
#
#         mu = self.fc_mu(z)
#         var = self.fc_var(z)
#         var = F.softplus(var) + 1e-8  # Ensure positive variance
#
#         return z, mu, var



class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, out_channels=1):  # MNIST is grayscale (1 channel)
        super().__init__()
        self.latent_dim = latent_dim

        self.convT = nn.Sequential(
            nn.Conv2d(latent_dim, 512, 1),  # 1x1xlatent_dim -> 1x1x512
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 1, 0),  # 1x1x512 -> 4x4x256
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4x4x256 -> 8x8x128
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8x8x128 -> 16x16x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 3),  # 16x16x64 -> 28x28x1 (fixed padding)
            nn.Sigmoid()  # Grayscale pixel values between 0 and 1
        )

    def decode(self, z):
        # Reshape latent vector to [batch_size, latent_dim, 1, 1]
        z = z.view(-1, self.latent_dim, 1, 1)  # Reshape to 1x1
        z = self.convT(z)  # Apply transposed conv layers
        return z

class ConvDec(nn.Module):
    def __init__(self, latent_dim, out_dim=None):
        super().__init__()
        self.concept = 2
        self.z_dim = latent_dim
        self.z1_dim = self.z_dim // self.concept
        self.net1 = ConvDecoder(self.z1_dim, out_channels=1)  # Grayscale
        self.net2 = ConvDecoder(self.z1_dim, out_channels=1)  # Grayscale
        self.net3 = ConvDecoder(self.z1_dim, out_channels=1)  # Grayscale
        self.net4 = ConvDecoder(self.z1_dim, out_channels=1)  # Grayscale
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
            nn.ConvTranspose2d(32, 1, 4, 2, 1)  # MNIST grayscale output (28x28x1)
        )

    def decode_sep(self, z, u=None, y=None):
        z = z.view(-1, self.concept * self.z1_dim)  # 16x64

        zy = z if y is None else torch.cat((z, y), dim=1)
        zy1, zy2 = torch.split(zy, self.z_dim // self.concept, dim=1)  # Each is 16x16
        rx1 = self.net1.decode(zy1)
        rx2 = self.net2.decode(zy2)


        z = (rx1 + rx2) / 2  # Combine outputs

        return z, z, z, z, z  # You can adjust return values as needed

# class ConvDecoder(nn.Module):
#     def __init__(self, latent_dim, out_channels=1):  # MNIST is grayscale (1 channel)
#         super().__init__()
#         self.latent_dim = latent_dim
#
#         self.convT = nn.Sequential(
#             nn.Conv2d(latent_dim, 512, 1),               # 1x1xlatent_dim -> 1x1x512
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.ConvTranspose2d(512, 256, 4, 1, 0),       # 1x1x512 -> 4x4x256
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.ConvTranspose2d(256, 128, 4, 2, 1),       # 4x4x256 -> 8x8x128
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.ConvTranspose2d(128, 64, 4, 2, 1),        # 8x8x128 -> 16x16x64
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.ConvTranspose2d(64, out_channels, 4, 2, 3),  # 16x16x64 -> 28x28x1 (fixed padding)
#             nn.Sigmoid()  # Grayscale pixel values between 0 and 1
#         )
#
#     def decode(self, z):
#         # Reshape latent vector to [batch_size, latent_dim, 1, 1]
#         z = z.view(-1, self.latent_dim, 1, 1)  # Reshape to 1x1
#         z = self.convT(z)  # Apply transposed conv layers
#         return z
#
#
# class ConvDec(nn.Module):
#     def __init__(self, latent_dim, out_dim=None):
#         super().__init__()
#         self.concept = 2
#         self.z_dim = latent_dim
#         self.z1_dim = self.z_dim // self.concept  # 512 if latent_dim is 1024
#         self.net1 = ConvDecoder(self.z1_dim, out_channels=1)  # Grayscale
#         self.net2 = ConvDecoder(self.z1_dim, out_channels=1)  # Grayscale
#         self.net3 = ConvDecoder(self.z1_dim, out_channels=1)  # Grayscale
#         self.net4 = ConvDecoder(self.z1_dim, out_channels=1)  # Grayscale
#         self.net5 = nn.Sequential(
#             nn.Linear(16, 512),
#             nn.BatchNorm1d(512),
#             nn.Linear(512, 1024),
#             nn.BatchNorm1d(1024)
#         )
#         self.net6 = nn.Sequential(
#             nn.Conv2d(16, 128, 1),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(128, 64, 4),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(64, 64, 4, 2, 1),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(64, 32, 4, 2, 1),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(32, 1, 4, 2, 1)  # MNIST grayscale output (28x28x1)
#         )
#
#     def decode_sep(self, z, u=None, y=None):
#         z = z.view(-1, self.concept * self.z1_dim)  # 16x512 if latent_dim is 1024
#
#         zy = z if y is None else torch.cat((z, y), dim=1)
#         zy1, zy2 = torch.split(zy, self.z_dim // self.concept, dim=1)  # Each is 16x512
#         rx1 = self.net1.decode(zy1)
#         rx2 = self.net2.decode(zy2)
#
#         z = (rx1 + rx2) / 2  # Combine outputs
#
#         return z, z, z, z, z  # You can adjust return values as needed

# class ConvDecoder(nn.Module):
#     def __init__(self, latent_dim, out_channels=1):  # MNIST is grayscale (1 channel)
#         super().__init__()
#         self.latent_dim = latent_dim
#
#         self.convT = nn.Sequential(
#             nn.Conv2d(latent_dim, 128, 1),               # 1x1xlatent_dim -> 1x1x128
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.ConvTranspose2d(128, 64, 4, 1, 0),        # 1x1x128 -> 4x4x64
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.ConvTranspose2d(64, 32, 4, 2, 1),         # 4x4x64 -> 8x8x32
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.ConvTranspose2d(32, 16, 4, 2, 1),         # 8x8x32 -> 16x16x16
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.ConvTranspose2d(16, out_channels, 4, 2, 3),  # 16x16x16 -> 28x28x1 (fixed padding)
#             nn.Sigmoid()  # Grayscale pixel values between 0 and 1
#         )
#
#     def decode(self, z):
#         # Reshape latent vector to [batch_size, latent_dim, 1, 1]
#         z = z.view(-1, self.latent_dim, 1, 1)  # Reshape to 1x1
#         z = self.convT(z)  # Apply transposed conv layers
#         return z
#
#
# class ConvDec(nn.Module):
#     def __init__(self, latent_dim, out_dim=None):
#         super().__init__()
#         self.concept = 2
#         self.z_dim = latent_dim
#         self.z1_dim = self.z_dim // self.concept  # 128 if latent_dim is 256
#         self.net1 = ConvDecoder(self.z1_dim, out_channels=1)  # Grayscale
#         self.net2 = ConvDecoder(self.z1_dim, out_channels=1)  # Grayscale
#         self.net3 = ConvDecoder(self.z1_dim, out_channels=1)  # Grayscale
#         self.net4 = ConvDecoder(self.z1_dim, out_channels=1)  # Grayscale
#         self.net5 = nn.Sequential(
#             nn.Linear(16, 128),
#             nn.BatchNorm1d(128),
#             nn.Linear(128, 256),
#             nn.BatchNorm1d(256)
#         )
#         self.net6 = nn.Sequential(
#             nn.Conv2d(16, 64, 1),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(64, 32, 4),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(32, 32, 4, 2, 1),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(32, 16, 4, 2, 1),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(16, 1, 4, 2, 1)  # MNIST grayscale output (28x28x1)
#         )
#
#     def decode_sep(self, z, u=None, y=None):
#         z = z.view(-1, self.concept * self.z1_dim)  # 16x128 if latent_dim is 256
#
#         zy = z if y is None else torch.cat((z, y), dim=1)
#         zy1, zy2 = torch.split(zy, self.z_dim // self.concept, dim=1)  # Each is 16x128
#         rx1 = self.net1.decode(zy1)
#         rx2 = self.net2.decode(zy2)
#
#         z = (rx1 + rx2) / 2  # Combine outputs
#
#         return z, z, z, z, z  # You can adjust return values as needed


# class ConvDecoder(nn.Module):
#     def __init__(self, latent_dim, out_channels=1):  # MNIST is grayscale (1 channel)
#         super().__init__()
#         self.latent_dim = latent_dim
#
#         self.convT = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 64x14x14
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 32x28x28
#             nn.ReLU(),
#             nn.Conv2d(32, 1, kernel_size=1),  # Output: 1x28x28
#             # nn.Sigmoid()  # Maps output to [0, 1] for BCE loss
#         )
#         self.fc_decode = nn.Linear(latent_dim, 128 * 7 * 7)
#
#     def decode(self, z):
#         # Reshape latent vector to [batch_size, latent_dim, 1, 1]
#         z = self.fc_decode(z).view(-1, 128, 7, 7)  # Reshape to 1x1
#         z = self.convT(z)  # Apply transposed conv layers
#         return z
#
#
# class ConvDec(nn.Module):
#     def __init__(self, latent_dim, out_dim=None):
#         super().__init__()
#         self.concept = 2
#         self.z_dim = latent_dim
#         self.z1_dim = self.z_dim // self.concept  # 128 if latent_dim is 256
#         self.net1 = ConvDecoder(self.z1_dim, out_channels=1)  # Grayscale
#         self.net2 = ConvDecoder(self.z1_dim, out_channels=1)  # Grayscale
#         self.net3 = ConvDecoder(self.z1_dim, out_channels=1)  # Grayscale
#         self.net4 = ConvDecoder(self.z1_dim, out_channels=1)  # Grayscale
#         self.net5 = nn.Sequential(
#             nn.Linear(16, 128),
#             nn.BatchNorm1d(128),
#             nn.Linear(128, 256),
#             nn.BatchNorm1d(256)
#         )
#         self.net6 = nn.Sequential(
#             nn.Conv2d(16, 64, 1),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(64, 32, 4),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(32, 32, 4, 2, 1),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(32, 16, 4, 2, 1),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(16, 1, 4, 2, 1)  # MNIST grayscale output (28x28x1)
#         )
#
#     def decode_sep(self, z, u=None, y=None):
#         z = z.view(-1, self.concept * self.z1_dim)  # 16x128 if latent_dim is 256
#
#         zy = z if y is None else torch.cat((z, y), dim=1)
#         zy1, zy2 = torch.split(zy, self.z_dim // self.concept, dim=1)  # Each is 16x128
#         rx1 = self.net1.decode(zy1)
#         rx2 = self.net2.decode(zy2)
#
#         z = (rx1 + rx2) / 2  # Combine outputs
#
#         return z, z, z, z, z  # You can adjust return values as needed



