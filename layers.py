import torch
import torch.nn as nn
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput

class Encoder_Conv_VAE(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = args.input_dim[0]

        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.n_channels, 32, 5, 2, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 5, 2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 5, 2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 5, 2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 16, 1, 1, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
        )

        self.embedding = nn.Linear(self.input_dim[1]*self.input_dim[2]//16, args.latent_dim)
        self.log_var = nn.Linear(self.input_dim[1]*self.input_dim[2]//16, args.latent_dim)

    def forward(self, x: torch.Tensor):
        h1 = self.conv_layers(x).reshape(x.shape[0], -1)
        output = ModelOutput(
            embedding=self.embedding(h1),
            log_covariance=self.log_var(h1)
        )
        return output


class Encoder_Conv_AE(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = args.input_dim[0]

        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.n_channels, 32, 5, 2, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 5, 2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 5, 2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 5, 2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 16, 1, 1, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
        )

        self.embedding = nn.Linear(self.input_dim[1]*self.input_dim[2]//16, args.latent_dim)

    def forward(self, x: torch.Tensor):
        h1 = self.conv_layers(x).reshape(x.shape[0], -1)
        output = ModelOutput(
            embedding=self.embedding(h1)
        )
        return output


class Decoder_Conv_AE(BaseDecoder):
    def __init__(self, args):
        BaseDecoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = args.input_dim[0]
            

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(16, 128, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 128, 5, 2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 5, 2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 5, 2, padding=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 32, 5, 2, padding=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, self.n_channels, 1, 1),
            nn.Sigmoid()
        )

        self.fc = nn.Linear(args.latent_dim, self.input_dim[1]*self.input_dim[2]//16)

    def forward(self, z: torch.Tensor):
        h1 = self.fc(z)
        hw = int((h1.shape[-1] // 16) ** 0.5)
        h1 = h1.reshape(h1.shape[0], 16, hw, hw)
        output = ModelOutput(reconstruction=self.deconv_layers(h1))

        return output


class Decoder_Conv_GaussVAE(BaseDecoder):
    def __init__(self, args):
        BaseDecoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = args.input_dim[0]
            

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(16, 128, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 128, 5, 2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 5, 2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 5, 2, padding=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 32, 5, 2, padding=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        self.mu = nn.Sequential(
            nn.ConvTranspose2d(32, self.n_channels, 1, 1),
            nn.Sigmoid()
        )
        self.log_var = nn.ConvTranspose2d(32, self.n_channels, 1, 1)

        self.fc = nn.Linear(args.latent_dim, self.input_dim[1]*self.input_dim[2]//16)

    def forward(self, z: torch.Tensor):
        h1 = self.fc(z)
        hw = int((h1.shape[-1] // 16) ** 0.5)
        h1 = h1.reshape(h1.shape[0], 16, hw, hw)
        out = self.deconv_layers(h1)
        mu = self.mu(out)
        log_var = self.log_var(out)
        output = ModelOutput(reconstruction=mu, mu=mu, log_var=log_var)

        return output