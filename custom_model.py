import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from pythae.data.datasets import BaseDataset
from pythae.models.base import BaseAE
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseEncoder
from pythae.models.nn.default_architectures import Encoder_VAE_MLP

from pydantic.dataclasses import dataclass
from pythae.models.base.base_config import BaseAEConfig


@dataclass
class MOLVAEConfig(BaseAEConfig):
    """VAE config class.
    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
    """

    num_mixtures: int = 10


def _compute_inv_stdv(logits):
    softplus = nn.Softplus(beta=hparams.model.output_gradient_smoothing_beta)
    if hparams.model.output_distribution_base == 'std':
        scales = torch.maximum(softplus(logits),
                               torch.as_tensor(np.exp(hparams.loss.min_mol_logscale)))
        inv_stdv = 1. / scales  # Not stable for sharp distributions
        log_scales = torch.log(scales)

    elif hparams.model.output_distribution_base == 'logstd':
        log_scales = torch.maximum(logits, torch.as_tensor(np.array(hparams.loss.min_mol_logscale)))
        inv_stdv = torch.exp(-hparams.model.output_gradient_smoothing_beta * log_scales)
    else:
        raise ValueError(f'distribution base {hparams.model.output_distribution_base} not known!!')

    return inv_stdv, log_scales


class MOLVAE(BaseAE):
    """Vanilla Variational Autoencoder model.
    Args:
        model_config (VAEConfig): The Variational Autoencoder configuration setting the main
        parameters of the model.
        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.
        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.
    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: MOLVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        BaseAE.__init__(self, model_config=model_config, decoder=decoder)

        self.model_name = "VAE"

        if encoder is None:
            if model_config.input_dim is None:
                raise AttributeError(
                    "No input dimension provided !"
                    "'input_dim' parameter of BaseAEConfig instance must be set to 'data_shape' "
                    "where the shape of the data is (C, H, W ..). Unable to build encoder "
                    "automatically"
                )

            encoder = Encoder_VAE_MLP(model_config)
            self.model_config.uses_default_encoder = True

        else:
            self.model_config.uses_default_encoder = False

        self.set_encoder(encoder)

    def forward(self, inputs: BaseDataset, **kwargs):
        """
        The VAE model
        Args:
            inputs (BaseDataset): The training dataset with labels
        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """

        x = inputs["data"]

        encoder_output = self.encoder(x)

        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)
        recon_x = self.decoder(z)["reconstruction"]

        loss, recon_loss, kld = self.loss_function(recon_x, x, mu, log_var, z)

        output = ModelOutput(
            recon_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x,
            z=z,
        )

        return output

    def loss_function(self, recon_x, x, mu, log_var, z):

        assert recon_x.size(1) == self.model_config.num_mixtures * (x.size(1) * 3 +1)

        logit_probs = recon_x[:, :self.model_config.num_mixtures, :, :]  # B, M, H, W
        l = recon_x[:, self.model_config.num_mixtures:, :, :]  # B, M*C*3 ,H, W
        l = l.reshape(x.size(0), x.size(1), 3 * self.model_config.num_mixtures , x.size(2), x.size(3))  # B, C, 3 * M, H, W

        means = l[:, :, :self.model_config.num_mixtures, :, :]  # B, C, M, H, W

        inv_stdv, log_scales = _compute_inv_stdv(
            l[:, :, self.model_config.num_mixtures: 2 * self.model_config.num_mixtures, :, :])

        model_coeffs = torch.tanh(
            l[:, :, 2 * self.model_config.num_mixtures: 3 * self.model_config.num_mixtures, :,
            :])  # B, C, M, H, W

        centered = targets - means  # B, C, M, H, W

        plus_in = inv_stdv * (centered + 1. / self.num_classes)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered - 1. / self.num_classes)
        cdf_min = torch.sigmoid(min_in)

        log_cdf_plus = plus_in - F.softplus(plus_in)  # log probability for edge case of 0 (before scaling)
        log_one_minus_cdf_min = -F.softplus(min_in)  # log probability for edge case of 255 (before scaling)

        # probability for all other cases
        cdf_delta = cdf_plus - cdf_min  # B, C, M, H, W

        mid_in = inv_stdv * centered
        # log probability in the center of the bin, to be used in extreme cases
        # (not actually used in this code)
        log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

        # the original implementation uses samples > 0.999, this ignores the largest possible pixel value (255)
        # which is mapped to 0.9922
        broadcast_targets = torch.broadcast_to(targets, size=[B, C, hparams.model.num_output_mixtures, H, W])
        log_probs = torch.where(broadcast_targets == self.min_pix_value, log_cdf_plus,
                                torch.where(broadcast_targets == self.max_pix_value, log_one_minus_cdf_min,
                                            torch.where(cdf_delta > 1e-5,
                                                        torch.log(torch.clamp(cdf_delta, min=1e-12)),
                                                        log_pdf_mid - np.log(self.num_classes / 2))))  # B, C, M, H, W

        log_probs = torch.sum(log_probs, dim=1) + F.log_softmax(logit_probs, dim=1)  # B, M, H, W
        negative_log_probs = -torch.logsumexp(log_probs, dim=1)  # B, H, W

        mean_axis = list(range(1, len(negative_log_probs.size())))
        per_example_loss = torch.sum(negative_log_probs, dim=mean_axis)  # B
        avg_per_example_loss = per_example_loss / (
                np.prod([negative_log_probs.size()[i] for i in mean_axis]) * hparams.data.channels)  # B

        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        return (recon_loss + KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def get_nll(self, data, n_samples=1, batch_size=100):
        """
        Function computed the estimate negative log-likelihood of the model. It uses importance
        sampling method with the approximate posterior distribution. This may take a while.
        Args:
            data (torch.Tensor): The input data from which the log-likelihood should be estimated.
                Data must be of shape [Batch x n_channels x ...]
            n_samples (int): The number of importance samples to use for estimation
            batch_size (int): The batchsize to use to avoid memory issues
        """

        if n_samples <= batch_size:
            n_full_batch = 1
        else:
            n_full_batch = n_samples // batch_size
            n_samples = batch_size

        log_p = []

        for i in range(len(data)):
            x = data[i].unsqueeze(0)

            log_p_x = []

            for j in range(n_full_batch):
                x_rep = torch.cat(batch_size * [x])

                encoder_output = self.encoder(x_rep)
                mu, log_var = encoder_output.embedding, encoder_output.log_covariance

                std = torch.exp(0.5 * log_var)
                z, _ = self._sample_gauss(mu, std)

                log_q_z_given_x = -0.5 * (
                    log_var + (z - mu) ** 2 / torch.exp(log_var)
                ).sum(dim=-1)
                log_p_z = -0.5 * (z ** 2).sum(dim=-1)

                recon_x = self.decoder(z)["reconstruction"]

                if self.model_config.reconstruction_loss == "mse":

                    log_p_x_given_z = -0.5 * F.mse_loss(
                        recon_x.reshape(x_rep.shape[0], -1),
                        x_rep.reshape(x_rep.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1) - torch.tensor(
                        [np.prod(self.input_dim) / 2 * np.log(np.pi * 2)]
                    ).to(
                        data.device
                    )  # decoding distribution is assumed unit variance  N(mu, I)

                elif self.model_config.reconstruction_loss == "bce":

                    log_p_x_given_z = -F.binary_cross_entropy(
                        recon_x.reshape(x_rep.shape[0], -1),
                        x_rep.reshape(x_rep.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1)

                log_p_x.append(
                    log_p_x_given_z + log_p_z - log_q_z_given_x
                )  # log(2*pi) simplifies

            log_p_x = torch.cat(log_p_x)

            log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))).item())
        return np.mean(log_p)