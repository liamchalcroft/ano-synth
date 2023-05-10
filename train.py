import argparse
from pipe import TrainingPipeline, WandbCallback
from pythae.models import VAE, AE, BetaVAE, VQVAE, RHVAE
from pythae.trainers import BaseTrainerConfig
import torch
import numpy as np
# from pythae.trainers.training_callbacks import WandbCallback
import dataloaders
import layers


parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--name", type=str, help="Name of WandB run.")
parser.add_argument("--model", type=str, help="Model to use. Full list of options available in Pythae docs.")
parser.add_argument(
    "--epochs", type=int, default=200, help="Number of epochs for training."
)
parser.add_argument(
    "--epoch_length", type=int, default=100, help="Number of iterations per epoch."
)
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
parser.add_argument(
    "--val_interval", type=int, default=2, help="Validation interval."
)
parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
parser.add_argument("--synth", action='store_true', help="Use synthetic training data.")
parser.add_argument("--gauss", action='store_true', help="Use different recon loss to better represent covariance.")
args = parser.parse_args()

my_training_config = BaseTrainerConfig(
	output_dir=args.name,
	num_epochs=args.epochs,
	learning_rate=args.lr,
	per_device_train_batch_size=args.batch_size,
	per_device_eval_batch_size=args.batch_size,
	train_dataloader_num_workers=0,
	eval_dataloader_num_workers=0,
	steps_saving=5,
    steps_predict=5,
	optimizer_cls="AdamW",
	optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.995)},
	scheduler_cls="ReduceLROnPlateau",
	scheduler_params={"patience": 5, "factor": 0.5}
)

if args.model == 'VAE':
    from pythae.models import VAE, VAEConfig
    Encoder = layers.Encoder_Conv_VAE 
    Decoder = layers.Decoder_Conv_AE 

    my_vae_config = VAEConfig(
        input_dim=(1, 128, 128),
        latent_dim=128 # match the 2020 Baur/Navab paper
    )

    class GaussVAE(VAE):
        def __init__(self, model_config, encoder, decoder):
            super().__init__(model_config, encoder, decoder)
            # self.logvar_x = torch.nn.Parameter(torch.tensor(np.log(0.1)))

        def loss_function(self, recon_x, x, mu, log_var, z):

            recon_loss = torch.nn.functional.mse_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

            # recon_loss = 0.5 * (torch.log(np.pi * 2 * self.logvar_x.exp()) + recon_loss / self.logvar_x.exp())
            recon_loss = 0.5 * (torch.log(2 * np.pi * recon_loss) + 1)

            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

            return (
                (recon_loss + KLD).mean(dim=0),
                recon_loss.mean(dim=0),
                KLD.mean(dim=0),
            )

    if args.gauss:
        my_vae_model = GaussVAE(
            model_config=my_vae_config,
            encoder=Encoder(my_vae_config),
            decoder=Decoder(my_vae_config),
        )
    else:
        my_vae_model = VAE(
            model_config=my_vae_config,
            encoder=Encoder(my_vae_config),
            decoder=Decoder(my_vae_config),
        )
elif args.model == 'AE':
    from pythae.models import AE, AEConfig
    Encoder = layers.Encoder_Conv_AE 
    Decoder = layers.Decoder_Conv_AE 

    my_vae_config = AEConfig(
        input_dim=(1, 128, 128),
        latent_dim=128 # match the 2020 Baur/Navab paper
    )

    my_vae_model = AE(
        model_config=my_vae_config,
        encoder=Encoder(my_vae_config),
        decoder=Decoder(my_vae_config),
    )
elif args.model == 'BetaVAE':
    from pythae.models import BetaVAE, BetaVAEConfig
    Encoder = layers.Encoder_Conv_VAE 
    Decoder = layers.Decoder_Conv_AE 

    my_vae_config = BetaVAEConfig(
        input_dim=(1, 128, 128),
        latent_dim=128 # match the 2020 Baur/Navab paper
    )

    class GaussBetaVAE(BetaVAE):
        def __init__(self, model_config, encoder, decoder):
            super().__init__(model_config, encoder, decoder)
            # self.logvar_x = torch.nn.Parameter(torch.tensor(np.log(0.1)))

        def loss_function(self, recon_x, x, mu, log_var, z):

            recon_loss = torch.nn.functional.mse_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

            # recon_loss = 0.5 * (torch.log(np.pi * 2 * self.logvar_x.exp()) + recon_loss / self.logvar_x.exp())
            recon_loss = 0.5 * (torch.log(2 * np.pi * recon_loss) + 1)

            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

            return (
                (recon_loss + self.beta * KLD).mean(dim=0),
                recon_loss.mean(dim=0),
                KLD.mean(dim=0),
            )

    if args.gauss:
        my_vae_model = GaussBetaVAE(
            model_config=my_vae_config,
            encoder=Encoder(my_vae_config),
            decoder=Decoder(my_vae_config),
        )
    else:
        my_vae_model = BetaVAE(
            model_config=my_vae_config,
            encoder=Encoder(my_vae_config),
            decoder=Decoder(my_vae_config),
        )
elif args.model == 'VQVAE':
    from pythae.models import VQVAE, VQVAEConfig
    Encoder = layers.Encoder_Conv_AE 
    Decoder = layers.Decoder_Conv_AE 

    my_vae_config = VQVAEConfig(
        input_dim=(1, 128, 128),
        latent_dim=128 # match the 2020 Baur/Navab paper
    )

    my_vae_model = VQVAE(
        model_config=my_vae_config,
        encoder=Encoder(my_vae_config),
        decoder=Decoder(my_vae_config),
    )
elif args.model == 'RHVAE':
    from pythae.models import RHVAE, RHVAEConfig
    Encoder = layers.Encoder_Conv_VAE 
    Decoder = layers.Decoder_Conv_AE 

    my_vae_config = RHVAEConfig(
        input_dim=(1, 128, 128),
        latent_dim=128 # match the 2020 Baur/Navab paper
    )

    my_vae_model = RHVAE(
        model_config=my_vae_config,
        encoder=Encoder(my_vae_config),
        decoder=Decoder(my_vae_config),
    )
else:
    raise ExceptionError('Invalid argument provided for --model flag. Please choose from [VAE, AE, BetaVAE, VQVAE, RHVAE].')

# import torch
# print(my_vae_model.decoder(torch.ones(1,128))['reconstruction'].shape)
# exit()

pipeline = TrainingPipeline(
 	training_config=my_training_config,
 	model=my_vae_model
)

wandb_cb = WandbCallback() 

wandb_cb.setup(
training_config=my_training_config, # training config
model_config=my_vae_config, # model config
project_name="ano-synth", # specify your wandb project
entity_name="lchalcroft", # specify your wandb entity
name = args.name,
)

callbacks = [wandb_cb]

if args.synth:
    your_train_data, your_eval_data = dataloaders.get_synth_data(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
else:
    your_train_data, your_eval_data = dataloaders.get_mri_data(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

pipeline(
    train_data=your_train_data,
    eval_data=your_eval_data,
    callbacks=callbacks
)