import argparse
import os, glob
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
parser.add_argument("--workers", type=int, default=0, help="Number of workers for dataloaders.")
parser.add_argument("--synth", action='store_true', help="Use synthetic training data.")
parser.add_argument("--gauss", action='store_true', help="Use different recon loss to better represent covariance.")
parser.add_argument("--amp", action='store_true', help="Use auto mixed precision in training.")
parser.add_argument("--resume", action='store_true', help="Find most recent run in output dir and resume from last checkpoint.")
parser.add_argument("--ffcv", action='store_true', help="Overwrite default dataloader with FFCV dataloaders.")
parser.add_argument("--root", type=str, default='./', help="Root dir to save output directory within.")
args = parser.parse_args()

my_training_config = BaseTrainerConfig(
	output_dir=os.path.join(args.root, args.name),
	num_epochs=args.epochs,
	learning_rate=args.lr,
	per_device_train_batch_size=args.batch_size,
	per_device_eval_batch_size=args.batch_size,
	train_dataloader_num_workers=args.workers,
	eval_dataloader_num_workers=args.workers,
	steps_saving=args.val_interval,
    steps_predict=args.val_interval,
	optimizer_cls="AdamW",
	optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.995)},
	scheduler_cls="ReduceLROnPlateau",
	scheduler_params={"patience": 5, "factor": 0.5},
    amp=args.amp,
)

if args.model == 'VAE':
    from pythae.models import VAE, VAEConfig
    Encoder = layers.Encoder_Conv_VAE 
    Decoder = layers.Decoder_Conv_GaussVAE  if args.gauss else layers.Decoder_Conv_AE 

    my_vae_config = VAEConfig(
        input_dim=(1, 192, 192),
        latent_dim=128 # match the 2020 Baur/Navab paper
    )

    class GaussVAE(VAE):
        def __init__(self, model_config, encoder, decoder):
            super().__init__(model_config, encoder, decoder)

        def loss_function(self, recon_x, x, mu, log_var, z):

            x_mu = recon_x["mu"].reshape(x.shape[0], -1)
            x_log_var = recon_x["log_var"].reshape(x.shape[0], -1)
            x = x.reshape(x.shape[0], -1)

            x_log_var = torch.clamp(x_log_var,-10,1)
            x_var = torch.exp(x_log_var)
            squared_difference = torch.square(x - x_mu)
            squared_diff_normed = torch.true_divide(squared_difference, x_var)
            log_likelihood_per_dim = 0.5 * (np.log(2 * np.pi) + x_log_var + squared_diff_normed)
            recon_loss = torch.sum(log_likelihood_per_dim, dim=1)

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
        input_dim=(1, 192, 192),
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
    Decoder = layers.Decoder_Conv_GaussVAE  if args.gauss else layers.Decoder_Conv_AE 

    my_vae_config = BetaVAEConfig(
        input_dim=(1, 192, 192),
        latent_dim=128 # match the 2020 Baur/Navab paper
    )

    class GaussBetaVAE(BetaVAE):
        def __init__(self, model_config, encoder, decoder):
            super().__init__(model_config, encoder, decoder)

        def loss_function(self, recon_x, x, mu, log_var, z):
            x_mu = recon_x["mu"].reshape(x.shape[0], -1)
            x_log_var = recon_x["log_var"].reshape(x.shape[0], -1)
            x = x.reshape(x.shape[0], -1)

            x_log_var = torch.clamp(x_log_var,-10,1)
            x_var = torch.exp(x_log_var)
            squared_difference = torch.square(x - x_mu)
            squared_diff_normed = torch.true_divide(squared_difference, x_var)
            log_likelihood_per_dim = 0.5 * (np.log(2 * np.pi) + x_log_var + squared_diff_normed)
            recon_loss = torch.sum(log_likelihood_per_dim, dim=1)

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
        input_dim=(1, 192, 192),
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
        input_dim=(1, 192, 192),
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


if args.resume:
    model_paths = glob.glob(os.path.join(args.root, args.name, '*', 'checkpoint_epoch_*'))
    model_paths = [{'Epoch':int(pth.split('_')[-1]), 'Path': pth} for pth in model_paths]
    model_path = sorted(model_paths, key=lambda d: d['Epoch'])[-1]
    print('Resuming training from folder {} at epoch #{}.'.format(model_path['Path'].split('/')[-2], model_path['Epoch']))
    my_vae_model.load_state_dict(torch.load(os.path.join(model_path['Path'],'model.pt'), map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))['model_state_dict'])
    epoch = model_path['Epoch']
    optimizer_state = torch.load(os.path.join(model_path['Path'],'optimizer.pt'), map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    scheduler_state = torch.load(os.path.join(model_path['Path'],'scheduler.pt'), map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
else:
    epoch = 0
    optimizer_state = None
    scheduler_state = None

pipeline = TrainingPipeline(
 	training_config=my_training_config,
 	model=my_vae_model
)

wandb_cb = WandbCallback() 

wandb_cb.setup(
training_config=my_training_config, # training config
model_config=my_vae_config, # model config
project_name="ano-synth", # specify your wandb project
entity_name="ff2023", # specify your wandb entity
name = args.name,
)

callbacks = [wandb_cb]

if args.synth:
    your_train_data, your_eval_data = dataloaders.get_synth_data(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
else:
    your_train_data, your_eval_data = dataloaders.get_mri_data(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

if args.ffcv:
    import dataloaders_ffcv
    if args.synth:
        ffcv_train, ffcv_val = dataloaders_ffcv.get_synth_ffcv(args.batch_size, args.workers)
    else:
        ffcv_train, ffcv_val = dataloaders_ffcv.get_mri_ffcv(args.batch_size, args.workers)
else:
    ffcv_train = None
    ffcv_val = None

pipeline(
    train_data=your_train_data,
    eval_data=your_eval_data,
    callbacks=callbacks,
    epoch=epoch+1,
    optimizer_state_dict=optimizer_state,
    scheduler_state_dict=scheduler_state,
    ffcv_train=ffcv_train,
    ffcv_val=ffcv_val
)