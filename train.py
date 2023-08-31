import argparse
import os, glob
from models import Autoencoder, AutoencoderKL, GaussAutoencoderKL, VQVAE
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import dataloaders
import train_utils
import wandb


if __name__ =='__main__':
    
    parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", type=str, help="Name of WandB run.")
    parser.add_argument("--model", type=str, help="Model to use.",
                        choices=["AE", "RAE", "SAMBA", "VAE", "BetaVAE", "GaussVAE", "VQVAE"])
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs for training.")
    parser.add_argument("--epoch_length", type=int, default=100, help="Number of iterations per epoch.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--recon_weight", type=float, default=100, help="Weighting for recon term in loss function.")
    parser.add_argument("--val_interval", type=int, default=2, help="Validation interval.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--beta_init", type=int, default=0, help="Initial beta (for BetaVAE only).")
    parser.add_argument("--beta_final", type=int, default=20, help="Final beta (for BetaVAE only).")
    parser.add_argument("--beta_cycles", type=int, default=1, help="Number of beta cycles (for BetaVAE only).")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers for dataloaders.")
    parser.add_argument("--mixtures", type=int, default=10, help="Number of mixtures for MOLVAE.")
    parser.add_argument("--synth", action='store_true', help="Use synthetic training data.")
    parser.add_argument("--mix", action='store_true', help="Use 50:50 mix of real and synthetic training data.")
    parser.add_argument("--gauss", action='store_true', help="Use different recon loss to better represent covariance.")
    parser.add_argument("--amp", action='store_true', help="Use auto mixed precision in training.")
    parser.add_argument("--resume", action='store_true', help="Resume from last checkpoint.")
    parser.add_argument("--resume_best", action='store_true', help="Resume from checkpoint with highest SSIM.")
    parser.add_argument("--root", type=str, default='./', help="Root dir to save output directory within.")
    args = parser.parse_args()
    
    if args.synth:
        args.name = 'synth-' + args.name
    elif args.mix:
        args.name = 'mix-' + args.name
    
    os.makedirs(os.path.join(args.root, args.name), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('\nUsing device:', device)
    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    
    if args.model in ['AE', 'RAE']:
        model = Autoencoder(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(16,16,32,64,128,128),
            num_res_blocks=2,
            norm_num_groups=16,
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
            attention_levels=(False,False,False,False,False,False),
            use_convtranspose=False,
            latent_channels=128,
        ).to(device)
    elif args.model in ['SAMBA', 'VAE']:
        model = AutoencoderKL(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(16,16,32,64,128,128),
            num_res_blocks=2,
            norm_num_groups=16,
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
            attention_levels=(False,False,False,False,False,False),
            use_convtranspose=False,
            latent_channels=128,
        ).to(device)
    elif args.model == 'BetaVAE':
        model = AutoencoderKL(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(16,16,32,64,128,128),
            num_res_blocks=2,
            norm_num_groups=16,
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
            attention_levels=(False,False,False,False,False,False),
            use_convtranspose=False,
            latent_channels=128,
        ).to(device)
        betas = list(range(args.beta_init, args.beta_final, args.epochs//args.beta_cycles)) * args.beta_cycles
    elif args.model == 'GaussVAE':
        model = GaussAutoencoderKL(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(16,16,32,64,128,128),
            num_res_blocks=2,
            norm_num_groups=16,
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
            attention_levels=(False,False,False,False,False,False),
            use_convtranspose=False,
            latent_channels=128,
        ).to(device)
    elif args.model == 'MOLVAE':
        model = AutoencoderKL(
            spatial_dims=2,
            in_channels=1,
            out_channels=args.mixtures * (3 * 3 + 1),
            num_channels=(16,16,32,64,128,128),
            num_res_blocks=2,
            norm_num_groups=16,
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
            attention_levels=(False,False,False,False,False,False),
            use_convtranspose=False,
            latent_channels=128,
        ).to(device)
    elif args.model == 'VQVAE':
        model = VQVAE(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(16,16,32,64,128,128),
            num_res_layers=2,
            num_res_channels=(16,16,32,64,128,128),
            embedding_dim=128,
        ).to(device)

    if args.resume or args.resume_best:
        ckpts = glob.glob(os.path.join(args.root, args.name, 'checkpoint.pt' if args.resume else 'checkpoint_best.pt'))
        if len(ckpts) == 0:
            args.resume = False
            print('\nNo checkpoints found. Beginning from epoch #0')
        else:
            checkpoint = torch.load(ckpts[0], map_location=device)
            print('\nResuming from epoch #{} with WandB ID {}'.format(checkpoint['epoch'], checkpoint["wandb"]))
    print()

    wandb.init(
        project="ano-synth",
        entity="ff2023",
        save_code=True,
        name=args.name,
        settings=wandb.Settings(start_method="fork"),
        resume="must" if args.resume else None,
        id=checkpoint["wandb"] if args.resume else None,
    )
    if not args.resume or args.resume_best:
        wandb.config.update(args)
    wandb.watch(model)

    opt = torch.optim.AdamW(model.parameters(), args.lr, fused=torch.cuda.is_available())
    def lambda1(epoch):
        return (1 - epoch / args.epochs) ** 0.9
    lr_scheduler = LambdaLR(opt, lr_lambda=[lambda1])

    class WandBID:
        def __init__(self, wandb_id):
            self.wandb_id = wandb_id

        def state_dict(self):
            return self.wandb_id

    class Epoch:
        def __init__(self, epoch):
            self.epoch = epoch

        def state_dict(self):
            return self.epoch
        
    class Metric:
        def __init__(self, metric):
            self.metric = metric

        def state_dict(self):
            return self.metric
        
    # Try to load most recent weight
    if args.resume or args.resume_best:
        model.load_state_dict(checkpoint["net"])
        opt.load_state_dict(checkpoint["opt"])
        lr_scheduler.load_state_dict(checkpoint["lr"])
        start_epoch = checkpoint["epoch"]
        metric_best = checkpoint["metric"]
    else:
        start_epoch = 0
        metric_best = 0
        
    if args.synth:
        your_train_data, your_eval_data = dataloaders.get_synth_data()
    elif args.mix:
        your_train_data, your_eval_data = dataloaders.get_mix_data()
    else:
        your_train_data, your_eval_data = dataloaders.get_mri_data()
    dataset_output = your_train_data[0]    
    dataset_output = your_eval_data[0]
    train_loader = DataLoader(your_train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(your_eval_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    print()

    train_iter = None
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if args.model == 'AE':
            train_iter = train_utils.train_epoch_ae(train_iter, args.epoch_length, train_loader, opt, model, epoch, device, args.amp)
        elif args.model == 'RAE':
            train_iter = train_utils.train_epoch_rae(train_iter, args.epoch_length, train_loader, opt, model, epoch, device, args.amp)
        elif args.model == 'SAMBA':
            train_iter = train_utils.train_epoch_samba(train_iter, args.epoch_length, train_loader, opt, model, epoch, device, args.amp)
        elif args.model == 'VAE':
            train_iter = train_utils.train_epoch_vae(train_iter, args.epoch_length, train_loader, opt, model, epoch, device, args.amp)
        elif args.model == 'BetaVAE':
            train_iter = train_utils.train_epoch_betavae(train_iter, args.epoch_length, train_loader, opt, model, epoch, device, betas[epoch], args.amp)
        elif args.model == 'GaussVAE':
            train_iter = train_utils.train_epoch_gaussvae(train_iter, args.epoch_length, train_loader, opt, model, epoch, device, args.amp)
        elif args.model == 'MOLVAE':
            train_iter = train_utils.train_epoch_molvae(train_iter, args.epoch_length, train_loader, opt, model, epoch, device, args.amp)
        elif args.model == 'VQVAE':
            train_iter = train_utils.train_epoch_vqvae(train_iter, args.epoch_length, train_loader, opt, model, epoch, device, args.amp)
        lr_scheduler.step()

        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            if args.model == 'AE':
                metric = train_utils.val_epoch_ae(val_loader, model, device, args.amp, epoch)
            elif args.model == 'RAE':
                metric = train_utils.val_epoch_rae(val_loader, model, device, args.amp, epoch, args.recon_weight)
            elif args.model == 'SAMBA':
                metric = train_utils.val_epoch_samba(val_loader, model, device, args.amp, epoch, args.recon_weight)
            elif args.model == 'VAE':
                metric = train_utils.val_epoch_vae(val_loader, model, device, args.amp, epoch, args.recon_weight)
            elif args.model == 'BetaVAE':
                metric = train_utils.val_epoch_vae(val_loader, model, device, args.amp, epoch, args.recon_weight)
            elif args.model == 'GaussVAE':
                metric = train_utils.val_epoch_gaussvae(val_loader, model, device, args.amp, epoch, args.recon_weight)
            elif args.model == 'MOLVAE':
                metric = train_utils.val_epoch_molvae(val_loader, model, device, args.amp, epoch, args.recon_weight)
            elif args.model == 'VQVAE':
                metric = train_utils.val_epoch_vqvae(val_loader, model, device, args.amp, epoch, args.recon_weight)
            if metric > metric_best:
                metric_best = metric
                torch.save(
                    {
                        "net": model.state_dict(),
                        "opt": opt.state_dict(),
                        "lr": lr_scheduler.state_dict(),
                        "wandb": WandBID(wandb.run.id).state_dict(),
                        "epoch": Epoch(epoch).state_dict(),
                        "metric": Metric(metric_best).state_dict()
                    },
                    os.path.join(args.root, args.name,'checkpoint_best.pt'.format(epoch)))
            torch.save(
                {
                    "net": model.state_dict(),
                    "opt": opt.state_dict(),
                    "lr": lr_scheduler.state_dict(),
                    "wandb": WandBID(wandb.run.id).state_dict(),
                    "epoch": Epoch(epoch).state_dict(),
                    "metric": Metric(metric_best).state_dict()
                },
                os.path.join(args.root, args.name,'checkpoint.pt'.format(epoch)))
            