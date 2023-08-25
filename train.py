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
    parser.add_argument("--model", type=str, help="Model to use. Full list of options available in Pythae docs.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs for training.")
    parser.add_argument("--epoch_length", type=int, default=100, help="Number of iterations per epoch.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--val_interval", type=int, default=2, help="Validation interval.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--beta_init", type=int, default=0, help="Initial beta (for BetaVAE only).")
    parser.add_argument("--beta_final", type=int, default=20, help="Final beta (for BetaVAE only).")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers for dataloaders.")
    parser.add_argument("--synth", action='store_true', help="Use synthetic training data.")
    parser.add_argument("--gauss", action='store_true', help="Use different recon loss to better represent covariance.")
    parser.add_argument("--amp", action='store_true', help="Use auto mixed precision in training.")
    parser.add_argument("--resume", action='store_true', help="Find most recent run in output dir and resume from last checkpoint.")
    parser.add_argument("--root", type=str, default='./', help="Root dir to save output directory within.")
    args = parser.parse_args()
    
    
    os.makedirs(os.path.join(args.root, args.name), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('\nUsing device:', device)
    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    
    if args.model == 'AE':
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
    elif args.model == 'VAE':
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
        betas = list(range(args.beta_init, args.beta_final, args.epochs))
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

    if args.resume:
        ckpts = glob.glob(os.path.join(args.root, args.name, 'checkpoint_epoch=*.pt'))
        if len(ckpts) == 0:
            args.resume = False
            print('\nNo checkpoints found. Beginning from epoch #0')
        else:
            ckpts = [{'path': p, 'epoch': int(p.split('_')[-1][:-3].split('=')[-1])} for p in ckpts]
            ckpt = sorted(ckpts, key=lambda d: d['epoch'])[-1]
            print('\nResuming from epoch #{} with WandB ID {}'.format(ckpt['epoch'],torch.load(ckpt['path'], map_location=device)["wandb"]))
    print()

    wandb.init(
        project="ano-synth",
        entity="ff2023",
        save_code=True,
        name=args.name,
        settings=wandb.Settings(start_method="fork"),
        resume="must" if args.resume else None,
        id=torch.load(ckpt['path'], map_location=device)["wandb"] if args.resume else None,
    )
    if not args.resume:
        wandb.config.update(args)
    wandb.watch(model)

    opt = torch.optim.Adam(model.parameters(), args.lr)
    def lambda1(epoch):
        return (1 - epoch / args.epochs) ** 0.9
    lr_scheduler = LambdaLR(opt, lr_lambda=[lambda1])

    class WandBID:
        def __init__(self, wandb_id):
            self.wandb_id = wandb_id

        def state_dict(self):
            return self.wandb_id
        
    # Try to load most recent weight
    if args.resume:
        checkpoint = torch.load(ckpt['path'], map_location=device) 
        model.load_state_dict(checkpoint["net"])
        opt.load_state_dict(checkpoint["opt"])
        lr_scheduler.load_state_dict(checkpoint["lr"])
        start_epoch = int(ckpt['epoch'])
    else:
        start_epoch = 0
        
    if args.synth:
        your_train_data, your_eval_data = dataloaders.get_synth_data(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    else:
        your_train_data, your_eval_data = dataloaders.get_mri_data(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
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
        elif args.model == 'VAE':
            train_iter = train_utils.train_epoch_vae(train_iter, args.epoch_length, train_loader, opt, model, epoch, device, args.amp)
        elif args.model == 'BetaVAE':
            train_iter = train_utils.train_epoch_betavae(train_iter, args.epoch_length, train_loader, opt, model, epoch, device, betas[epoch], args.amp)
        elif args.model == 'GaussVAE':
            train_iter = train_utils.train_epoch_gaussvae(train_iter, args.epoch_length, train_loader, opt, model, epoch, device, args.amp)
        elif args.model == 'VQVAE':
            train_iter = train_utils.train_epoch_vqvae(train_iter, args.epoch_length, train_loader, opt, model, epoch, device, args.amp)
        lr_scheduler.step()

        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            if args.model == 'AE':
                train_utils.val_epoch_ae(val_loader, model, device, args.amp, epoch)
            elif args.model == 'VAE':
                train_utils.val_epoch_vae(val_loader, model, device, args.amp, epoch)
            elif args.model == 'BetaVAE':
                train_utils.val_epoch_vae(val_loader, model, device, args.amp, epoch)
            elif args.model == 'GaussVAE':
                train_utils.val_epoch_gaussvae(val_loader, model, device, args.amp, epoch)
            elif args.model == 'VQVAE':
                train_utils.val_epoch_vqvae(val_loader, model, device, args.amp, epoch)
            torch.save(
                {
                    "net": model.state_dict(),
                    "opt": opt.state_dict(),
                    "lr": lr_scheduler.state_dict(),
                    "wandb": WandBID(wandb.run.id).state_dict()
                },
                os.path.join(args.root, args.name,'checkpoint_epoch={}.pt'.format(epoch)))
            