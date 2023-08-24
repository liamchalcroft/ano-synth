from tqdm import tqdm
import torch
import wandb
import numpy as np


def kld(mu, log_var):
    return -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(-1)

def l2(recon_x, x):
    return torch.square(x - recon_x).sum(-1)

def gauss_l2(x_mu, x_sigma, x):
    squared_difference = torch.square(x - x_mu)
    x_var = x_sigma ** 2
    x_log_var = x_var.log()
    squared_diff_normed = torch.true_divide(squared_difference, x_var)
    return 0.5 * (np.log(2 * np.pi) + x_log_var + squared_diff_normed).sum(-1)


### training stuff ###


def train_epoch_ae(train_loader, opt, model, epoch, device):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["image"].to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            reconstruction = model(images)
            recons_loss = l2(reconstruction.float(), images.float()).sum()
        loss = recons_loss
        loss.backward()
        opt.step()
        epoch_loss += recons_loss.item()
        wandb.log({"train/recon_loss": recons_loss.item()})
        progress_bar.set_postfix({"recons_loss": epoch_loss / (step + 1)})

def train_epoch_vae(train_loader, opt, model, epoch, device):
    model.train()
    epoch_loss = 0
    kld_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["image"].to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            reconstruction, z_mu, z_sigma = model(images)
            recons_loss = l2(reconstruction.float(), images.float()).sum()
            kl_loss = kld(z_mu, 2*(z_sigma).log()).sum()
        loss = recons_loss + kl_loss
        loss.backward()
        opt.step()
        epoch_loss += recons_loss.item()
        kld_loss += kl_loss.item()
        wandb.log({"train/recon_loss": recons_loss.item()})
        wandb.log({"train/kld_loss": kld_loss.item()})
        progress_bar.set_postfix({"recons_loss": epoch_loss / (step + 1), "kld_loss": kld_loss / (step + 1)})

def train_epoch_betavae(train_loader, opt, model, epoch, device, beta):
    model.train()
    epoch_loss = 0
    kld_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["image"].to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            reconstruction, z_mu, z_sigma = model(images)
            recons_loss = l2(reconstruction.float(), images.float()).sum()
            kl_loss = kld(z_mu, 2*(z_sigma).log()).sum()
        loss = recons_loss + beta * kl_loss
        loss.backward()
        opt.step()
        epoch_loss += recons_loss.item()
        kld_loss += kl_loss.item()
        wandb.log({"train/recon_loss": recons_loss.item()})
        wandb.log({"train/kld_loss": kld_loss.item()})
        progress_bar.set_postfix({"recons_loss": epoch_loss / (step + 1), "kld_loss": kld_loss / (step + 1)})

def train_epoch_gaussvae(train_loader, opt, model, epoch, device):
    model.train()
    epoch_loss = 0
    kld_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["image"].to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            reconstruction, recon_sigma, z_mu, z_sigma = model(images)
            recons_loss = gauss_l2(reconstruction.float(), recon_sigma.float(), images.float()).sum()
            kl_loss = kld(z_mu, 2*(z_sigma).log()).sum()
        loss = recons_loss + kl_loss
        loss.backward()
        opt.step()
        epoch_loss += recons_loss.item()
        kld_loss += kl_loss.item()
        wandb.log({"train/recon_loss": recons_loss.item()})
        wandb.log({"train/kld_loss": kld_loss.item()})
        progress_bar.set_postfix({"recons_loss": epoch_loss / (step + 1), "kld_loss": kld_loss / (step + 1)})

def train_epoch_vqvae(train_loader, opt, model, epoch, device):
    model.train()
    epoch_loss = 0
    quant_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["image"].to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            reconstruction, quantization_loss = model(images)
            recons_loss = l2(reconstruction.float(), images.float()).sum()
        loss = recons_loss + quantization_loss
        loss.backward()
        opt.step()
        epoch_loss += recons_loss.item()
        quant_loss += quantization_loss.item()
        wandb.log({"train/recon_loss": recons_loss.item()})
        wandb.log({"train/quant_loss": quantization_loss.item()})
        progress_bar.set_postfix({"recons_loss": epoch_loss / (step + 1), "quantization_loss": quant_loss / (step + 1)})


### validation stuff ###


def val_epoch_ae(val_loader, model, device):
    val_loss = 0
    with torch.no_grad():
        for val_step, batch in enumerate(val_loader, start=1):
            images = batch["image"].to(device)
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                reconstruction = model(images)
            if val_step == 1:
                wandb.log({"input": wandb.Image(images[0,0,...,images.shape[-1]//2].cpu().numpy()),
                            "recon": wandb.Image(reconstruction[0,0,...,images.shape[-1]//2].cpu().numpy())})
            recons_loss = l2(reconstruction.float(), images.float())
            val_loss += recons_loss.item()
    wandb.log({"val/recon_loss": val_loss / val_step})

def val_epoch_vae(val_loader, model, device):
    val_loss = 0
    kld_loss = 0
    with torch.no_grad():
        for val_step, batch in enumerate(val_loader, start=1):
            images = batch["image"].to(device)
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                reconstruction, z_mu, z_sigma = model(images)
            if val_step == 1:
                wandb.log({"input": wandb.Image(images[0,0,...,images.shape[-1]//2].cpu().numpy()),
                            "recon": wandb.Image(reconstruction[0,0,...,images.shape[-1]//2].cpu().numpy())})
            recons_loss = l2(reconstruction.float(), images.float()).sum()
            kl_loss = kld(z_mu, 2*(z_sigma).log()).sum()
            val_loss += recons_loss.item()
            kld_loss += kl_loss.item()
    wandb.log({"val/recon_loss": val_loss / val_step})
    wandb.log({"val/kld_loss": val_loss / val_step})

def val_epoch_vae(val_loader, model, device):
    val_loss = 0
    kld_loss = 0
    with torch.no_grad():
        for val_step, batch in enumerate(val_loader, start=1):
            images = batch["image"].to(device)
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                reconstruction, recon_sigma, z_mu, z_sigma = model(images)
            if val_step == 1:
                wandb.log({"input": wandb.Image(images[0,0,...,images.shape[-1]//2].cpu().numpy()),
                            "recon": wandb.Image(reconstruction[0,0,...,images.shape[-1]//2].cpu().numpy())})
            recons_loss = gauss_l2(reconstruction.float(), recon_sigma.float(), images.float()).sum()
            kl_loss = kld(z_mu, 2*(z_sigma).log()).sum()
            val_loss += recons_loss.item()
            kld_loss += kl_loss.item()
    wandb.log({"val/recon_loss": val_loss / val_step})
    wandb.log({"val/kld_loss": val_loss / val_step})

def val_epoch_vqvae(val_loader, model, device):
    val_loss = 0
    val_quant = 0
    with torch.no_grad():
        for val_step, batch in enumerate(val_loader, start=1):
            images = batch["image"].to(device)
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                reconstruction, quantization_loss = model(images)
            if val_step == 1:
                wandb.log({"input": wandb.Image(images[0,0,...,images.shape[-1]//2].cpu().numpy()),
                            "recon": wandb.Image(reconstruction[0,0,...,images.shape[-1]//2].cpu().numpy())})
            recons_loss = l2(reconstruction.float(), images.float())
            val_loss += recons_loss.item()
            val_quant += quantization_loss.item()
    wandb.log({"val/recon_loss": val_loss / val_step})
    wandb.log({"val/quant_loss": val_quant / val_step})