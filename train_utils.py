from tqdm import tqdm
import torch
import wandb
import math
import numpy as np
from contextlib import nullcontext
from torchvision.utils import make_grid
from torch.cuda.amp import GradScaler


### loss functions ###


def kld(mu, log_var):
    mu = mu.reshape(mu.shape[0], -1)
    log_var = log_var.reshape(log_var.shape[0], -1)
    return -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(-1)

def l2(recon_x, x):
    x = x.reshape(x.shape[0], -1)
    recon_x = recon_x.reshape(recon_x.shape[0], -1)
    return torch.square(x - recon_x).mean(-1)

def gauss_l2(x_mu, x_sigma, x):
    x = x.reshape(x.shape[0], -1)
    x_mu = x_mu.reshape(x_mu.shape[0], -1)
    x_sigma = x_sigma.reshape(x_sigma.shape[0], -1)
    squared_difference = torch.square(x - x_mu)
    x_var = x_sigma ** 2
    x_log_var = x_var.log()
    squared_diff_normed = torch.true_divide(squared_difference, x_var)
    return 0.5 * (torch.log(2 * torch.tensor(math.pi, dtype=x.dtype, device=x.device)) + x_log_var + squared_diff_normed).mean(-1)


### training stuff ###


def train_epoch_ae(train_iter, epoch_length, train_loader, opt, model, epoch, device, amp):
    model.train()
    epoch_loss = 0
    if amp:
        ctx = torch.autocast("cuda" if torch.cuda.is_available() else "cpu")
        scaler = GradScaler()
    else:
        ctx = nullcontext()
    progress_bar = tqdm(range(epoch_length), total=epoch_length, ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    if train_iter is None:
        train_iter = iter(train_loader)
    for step in progress_bar:
        try:
            batch = next(train_iter)
        except:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        images = batch["image"].to(device)
        opt.zero_grad(set_to_none=True)
        with ctx:
            reconstruction = model(images)
            recons_loss = l2(reconstruction.float(), images.float())
            loss = recons_loss.sum()
        if amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            opt.step()
        epoch_loss += recons_loss.sum().item()
        wandb.log({"train/recon_loss": recons_loss.sum().item()})
        progress_bar.set_postfix({"recons_loss": epoch_loss / (step + 1)})
    return train_iter

def train_epoch_vae(train_iter, epoch_length, train_loader, opt, model, epoch, device, amp):
    model.train()
    epoch_loss = 0
    kld_loss = 0
    if amp:
        ctx = torch.autocast("cuda" if torch.cuda.is_available() else "cpu")
        scaler = GradScaler()
    else:
        ctx = nullcontext()
    progress_bar = tqdm(range(epoch_length), total=epoch_length, ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    if train_iter is None:
        train_iter = iter(train_loader)
    for step in progress_bar:
        try:
            batch = next(train_iter)
        except:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        images = batch["image"].to(device)
        opt.zero_grad(set_to_none=True)
        with ctx:
            reconstruction, z_mu, z_sigma = model(images)
            recons_loss = l2(reconstruction.float(), images.float())
            kl_loss = kld(z_mu, 2*(z_sigma).log())
            loss = (recons_loss + kl_loss).sum()
        if amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            opt.step()
        epoch_loss += recons_loss.sum().item()
        kld_loss += kl_loss.sum().item()
        wandb.log({"train/recon_loss": recons_loss.sum().item()})
        wandb.log({"train/kld_loss": kl_loss.sum().item()})
        progress_bar.set_postfix({"recons_loss": epoch_loss / (step + 1), "kld_loss": kld_loss / (step + 1)})
    return train_iter

def train_epoch_betavae(train_iter, epoch_length, train_loader, opt, model, epoch, device, beta, amp):
    model.train()
    epoch_loss = 0
    kld_loss = 0
    if amp:
        ctx = torch.autocast("cuda" if torch.cuda.is_available() else "cpu")
        scaler = GradScaler()
    else:
        ctx = nullcontext()
    progress_bar = tqdm(range(epoch_length), total=epoch_length, ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    if train_iter is None:
        train_iter = iter(train_loader)
    for step in progress_bar:
        try:
            batch = next(train_iter)
        except:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        images = batch["image"].to(device)
        opt.zero_grad(set_to_none=True)
        with ctx:
            reconstruction, z_mu, z_sigma = model(images)
            recons_loss = l2(reconstruction.float(), images.float())
            kl_loss = kld(z_mu, 2*(z_sigma).log())
            loss = (recons_loss + beta * kl_loss).sum()
        if amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            opt.step()
        epoch_loss += recons_loss.sum().item()
        kld_loss += kl_loss.sum().item()
        wandb.log({"train/recon_loss": recons_loss.sum().item()})
        wandb.log({"train/kld_loss": kl_loss.sum().item()})
        progress_bar.set_postfix({"recons_loss": epoch_loss / (step + 1), "kld_loss": kld_loss / (step + 1)})
    return train_iter

def train_epoch_gaussvae(train_iter, epoch_length, train_loader, opt, model, epoch, device, amp):
    model.train()
    epoch_loss = 0
    kld_loss = 0
    if amp:
        ctx = torch.autocast("cuda" if torch.cuda.is_available() else "cpu")
        scaler = GradScaler()
    else:
        ctx = nullcontext()
    progress_bar = tqdm(range(epoch_length), total=epoch_length, ncols=110)
    progress_bar.set_description(f"[Training] Epoch {epoch}")
    if train_iter is None:
        train_iter = iter(train_loader)
    for step in progress_bar:
        try:
            batch = next(train_iter)
        except:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        images = batch["image"].to(device)
        opt.zero_grad(set_to_none=True)
        with ctx:
            reconstruction, recon_sigma, z_mu, z_sigma = model(images)
            recons_loss = gauss_l2(reconstruction.float(), recon_sigma.float(), images.float())
            kl_loss = kld(z_mu, 2*z_sigma.log())
            loss = (recons_loss + kl_loss).sum()
        if amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            opt.step()
        epoch_loss += recons_loss.sum().item()
        kld_loss += kl_loss.sum().item()
        wandb.log({"train/recon_loss": recons_loss.sum().item()})
        wandb.log({"train/kld_loss": kl_loss.sum().item()})
        progress_bar.set_postfix({"recons_loss": epoch_loss / (step + 1), "kld_loss": kld_loss / (step + 1)})
    return train_iter

def train_epoch_vqvae(train_iter, epoch_length, train_loader, opt, model, epoch, device, amp):
    model.train()
    epoch_loss = 0
    quant_loss = 0
    if amp:
        ctx = torch.autocast("cuda" if torch.cuda.is_available() else "cpu")
        scaler = GradScaler()
    else:
        ctx = nullcontext()
    progress_bar = tqdm(range(epoch_length), total=epoch_length, ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    if train_iter is None:
        train_iter = iter(train_loader)
    for step in progress_bar:
        try:
            batch = next(train_iter)
        except:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        images = batch["image"].to(device)
        opt.zero_grad(set_to_none=True)
        with ctx:
            reconstruction, quantization_loss = model(images)
            recons_loss = l2(reconstruction.float(), images.float())
            loss = recons_loss.sum() + quantization_loss
        if amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            opt.step()
        epoch_loss += recons_loss.sum().item()
        quant_loss += quantization_loss.item()
        wandb.log({"train/recon_loss": recons_loss.sum().item()})
        wandb.log({"train/quant_loss": quantization_loss.item()})
        progress_bar.set_postfix({"recons_loss": epoch_loss / (step + 1), "quantization_loss": quant_loss / (step + 1)})
    return train_iter


### validation stuff ###


def val_epoch_ae(val_loader, model, device, amp, epoch):
    val_loss = 0
    inputs = []
    recons = []
    ctx = torch.autocast("cuda" if torch.cuda.is_available() else "cpu") if amp else nullcontext()
    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=110)
    progress_bar.set_description(f"[Validation] Epoch {epoch}")
    with torch.no_grad():
        for val_step, batch in progress_bar:
            images = batch["image"].to(device)
            with ctx:
                reconstruction = model(images)
            if val_step < 16:
                inputs.append(images[0].cpu().float())
                recons.append(reconstruction[0].cpu().float())
            elif val_step == 16:
                grid_inputs = make_grid(inputs, nrow=4, padding=5, normalize=True, scale_each=True)
                grid_recons = make_grid(recons, nrow=4, padding=5, normalize=True, scale_each=True)
                wandb.log({"val/examples": [wandb.Image(grid_inputs[0].numpy(), caption="Real images"),
                                            wandb.Image(grid_recons[0].numpy(), caption="Reconstructions")]})
            recons_loss = l2(reconstruction.float(), images.float())
            val_loss += recons_loss.sum().item()
    progress_bar.set_postfix({"recons_loss": val_loss / (val_step + 1)})
    wandb.log({"val/recon_loss": val_loss / (val_step + 1)})

def val_epoch_vae(val_loader, model, device, amp, epoch):
    val_loss = 0
    kld_loss = 0
    inputs = []
    recons = []
    ctx = torch.autocast("cuda" if torch.cuda.is_available() else "cpu") if amp else nullcontext()
    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=110)
    progress_bar.set_description(f"[Validation] Epoch {epoch}")
    with torch.no_grad():
        for val_step, batch in progress_bar:
            images = batch["image"].to(device)
            with ctx:
                reconstruction, z_mu, z_sigma = model(images)
            if val_step < 16:
                inputs.append(images[0].cpu().float())
                recons.append(reconstruction[0].cpu().float())
            elif val_step == 16:
                grid_inputs = make_grid(inputs, nrow=4, padding=5, normalize=True, scale_each=True)
                grid_recons = make_grid(recons, nrow=4, padding=5, normalize=True, scale_each=True)
                wandb.log({"val/examples": [wandb.Image(grid_inputs[0].numpy(), caption="Real images"),
                                            wandb.Image(grid_recons[0].numpy(), caption="Reconstructions")]})
            recons_loss = l2(reconstruction.float(), images.float())
            kl_loss = kld(z_mu, 2*(z_sigma).log())
            val_loss += recons_loss.sum().item()
            kld_loss += kl_loss.sum().item()
    progress_bar.set_postfix({"recons_loss": val_loss / (val_step + 1), "kld_loss": val_loss / (val_step + 1)})
    wandb.log({"val/recon_loss": val_loss / (val_step + 1)})
    wandb.log({"val/kld_loss": val_loss / (val_step + 1)})

def val_epoch_gaussvae(val_loader, model, device, amp, epoch):
    val_loss = 0
    kld_loss = 0
    inputs = []
    recons = []
    sigmas = []
    ctx = torch.autocast("cuda" if torch.cuda.is_available() else "cpu") if amp else nullcontext()
    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=110)
    progress_bar.set_description(f"[Validation] Epoch {epoch}")
    with torch.no_grad():
        for val_step, batch in progress_bar:
            images = batch["image"].to(device)
            with ctx:
                reconstruction, recon_sigma, z_mu, z_sigma = model(images)
            if val_step < 16:
                inputs.append(images[0].cpu().float())
                recons.append(reconstruction[0].cpu().float())
                sigmas.append(recon_sigma[0].cpu().float())
            elif val_step == 16:
                grid_inputs = make_grid(inputs, nrow=4, padding=5, normalize=True, scale_each=True)
                grid_recons = make_grid(recons, nrow=4, padding=5, normalize=True, scale_each=True)
                grid_sigmas = make_grid(sigmas, nrow=4, padding=5, normalize=True, scale_each=False)
                wandb.log({"val/examples": [wandb.Image(grid_inputs[0].numpy(), caption="Real images"),
                                            wandb.Image(grid_recons[0].numpy(), caption="Reconstructions"),
                                            wandb.Image(grid_sigmas[0].numpy(), caption="Uncertanties")]})
            recons_loss = gauss_l2(reconstruction.float(), recon_sigma.float(), images.float())
            kl_loss = kld(z_mu, 2*(z_sigma).log())
            val_loss += recons_loss.sum().item()
            kld_loss += kl_loss.sum().item()
    progress_bar.set_postfix({"recons_loss": val_loss / (val_step + 1), "kld_loss": val_loss / (val_step + 1)})
    wandb.log({"val/recon_loss": val_loss / (val_step + 1)})
    wandb.log({"val/kld_loss": val_loss / (val_step + 1)})

def val_epoch_vqvae(val_loader, model, device, amp, epoch):
    val_loss = 0
    val_quant = 0
    inputs = []
    recons = []
    ctx = torch.autocast("cuda" if torch.cuda.is_available() else "cpu") if amp else nullcontext()
    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=110)
    progress_bar.set_description(f"[Validation] Epoch {epoch}")
    with torch.no_grad():
        for val_step, batch in progress_bar:
            images = batch["image"].to(device)
            with ctx:
                reconstruction, quantization_loss = model(images)
            if val_step < 16:
                inputs.append(images[0].cpu().float())
                recons.append(reconstruction[0].cpu().float())
            elif val_step == 16:
                grid_inputs = make_grid(inputs, nrow=4, padding=5, normalize=True, scale_each=True)
                grid_recons = make_grid(recons, nrow=4, padding=5, normalize=True, scale_each=True)
                wandb.log({"val/examples": [wandb.Image(grid_inputs[0].numpy(), caption="Real images"),
                                            wandb.Image(grid_recons[0].numpy(), caption="Reconstructions")]})
            recons_loss = l2(reconstruction.float(), images.float())
            val_loss += recons_loss.sum().item()
            val_quant += quantization_loss.sum().item()
    progress_bar.set_postfix({"recons_loss": val_loss / (val_step + 1), "quantization_loss": val_quant / (val_step + 1)})
    wandb.log({"val/recon_loss": val_loss / (val_step + 1)})
    wandb.log({"val/quant_loss": val_quant / (val_step + 1)})
