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
    mu = mu.reshape(mu.shape[0], mu.shape[1], -1)
    log_var = log_var.reshape(log_var.shape[0], log_var.shape[1], -1)
    return -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(1).mean(-1)

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

def compute_scales(logits):
    softplus = torch.nn.Softplus(beta=0.6931472)
    scales = torch.maximum(softplus(logits), torch.as_tensor(np.exp(-250.)))
    return scales

def _compute_inv_stdv(logits):
    scales = compute_scales(logits)
    inv_stdv = 1. / scales
    log_scales = torch.log(scales)
    return inv_stdv, log_scales

def scale_pixels(img, bits):
    img = np.floor(img / np.uint8(2 ** (8 - bits))) * 2 ** (8 - bits)
    shift = scale = (2 ** 8 - 1) / 2
    img = (img - shift) / scale
    return img

def mol(logits, targets, bits=32, min_pix_value=0., max_pix_value=1.):
    bit_classes = 2. ** bits - 1.
    min_pix_value = scale_pixels(min_pix_value, bits)
    max_pix_value = scale_pixels(max_pix_value, bits)
    B, C, H, W = targets.size()
    if C == 1:
        targets = torch.cat(3*[targets], dim=1)
        C = targets.size(1)
    assert C == 3
    M = logits.size(1) // (3 * C + 1)
    targets = targets.unsqueeze(2)
    logit_probs = logits[:, :M, :, :]
    l = logits[:, M:, :, :]
    l = l.reshape(B, C, 3 * M, H, W)
    model_means = l[:, :, :M, :, :]
    inv_stdv, log_scales = _compute_inv_stdv(l[:, :, M: 2 * M, :, :])
    model_coeffs = torch.tanh(l[:, :, 2 * M: 3 * M, :, :])
    mean1 = model_means[:, 0:1, :, :, :]
    mean2 = model_means[:, 1:2, :, :, :] + model_coeffs[:, 0:1, :, :, :] * targets[:, 0:1, :, :, :]
    mean3 = model_means[:, 2:3, :, :, :] + model_coeffs[:, 1:2, :, :, :] * targets[:, 0:1, :, :, :] \
                                         + model_coeffs[
                                         :, 2:3, :, :,
                                         :] * targets[
                                         :, 1:2,
                                         :, :,
                                         :]
    means = torch.cat([mean1, mean2, mean3], dim=1)
    centered = targets - means
    plus_in = inv_stdv * (centered + 1. / bit_classes)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered - 1. / bit_classes)
    cdf_min = torch.sigmoid(min_in)
    log_cdf_plus = plus_in - torch.nn.functional.softplus(plus_in)
    log_one_minus_cdf_min = -torch.nn.functional.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min
    mid_in = inv_stdv * centered
    log_pdf_mid = mid_in - log_scales - 2. * torch.nn.functional.softplus(mid_in)
    broadcast_targets = torch.broadcast_to(targets, size=[B, C, M, H, W])
    log_probs = torch.where(broadcast_targets == min_pix_value, log_cdf_plus,
                            torch.where(broadcast_targets == max_pix_value, log_one_minus_cdf_min,
                                        torch.where(cdf_delta > 1e-5,
                                                    torch.log(torch.clamp(cdf_delta, min=1e-12)),
                                                    log_pdf_mid - np.log(bit_classes / 2))))
    log_probs = torch.sum(log_probs, dim=1) + torch.nn.functional.log_softmax(logit_probs, dim=1)
    negative_log_probs = -torch.logsumexp(log_probs, dim=1)
    mean_axis = list(range(1, len(negative_log_probs.size())))
    per_example_loss = torch.sum(negative_log_probs, dim=mean_axis)
    avg_per_example_loss = per_example_loss / (np.prod([negative_log_probs.size()[i] for i in mean_axis]) * C)
    assert len(per_example_loss.size()) == len(avg_per_example_loss.size()) == 1
    scalar = B * H * W * C
    loss = torch.sum(per_example_loss) / scalar
    avg_loss = torch.sum(avg_per_example_loss) / (B * np.log(2))
    return loss, avg_loss, model_means, log_scales

def one_hot(indices, depth, dim):
    indices = indices.unsqueeze(dim)
    size = list(indices.size())
    size[dim] = depth
    y_onehot = torch.zeros(size, device=indices.device)
    y_onehot.zero_()
    y_onehot.scatter_(dim, indices, 1)
    return y_onehot

def sample_from_mol(logits, targets, bits=32, min_pix_value=0., max_pix_value=1., temp=1.):
    bit_classes = 2. ** bits - 1.
    min_pix_value = scale_pixels(min_pix_value, bits)
    max_pix_value = scale_pixels(max_pix_value, bits)
    B, C, H, W = targets.size()
    if C == 1:
        targets = torch.cat(3*[targets], dim=1)
        C = targets.size(1)
    assert C == 3
    M = logits.size(1) / (3 * C + 1)
    logit_probs = logits[:, :M, :, :]
    l = logits[:, M:, :, :]
    l = l.reshape(B, C, 3 * M, H, W)
    model_means = l[:, :, :M, :, :]
    scales = compute_scales(l[:, :, M: 2 * M, :, :])
    model_coeffs = torch.tanh(l[:, :, 2 * M: 3 * M, :, :])
    gumbel_noise = -torch.log(-torch.log(torch.Tensor(logit_probs.size(), 
                            dtype=logits.dtype, device=logits.device).uniform_(1e-5, 1. - 1e-5)))
    logit_probs = logit_probs / temp + gumbel_noise
    lambda_ = one_hot(torch.argmax(logit_probs, dim=1), logit_probs.size(1), dim=1)
    lambda_ = lambda_.unsqueeze(1)
    means = torch.sum(model_means * lambda_, dim=2)
    scales = torch.sum(scales * lambda_, dim=2)
    coeffs = torch.sum(model_coeffs * lambda_, dim=2)
    u = torch.Tensor(means.size(),dtype=logits.dtype,device=logits.device).uniform_(1e-5, 1. - 1e-5)
    x = means + scales * temp * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(x[:, 0:1, :, :], min=min_pix_value, max=max_pix_value)
    x1 = torch.clamp(x[:, 1:2, :, :] + coeffs[:, 0:1, :, :] * x0, min=min_pix_value, max=max_pix_value)
    x2 = torch.clamp(x[:, 2:3, :, :] + coeffs[:, 1:2, :, :] * x0 + coeffs[:, 2:3, :, :] * x1, 
                     min=min_pix_value, max=max_pix_value)
    x = torch.cat([x0, x1, x2], dim=1)
    return x


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
            reconstruction = model(images)
            reconstruction = torch.sigmoid(reconstruction)
            recons_loss = l2(reconstruction, images)
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
            reconstruction, z_mu, z_sigma = model(images)
            reconstruction = torch.sigmoid(reconstruction)
            recons_loss = l2(reconstruction, images)
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
            reconstruction, z_mu, z_sigma = model(images)
            reconstruction = torch.sigmoid(reconstruction)
            recons_loss = l2(reconstruction, images)
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
            reconstruction = torch.sigmoid(reconstruction)
            recons_loss = gauss_l2(reconstruction, recon_sigma, images)
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

def train_epoch_molvae(train_iter, epoch_length, train_loader, opt, model, epoch, device, amp):
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
            reconstruction, z_mu, z_sigma = model(images)
            recons_loss, avg_loss, model_means, log_scales = mol(reconstruction, images)
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
            reconstruction, quantization_loss = model(images)
            reconstruction = torch.sigmoid(reconstruction)
            recons_loss = l2(reconstruction, images)
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
                reconstruction = torch.sigmoid(reconstruction)
                recons_loss = l2(reconstruction, images)
            if val_step < 16:
                inputs.append(images[0].cpu().float())
                recons.append(reconstruction[0].cpu().float())
            elif val_step == 16:
                grid_inputs = make_grid(inputs, nrow=4, padding=5, normalize=True, scale_each=True)
                grid_recons = make_grid(recons, nrow=4, padding=5, normalize=True, scale_each=True)
                wandb.log({"val/examples": [wandb.Image(grid_inputs[0].numpy(), caption="Real images"),
                                            wandb.Image(grid_recons[0].numpy(), caption="Reconstructions")]})
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
                reconstruction = torch.sigmoid(reconstruction)
                recons_loss = l2(reconstruction, images)
                kl_loss = kld(z_mu, 2*(z_sigma).log())
            if val_step < 16:
                inputs.append(images[0].cpu().float())
                recons.append(reconstruction[0].cpu().float())
            elif val_step == 16:
                grid_inputs = make_grid(inputs, nrow=4, padding=5, normalize=True, scale_each=True)
                grid_recons = make_grid(recons, nrow=4, padding=5, normalize=True, scale_each=True)
                wandb.log({"val/examples": [wandb.Image(grid_inputs[0].numpy(), caption="Real images"),
                                            wandb.Image(grid_recons[0].numpy(), caption="Reconstructions")]})
            val_loss += recons_loss.sum().item()
            kld_loss += kl_loss.sum().item()
            progress_bar.set_postfix({"recons_loss": val_loss / (val_step + 1), "kld_loss": kld_loss / (val_step + 1)})
    wandb.log({"val/recon_loss": val_loss / (val_step + 1)})
    wandb.log({"val/kld_loss": kld_loss / (val_step + 1)})

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
                reconstruction = torch.sigmoid(reconstruction)
                recons_loss = gauss_l2(reconstruction, recon_sigma, images)
                kl_loss = kld(z_mu, 2*(z_sigma).log())
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
            val_loss += recons_loss.sum().item()
            kld_loss += kl_loss.sum().item()
            progress_bar.set_postfix({"recons_loss": val_loss / (val_step + 1), "kld_loss": kld_loss / (val_step + 1)})
    wandb.log({"val/recon_loss": val_loss / (val_step + 1)})
    wandb.log({"val/kld_loss": kld_loss / (val_step + 1)})

def val_epoch_molvae(val_loader, model, device, amp, epoch):
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
                reconstruction, z_mu, z_sigma = model(images)
                reconstruction = sample_from_mol(reconstruction, images)
                recons_loss = mol(reconstruction, images)
                kl_loss = kld(z_mu, 2*(z_sigma).log())
            if val_step < 16:
                inputs.append(images[0].cpu().float().mean(0, keepdim=True))
                recons.append(reconstruction[0].cpu().float().mean(0, keepdim=True))
            elif val_step == 16:
                grid_inputs = make_grid(inputs, nrow=4, padding=5, normalize=True, scale_each=True)
                grid_recons = make_grid(recons, nrow=4, padding=5, normalize=True, scale_each=True)
                wandb.log({"val/examples": [wandb.Image(grid_inputs[0].numpy(), caption="Real images"),
                                            wandb.Image(grid_recons[0].numpy(), caption="Reconstructions")]})
            val_loss += recons_loss.sum().item()
            kld_loss += kl_loss.sum().item()
            progress_bar.set_postfix({"recons_loss": val_loss / (val_step + 1), "kld_loss": kld_loss / (val_step + 1)})
    wandb.log({"val/recon_loss": val_loss / (val_step + 1)})
    wandb.log({"val/kld_loss": kld_loss / (val_step + 1)})

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
                reconstruction = torch.sigmoid(reconstruction)
                recons_loss = l2(reconstruction, images)
            if val_step < 16:
                inputs.append(images[0].cpu().float())
                recons.append(reconstruction[0].cpu().float())
            elif val_step == 16:
                grid_inputs = make_grid(inputs, nrow=4, padding=5, normalize=True, scale_each=True)
                grid_recons = make_grid(recons, nrow=4, padding=5, normalize=True, scale_each=True)
                wandb.log({"val/examples": [wandb.Image(grid_inputs[0].numpy(), caption="Real images"),
                                            wandb.Image(grid_recons[0].numpy(), caption="Reconstructions")]})
            val_loss += recons_loss.sum().item()
            val_quant += quantization_loss.sum().item()
            progress_bar.set_postfix({"recons_loss": val_loss / (val_step + 1), "quantization_loss": val_quant / (val_step + 1)})
    wandb.log({"val/recon_loss": val_loss / (val_step + 1)})
    wandb.log({"val/quant_loss": val_quant / (val_step + 1)})
