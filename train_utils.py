from tqdm import tqdm
import torch
import wandb
import math
import numpy as np
from contextlib import nullcontext
from torchvision.utils import make_grid
from torch.cuda.amp import GradScaler
import random
from monai.utils import set_determinism
import os


### Reproducibility ### 
def set_global_seed(seed = 42):

    set_determinism(seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


### loss functions ###
def kld(mu, log_var):
    mu = mu.reshape(mu.shape[0], mu.shape[1], -1)
    log_var = log_var.reshape(log_var.shape[0], log_var.shape[1], -1)
    #return -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(1).mean(-1)
    return torch.mean((-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())),  dim=(1, 2)) # new kld_batch_mean

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

def _fspecial_gauss_1d(size, sigma):
    # https://github.com/VainF/pytorch-msssim
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)

def gaussian_filter(input, win):
    # https://github.com/VainF/pytorch-msssim
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = torch.nn.functional.conv2d
    elif len(input.shape) == 5:
        conv = torch.nn.functional.conv3d
    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
    return out

def ssim(X, Y, data_range=1., K=(0.01, 0.03)):
    # https://github.com/VainF/pytorch-msssim
    K1, K2 = K
    win = _fspecial_gauss_1d(11, 1.5)
    win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))
    compensation = 1.0
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    win = win.to(X.device, dtype=X.dtype)
    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)
    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map
    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    return torch.relu(ssim_per_channel).mean()

def decoder_jacobian(model, x, z, loss):
    # https://github.com/rpatrik96/vae-sam
    if model.training is False:
        z.requires_grad = True
    with torch.set_grad_enabled(True):
        grad = torch.autograd.grad(
            outputs=loss(model.decoder(z), x),
            inputs=z,
            create_graph=True,
        )[0]
    if model.training is False:
        z.requires_grad = False
    return grad

def rae_penalty(model, x, z):
    # https://github.com/rpatrik96/vae-sam
    if model.training is False:
        z.requires_grad = True
    with torch.set_grad_enabled(True):
        grad = torch.autograd.grad(
            outputs=l2(torch.sigmoid(model.decode(z)), x).sum(),
            inputs=z,
            create_graph=True,
        )[0]
    if model.training is False:
        z.requires_grad = False
    return grad.norm(p=2.)

def samba_l2(model, x, z_mu, z_std):
    # https://github.com/rpatrik96/vae-sam
    if model.training is False:
        z_mu.requires_grad = True
    with torch.set_grad_enabled(True):
        grad = torch.autograd.grad(
            outputs=l2(torch.sigmoid(model.decode(z_mu)), x).sum(),
            inputs=z_mu,
            create_graph=True,
        )[0]
    if model.training is False:
        z_mu.requires_grad = False
    grad = grad * z_std.mean(0, keepdim=True).detach()
    scale = (z_mu.size(1) ** 0.5) / grad.norm(p=2., dim=1, keepdim=True)
    return l2(model.decode(z_mu + scale * z_std * grad), x)

def compute_scales(logits):
    # https://github.com/Rayhane-mamah/Efficient-VDVAE
    softplus = torch.nn.Softplus(beta=0.6931472)
    scales = torch.maximum(softplus(logits), torch.as_tensor(np.exp(-250.)))
    return scales

def _compute_inv_stdv(logits):
    # https://github.com/Rayhane-mamah/Efficient-VDVAE
    scales = compute_scales(logits)
    inv_stdv = 1. / scales
    log_scales = torch.log(scales)
    return inv_stdv, log_scales

def scale_pixels(img, bits):
    # https://github.com/Rayhane-mamah/Efficient-VDVAE
    img = np.floor(img / np.uint8(2 ** (8 - bits))) * 2 ** (8 - bits)
    shift = scale = (2 ** 8 - 1) / 2
    img = (img - shift) / scale
    return img

def mol(logits, targets, bits=32, min_pix_value=0., max_pix_value=1.):
    # https://github.com/Rayhane-mamah/Efficient-VDVAE
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
    # https://github.com/Rayhane-mamah/Efficient-VDVAE
    indices = indices.unsqueeze(dim)
    size = list(indices.size())
    size[dim] = depth
    y_onehot = torch.zeros(size, device=indices.device)
    y_onehot.zero_()
    y_onehot.scatter_(dim, indices, 1)
    return y_onehot

def sample_from_mol(logits, targets, bits=32, min_pix_value=0., max_pix_value=1., temp=1.):
    # https://github.com/Rayhane-mamah/Efficient-VDVAE
    bit_classes = 2. ** bits - 1.
    min_pix_value = scale_pixels(min_pix_value, bits)
    max_pix_value = scale_pixels(max_pix_value, bits)
    B, C, H, W = targets.size()
    if C == 1:
        targets = torch.cat(3*[targets], dim=1)
        C = targets.size(1)
    assert C == 3
    M = logits.size(1) // (3 * C + 1)
    logit_probs = logits[:, :M, :, :]
    l = logits[:, M:, :, :]
    l = l.reshape(B, C, 3 * M, H, W)
    model_means = l[:, :, :M, :, :]
    scales = compute_scales(l[:, :, M: 2 * M, :, :])
    model_coeffs = torch.tanh(l[:, :, 2 * M: 3 * M, :, :])
    gumbel_noise = -torch.log(-torch.log(torch.Tensor(logit_probs.size()).uniform_(1e-5, 1. - 1e-5))).to(logits.device)
    logit_probs = logit_probs / temp + gumbel_noise
    lambda_ = one_hot(torch.argmax(logit_probs, dim=1), logit_probs.size(1), dim=1)
    lambda_ = lambda_.unsqueeze(1)
    means = torch.sum(model_means * lambda_, dim=2)
    scales = torch.sum(scales * lambda_, dim=2)
    coeffs = torch.sum(model_coeffs * lambda_, dim=2)
    u = torch.Tensor(means.size()).uniform_(1e-5, 1. - 1e-5).to(logits.device)
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
        if loss.isnan().sum() > 0:
            print("NaN found in loss!")
        elif amp:
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

def train_epoch_rae(train_iter, epoch_length, train_loader, opt, model, epoch, device, amp, beta):
    model.train()
    epoch_loss = 0
    grad_loss = 0
    z_loss = 0
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
            z = model.encode(images)
            reconstruction = model.decode(z)
            reconstruction = torch.sigmoid(reconstruction)
            recons_loss = l2(reconstruction, images)
            grads_loss = rae_penalty(model, images, z)
            zs_loss = z.norm(p=2.) / 2.
            loss = recons_loss.sum() + beta * grads_loss + 0.1 * beta * zs_loss
        if loss.isnan().sum() > 0:
            print("NaN found in loss!")
        elif amp:
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
        grad_loss += grads_loss.item()
        z_loss += zs_loss.item()
        wandb.log({"train/recon_loss": recons_loss.sum().item(),
                   "train/grad_loss": grads_loss.item(),
                   "train/z_loss": zs_loss.item()})
        progress_bar.set_postfix({"recons_loss": epoch_loss / (step + 1),
                                  "grad_loss": grad_loss / (step + 1),
                                  "z_loss": z_loss / (step + 1)})
    return train_iter

def train_epoch_samba(train_iter, epoch_length, train_loader, opt, model, epoch, device, amp, beta):
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
            z_mu, z_sigma = model.encode(images)
            recons_loss = samba_l2(model, images, z_mu, z_sigma)
            kl_loss = kld(z_mu, 2*(z_sigma).log())
            loss = (recons_loss + beta * kl_loss).sum()
        if loss.isnan().sum() > 0:
            print("NaN found in loss!")
        elif amp:
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

def train_epoch_vae(train_iter, epoch_length, train_loader, opt, model, epoch, device, amp, beta):
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
    DEBUG = False
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
            if DEBUG:
              print("reconstruction:",reconstruction.size())
              print("reconstruction max:",torch.max(reconstruction).item())
              print("reconstruction min:",torch.min(reconstruction).item())
              print("reconstruction std:",torch.std(reconstruction).item())
            reconstruction = torch.sigmoid(reconstruction)
            recons_loss = l2(reconstruction, images)
            if DEBUG:
              print("")
              print("images:",images.size())
              print("images max:",torch.max(images).item())
              print("images min:",torch.min(images).item())
              print("images std:",torch.std(images).item())
              print("reconstruction:",reconstruction.size())
              print("reconstruction max:",torch.max(reconstruction).item())
              print("reconstruction min:",torch.min(reconstruction).item())
              print("reconstruction std:",torch.std(reconstruction).item())
              print("loss")
              print("z_mu:",z_mu.size())
              print("2*((z_sigma).log()):",(2*((z_sigma).log())).size())
            #kl_loss = kld(z_mu, 2*(z_sigma).log())
            kl_loss = kld(z_mu, 2 * torch.log(z_sigma)) # new log var
            if DEBUG:
              print("recons_loss:",recons_loss.size())
              print("kl_loss:",kl_loss.size())
            loss = (recons_loss + beta * kl_loss).sum() #sum() #mean
        if loss.isnan().sum() > 0:
            print("NaN found in loss!")
        elif amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            opt.step()
        epoch_loss += recons_loss.sum().item() #sum() #mean
        kld_loss += kl_loss.sum().item() #sum() #mean
        wandb.log({"train/recon_loss": recons_loss.sum().item()}) #sum() #mean
        wandb.log({"train/kld_loss": kl_loss.sum().item()}) #sum() #mean
        wandb.log({"train/total_loss": loss.sum().item()}) #sum() #mean
        progress_bar.set_postfix({"recons_loss": epoch_loss / (step + 1), "kld_loss": kld_loss / (step + 1)})
    return train_iter

def train_epoch_gaussvae(train_iter, epoch_length, train_loader, opt, model, epoch, device, amp, beta):
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
            kl_loss = kld(z_mu, 2 * torch.log(z_sigma))
            loss = (recons_loss + beta * kl_loss).sum()
        if loss.isnan().sum() > 0:
            print("NaN found in loss!")
        elif amp:
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

def train_epoch_molvae(train_iter, epoch_length, train_loader, opt, model, epoch, device, amp, beta):
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
            kl_loss = kld(z_mu, 2 * torch.log(z_sigma))
            loss = (recons_loss + beta * kl_loss).sum()
        if loss.isnan().sum() > 0:
            print("NaN found in loss!")
        elif amp:
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
        if loss.isnan().sum() > 0:
            print("NaN found in loss!")
        elif amp:
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
    ssim_loss = 0
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
                _ssim = ssim(reconstruction, images)
            if val_step < 16:
                inputs.append(images[0].cpu().float())
                recons.append(reconstruction[0].cpu().float())
            elif val_step == 16:
                grid_inputs = make_grid(inputs, nrow=4, padding=5, normalize=True, scale_each=True)
                grid_recons = make_grid(recons, nrow=4, padding=5, normalize=True, scale_each=True)
                wandb.log({"val/examples": [wandb.Image(grid_inputs[0].numpy(), caption="Real images"),
                                            wandb.Image(grid_recons[0].numpy(), caption="Reconstructions")]})
            val_loss += recons_loss.sum().item()
            ssim_loss += _ssim.item()
            progress_bar.set_postfix({"recons_loss": val_loss / (val_step + 1),
                                      "ssim": ssim_loss / (val_step + 1)})
    wandb.log({"val/recon_loss": val_loss / (val_step + 1)})
    wandb.log({"val/ssim": ssim_loss / (val_step + 1)})
    return ssim_loss / (val_step + 1)

def val_epoch_rae(val_loader, model, device, amp, epoch):
    val_loss = 0
    grad_loss = 0
    z_loss = 0
    ssim_loss = 0
    inputs = []
    recons = []
    ctx = torch.autocast("cuda" if torch.cuda.is_available() else "cpu") if amp else nullcontext()
    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=110)
    progress_bar.set_description(f"[Validation] Epoch {epoch}")
    with torch.no_grad():
        for val_step, batch in progress_bar:
            images = batch["image"].to(device)
            with ctx:
                z = model.encode(images)
                reconstruction = model.decode(z)
                reconstruction = torch.sigmoid(reconstruction)
                recons_loss = l2(reconstruction, images)
                grads_loss = rae_penalty(model, images, z)
                zs_loss = z.norm(p=2.) / 2.
                _ssim = ssim(reconstruction, images)
            if val_step < 16:
                inputs.append(images[0].cpu().float())
                recons.append(reconstruction[0].cpu().float())
            elif val_step == 16:
                grid_inputs = make_grid(inputs, nrow=4, padding=5, normalize=True, scale_each=True)
                grid_recons = make_grid(recons, nrow=4, padding=5, normalize=True, scale_each=True)
                wandb.log({"val/examples": [wandb.Image(grid_inputs[0].numpy(), caption="Real images"),
                                            wandb.Image(grid_recons[0].numpy(), caption="Reconstructions")]})
            val_loss += recons_loss.sum().item()
            grad_loss += grads_loss.sum().item()
            z_loss += zs_loss.sum().item()
            ssim_loss += _ssim.item()
            progress_bar.set_postfix({"recon_loss": val_loss / (val_step + 1),
                                      "grad_loss": grad_loss / (val_step + 1),
                                      "z_loss": z_loss / (val_step + 1),
                                      "ssim": ssim_loss / (val_step + 1)})
    wandb.log({"val/recon_loss": val_loss / (val_step + 1)})
    wandb.log({"val/grad_loss": grad_loss / (val_step + 1)})
    wandb.log({"val/z_loss": z_loss / (val_step + 1)})
    wandb.log({"val/ssim": ssim_loss / (val_step + 1)})
    return ssim_loss / (val_step + 1)

def val_epoch_samba(val_loader, model, device, amp, epoch):
    val_loss = 0
    kld_loss = 0
    ssim_loss = 0
    inputs = []
    recons = []
    samples = []
    ctx = torch.autocast("cuda" if torch.cuda.is_available() else "cpu") if amp else nullcontext()
    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=110)
    progress_bar.set_description(f"[Validation] Epoch {epoch}")
    with torch.no_grad():
        for val_step, batch in progress_bar:
            images = batch["image"].to(device)
            with ctx:
                z_mu, z_sigma = model.encode(images)
                recons_loss = samba_l2(model, images, z_mu, z_sigma)
                kl_loss = kld(z_mu, 2*(z_sigma).log())
                reconstruction = model.decode(z_mu)
                reconstruction = torch.sigmoid(reconstruction)
                _ssim = ssim(reconstruction, images)
            if val_step < 16:
                inputs.append(images[0].cpu().float())
                recons.append(reconstruction[0].cpu().float())
                with ctx:
                    samples.append(torch.sigmoid(model.decode(torch.randn_like(z_mu))[0]).cpu().float())
            elif val_step == 16:
                grid_inputs = make_grid(inputs, nrow=4, padding=5, normalize=True, scale_each=True)
                grid_recons = make_grid(recons, nrow=4, padding=5, normalize=True, scale_each=True)
                grid_samples = make_grid(samples, nrow=4, padding=5, normalize=True, scale_each=True)
                wandb.log({"val/examples": [wandb.Image(grid_inputs[0].numpy(), caption="Real images"),
                                            wandb.Image(grid_recons[0].numpy(), caption="Reconstructions"),
                                            wandb.Image(grid_samples[0].numpy(), caption="Random samples")]})
            val_loss += recons_loss.sum().item()
            kld_loss += kl_loss.sum().item()
            ssim_loss += _ssim.item()
            progress_bar.set_postfix({"recons_loss": val_loss / (val_step + 1),
                                      "kld_loss": kld_loss / (val_step + 1),
                                      "ssim": ssim_loss / (val_step + 1)})
    wandb.log({"val/recon_loss": val_loss / (val_step + 1)})
    wandb.log({"val/kld_loss": kld_loss / (val_step + 1)})
    wandb.log({"val/ssim": ssim_loss / (val_step + 1)})
    return ssim_loss / (val_step + 1)

def val_epoch_vae(val_loader, model, device, amp, epoch):
    val_loss = 0
    kld_loss = 0
    ssim_loss = 0
    inputs = []
    recons = []
    samples = []
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
                _ssim = ssim(reconstruction, images)
            if val_step < 16:
                inputs.append(images[0].cpu().float())
                recons.append(reconstruction[0].cpu().float())
                with ctx:
                    samples.append(torch.sigmoid(model.decode(torch.randn_like(z_mu))[0]).cpu().float())
            elif val_step == 16:
                grid_inputs = make_grid(inputs, nrow=4, padding=5, normalize=True, scale_each=True)
                grid_recons = make_grid(recons, nrow=4, padding=5, normalize=True, scale_each=True)
                grid_samples = make_grid(samples, nrow=4, padding=5, normalize=True, scale_each=True)
                wandb.log({"val/examples": [wandb.Image(grid_inputs[0].numpy(), caption="Real images"),
                                            wandb.Image(grid_recons[0].numpy(), caption="Reconstructions"),
                                            wandb.Image(grid_samples[0].numpy(), caption="Random samples")]})
            val_loss += recons_loss.sum().item()
            kld_loss += kl_loss.sum().item()
            ssim_loss += _ssim.item()
            progress_bar.set_postfix({"recons_loss": val_loss / (val_step + 1),
                                      "kld_loss": kld_loss / (val_step + 1),
                                      "ssim": ssim_loss / (val_step + 1)})
    wandb.log({"val/recon_loss": val_loss / (val_step + 1)})
    wandb.log({"val/kld_loss": kld_loss / (val_step + 1)})
    wandb.log({"val/ssim": ssim_loss / (val_step + 1)})
    return ssim_loss / (val_step + 1)

def val_epoch_gaussvae(val_loader, model, device, amp, epoch):
    val_loss = 0
    kld_loss = 0
    ssim_loss = 0
    inputs = []
    recons = []
    sigmas = []
    samples = []
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
                _ssim = ssim(reconstruction, images)
            if val_step < 16:
                inputs.append(images[0].cpu().float())
                recons.append(reconstruction[0].cpu().float())
                sigmas.append(recon_sigma[0].cpu().float())
                with ctx:
                    samples.append(torch.sigmoid(model.decode(torch.randn_like(z_mu))[0][0]).cpu().float())
            elif val_step == 16:
                grid_inputs = make_grid(inputs, nrow=4, padding=5, normalize=True, scale_each=True)
                grid_recons = make_grid(recons, nrow=4, padding=5, normalize=True, scale_each=True)
                grid_sigmas = make_grid(sigmas, nrow=4, padding=5, normalize=True, scale_each=False)
                grid_samples = make_grid(samples, nrow=4, padding=5, normalize=True, scale_each=True)
                wandb.log({"val/examples": [wandb.Image(grid_inputs[0].numpy(), caption="Real images"),
                                            wandb.Image(grid_recons[0].numpy(), caption="Reconstructions"),
                                            wandb.Image(grid_sigmas[0].numpy(), caption="Uncertanties"),
                                            wandb.Image(grid_samples[0].numpy(), caption="Random samples")]})
            val_loss += recons_loss.sum().item()
            kld_loss += kl_loss.sum().item()
            ssim_loss += _ssim.item()
            progress_bar.set_postfix({"recons_loss": val_loss / (val_step + 1),
                                      "kld_loss": kld_loss / (val_step + 1),
                                      "ssim": ssim_loss / (val_step + 1)})
    wandb.log({"val/recon_loss": val_loss / (val_step + 1)})
    wandb.log({"val/kld_loss": kld_loss / (val_step + 1)})
    wandb.log({"val/ssim": ssim_loss / (val_step + 1)})
    return ssim_loss / (val_step + 1)

def val_epoch_molvae(val_loader, model, device, amp, epoch):
    val_loss = 0
    kld_loss = 0
    ssim_loss = 0
    inputs = []
    recons = []
    samples = []
    ctx = torch.autocast("cuda" if torch.cuda.is_available() else "cpu") if amp else nullcontext()
    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=110)
    progress_bar.set_description(f"[Validation] Epoch {epoch}")
    with torch.no_grad():
        for val_step, batch in progress_bar:
            images = batch["image"].to(device)
            with ctx:
                reconstruction, z_mu, z_sigma = model(images)
                recons_loss = mol(reconstruction, images)
                reconstruction = sample_from_mol(reconstruction, images)
                kl_loss = kld(z_mu, 2*(z_sigma).log())
                _ssim = ssim(reconstruction, images)
            if val_step < 16:
                inputs.append(images[0].cpu().float().mean(0, keepdim=True))
                recons.append(reconstruction[0].cpu().float().mean(0, keepdim=True))
                with ctx:
                    samples.append(sample_from_mol(model.decode(torch.randn_like(z_mu)), images)[0].cpu().float())
            elif val_step == 16:
                grid_inputs = make_grid(inputs, nrow=4, padding=5, normalize=True, scale_each=True)
                grid_recons = make_grid(recons, nrow=4, padding=5, normalize=True, scale_each=True)
                grid_samples = make_grid(samples, nrow=4, padding=5, normalize=True, scale_each=True)
                wandb.log({"val/examples": [wandb.Image(grid_inputs[0].numpy(), caption="Real images"),
                                            wandb.Image(grid_recons[0].numpy(), caption="Reconstructions"),
                                            wandb.Image(grid_samples[0].numpy(), caption="Random samples")]})
            val_loss += recons_loss.sum().item()
            kld_loss += kl_loss.sum().item()
            ssim_loss += _ssim.item()
            progress_bar.set_postfix({"recons_loss": val_loss / (val_step + 1),
                                      "kld_loss": kld_loss / (val_step + 1),
                                      "ssim": ssim_loss / (val_step + 1)})
    wandb.log({"val/recon_loss": val_loss / (val_step + 1)})
    wandb.log({"val/kld_loss": kld_loss / (val_step + 1)})
    wandb.log({"val/ssim": ssim_loss / (val_step + 1)})
    return ssim_loss / (val_step + 1)

def val_epoch_vqvae(val_loader, model, device, amp, epoch):
    val_loss = 0
    val_quant = 0
    ssim_loss = 0
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
                _ssim = ssim(reconstruction, images)
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
            ssim_loss += _ssim.item()
            progress_bar.set_postfix({"recons_loss": val_loss / (val_step + 1),
                                      "quantization_loss": val_quant / (val_step + 1),
                                      "ssim": ssim_loss / (val_step + 1)})
    wandb.log({"val/recon_loss": val_loss / (val_step + 1)})
    wandb.log({"val/quant_loss": val_quant / (val_step + 1)})
    wandb.log({"val/ssim": ssim_loss / (val_step + 1)})
    return ssim_loss / (val_step + 1)
