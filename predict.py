import argparse
import os, csv, glob
from models import Autoencoder, AutoencoderKL, GaussAutoencoderKL, VQVAE
import torch
import monai as mn
import nibabel as nb
train_utils.set_global_seed(seed= 42)
import tqdm
import logging
from contextlib import nullcontext
from train_utils import l2, ssim, sample_from_mol
logging.getLogger("monai").setLevel(logging.ERROR)

if __name__ =='__main__':

    parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", type=str, help="Name of WandB run.")
    parser.add_argument("--model", type=str, help="Model to use.",
                        choices=["AE", "RAE", "SAMBA", "VAE", "GaussVAE", "MOLVAE", "VQVAE"])
    parser.add_argument("--workers", type=int, default=0, help="Number of workers for dataloaders.")
    parser.add_argument("--mixtures", type=int, default=10, help="Number of mixtures for MOLVAE.")
    parser.add_argument("--amp", action='store_true', help="Use auto mixed precision in training.")
    parser.add_argument("--use_best", action='store_true', help="Resume from checkpoint with highest SSIM.")
    parser.add_argument("--root", type=str, default='./', help="Root dir to save output directory within.")
    parser.add_argument("--data", type=str, help="Path to target data.")
    parser.add_argument("--data_name", type=str, help="Folder name to save predictions under.")
    parser.add_argument("--slice_batch_size", type=int, default=32, help="Number of slices to load as a batch at once.")
    args = parser.parse_args()

    assert os.path.exists(os.path.join(args.root, args.name, "checkpoint.pt"))
    odir = os.path.join(args.root, args.name, "predictions", args.data_name)
    os.makedirs(odir)
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
            num_embeddings=128,
            embedding_dim=128,
            decay=0.99,
            downsample_parameters=tuple([(1, 4, 1, 1)]+5*[(2, 4, 1, 1)]),
            upsample_parameters=tuple([(1, 4, 1, 1, 0)]+5*[(2, 4, 1, 1, 0)]),
        ).to(device)

    ckpt = torch.load(os.path.join(args.root, args.name,
                                   'checkpoint_best.pt' if args.use_best else 'checkpoint.pt'), 
                                   map_location=device)
    model.load_state_dict(ckpt["net"], strict=False)
    model.eval()
        
    img_list = glob.glob(os.path.join(args.data, "*.nii*"))
    img_list = [{"image": img, "fname": img.split('/')[-1].split('.')[0]} for img in img_list]
    print("\nTotal Images: {}".format(len(img_list)))

    load = mn.transforms.LoadImageD(keys=["image"])
    transforms = mn.transforms.Compose([
        mn.transforms.EnsureChannelFirstD(keys=["image"]),
        mn.transforms.OrientationD(keys=["image"], axcodes="RAS"),
        mn.transforms.ToTensorD(keys=["image"], 
                                dtype=float),
        mn.transforms.SpacingD(keys=["image"], pixdim=(1,1,-1), mode=["bilinear"]),
        mn.transforms.ResizeD(keys=["image"], spatial_size=(224,224,-1), mode=["bilinear"]),
        mn.transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1, clip=True),
    ])

    ctx = torch.autocast("cuda" if torch.cuda.is_available() else "cpu") if args.amp else nullcontext()

    inferer = mn.inferers.SliceInferer(roi_size=(224,224), spatial_dim=2, sw_batch_size=args.slice_batch_size)

    recon_scores = []

    for item in tqdm.tqdm(img_list, total=len(img_list)):
        unmodified_item = load(item)
        item = transforms(unmodified_item)
        fname = item["fname"]
        img = item["image"].to(device)[None]

        with torch.no_grad():
            with ctx:
                if args.model=="AE":
                    reconstruction = inferer(img, model)
                    reconstruction = torch.sigmoid(reconstruction)
                elif args.model=="RAE":
                    z = inferer(img, model.encode)
                    reconstruction = inferer(z, model.decode)
                    reconstruction = torch.sigmoid(reconstruction)
                elif args.model=="SAMBA":
                    z_mu, z_sigma = inferer(img, model.encode)
                    reconstruction = inferer(z_mu, model.decode)
                    reconstruction = torch.sigmoid(reconstruction)
                elif args.model=="VAE":
                    reconstruction, z_mu, z_sigma = inferer(img, model)
                    reconstruction = torch.sigmoid(reconstruction)
                elif args.model=="GaussVAE":
                    reconstruction, recon_sigma, z_mu, z_sigma = inferer(img, model)
                    reconstruction = torch.sigmoid(reconstruction)
                elif args.model=="MOLVAE":
                    reconstruction, z_mu, z_sigma = inferer(img, model)
                    reconstruction = sample_from_mol(reconstruction, img)
                elif args.model=="VQVAE":
                    reconstruction, quantization_loss = inferer(img, model)
                    reconstruction = torch.sigmoid(reconstruction)

        reconstruction = reconstruction[0]
        reconstruction.applied_operations = item["image"].applied_operations

        pred_dict = {}
        pred_dict["image"] = reconstruction
        with mn.transforms.utils.allow_missing_keys_mode(transforms):
            inverted_pred = transforms.inverse(pred_dict)

        reconstruction = inverted_pred["image"]
        img = unmodified_item["image"]
        affine = unmodified_item['image_meta_dict']['original_affine'][0].numpy()
                    
        recon_scores.append({
            "fname": fname, 
            "l2": float(l2(torch.Tensor(reconstruction[None]), torch.Tensor(img[None]))),
            "ssim": (ssim(torch.Tensor(reconstruction[None]), torch.Tensor(img[None])))
        })

        nb.save(nb.Nifti1Image(reconstruction, affine), os.path.join(odir, fname+".nii.gz"))
