import torch
import monai as mn
import numpy as np
import os, glob
from random import shuffle

img_list_train = glob.glob("2D/train/*_image.nii.gz")
img_list_val = glob.glob("2D/val/*_image.nii.gz")
shuffle(img_list_train)
shuffle(img_list_val)
print("\nTrain Images: {}\nVal Images: {}".format(len(img_list_train), len(img_list_val)))
preproc_list_train = [img.replace("image", "preproc") for img in img_list_train]
preproc_list_val = [img.replace("image", "preproc") for img in img_list_val]

def get_mri_data():
    train_transforms = mn.transforms.Compose([
        mn.transforms.LoadImageD(keys=["image"]),
        mn.transforms.EnsureChannelFirstD(keys=["image"]),
        mn.transforms.ToTensorD(keys=["image"], 
                                dtype=float),
        mn.transforms.SpacingD(keys=["image"], pixdim=1, mode=["bilinear"]),
        mn.transforms.ResizeD(keys=["image"], spatial_size=(192,192), mode=("bilinear")),
        mn.transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1, clip=True),
        mn.transforms.RandFlipD(keys=["image"], spatial_axis=0, prob=0.5),
        mn.transforms.RandFlipD(keys=["image"], spatial_axis=1, prob=0.5),
        mn.transforms.Rand2DElasticD(keys=["image"], spacing=(10,10), magnitude_range=(50,150),
                                  rotate_range=30, shear_range=0.15, translate_range=0.5, scale_range=0.2,
                                  padding_mode="reflection", mode=("bilinear")),
    ])
    val_transforms = mn.transforms.Compose([
        mn.transforms.LoadImageD(keys=["image"]),
        mn.transforms.EnsureChannelFirstD(keys=["image"]),
        mn.transforms.ToTensorD(keys=["image"], 
                                dtype=float),
        mn.transforms.SpacingD(keys=["image"], pixdim=1, mode=["bilinear"]),
        mn.transforms.ResizeD(keys=["image"], spatial_size=(192,192), mode=("bilinear")),
        mn.transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1, clip=True),
    ])
    subj_train = [{"image":img} for img in img_list_train+preproc_list_train]
    subj_val = [{"image":img} for img in img_list_val+preproc_list_val]
    os.makedirs("tmp_data", exist_ok=True)
    data_train = mn.data.PersistentDataset(subj_train, transform=train_transforms, cache_dir="tmp_data")
    data_val = mn.data.PersistentDataset(subj_val, transform=val_transforms, cache_dir="tmp_data")
    return data_train, data_val


def get_synth_data():
    train_transforms = mn.transforms.Compose([
        mn.transforms.LoadImageD(keys=["image", "label"]),
        mn.transforms.EnsureChannelFirstD(keys=["image", "label"]),
        mn.transforms.ToTensorD(keys=["image", "label"], 
                                dtype=float),
        mn.transforms.SpacingD(keys=["image", "label"], pixdim=1, mode=["bilinear", "bilinear"]),
        mn.transforms.OneOf(transforms=[
            mn.transforms.IdentityD(keys=["label"]),
            mn.transforms.MaskIntensityD(keys=["label"], mask_key="image"),
        ]),
        GMMSynthD(mu=255, std=16),
        mn.transforms.ResizeD(keys=["image", "label"], spatial_size=(192,192), mode=("bilinear", "bilinear")),
        mn.transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1, clip=True),
        mn.transforms.RandFlipD(keys=["image", "label"], spatial_axis=0, prob=0.5),
        mn.transforms.RandFlipD(keys=["image", "label"], spatial_axis=1, prob=0.5),
        mn.transforms.Rand2DElasticD(keys=["image", "label"], spacing=(10,10), magnitude_range=(50,150),
                                  rotate_range=30, shear_range=0.15, translate_range=0.5, scale_range=0.2,
                                  padding_mode="reflection", mode=("bilinear", "bilinear")),
    ])
    val_transforms = mn.transforms.Compose([
        mn.transforms.LoadImageD(keys=["image", "label"]),
        mn.transforms.EnsureChannelFirstD(keys=["image", "label"]),
        mn.transforms.ToTensorD(keys=["image", "label"], 
                                dtype=float),
        mn.transforms.SpacingD(keys=["image", "label"], pixdim=1, mode=["bilinear", "bilinear"]),
        mn.transforms.OneOf(transforms=[
            mn.transforms.IdentityD(keys=["label"]),
            mn.transforms.MaskIntensityD(keys=["label"], mask_key="image"),
        ]),
        GMMSynthD(mu=255, std=16),
        mn.transforms.ResizeD(keys=["image", "label"], spatial_size=(192,192), mode=("bilinear", "bilinear")),
        mn.transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1, clip=True),
    ])
    subj_train = [{"image":img, "label":[img.replace("preproc","l{}".format(i)) for i in range(1,10)]} for img in preproc_list_train]
    subj_val = [{"image":img, "label":[img.replace("preproc","l{}".format(i)) for i in range(1,10)]} for img in preproc_list_val]
    os.makedirs("tmp_data", exist_ok=True)
    data_train = mn.data.PersistentDataset(subj_train, transform=train_transforms, cache_dir="tmp_data")
    data_val = mn.data.PersistentDataset(subj_val, transform=val_transforms, cache_dir="tmp_data")
    return data_train, data_val


def get_mix_data():
    train_transforms = mn.transforms.Compose([
        mn.transforms.LoadImageD(keys=["image", "label"]),
        mn.transforms.EnsureChannelFirstD(keys=["image", "label"]),
        mn.transforms.ToTensorD(keys=["image", "label"], 
                                dtype=int),
        mn.transforms.SpacingD(keys=["image", "label"], pixdim=1, mode=["nearest"]),
        mn.transforms.OneOf(transforms=[
            mn.transforms.IdentityD(keys=["label"]),
            mn.transforms.MaskIntensityD(keys=["label"], mask_key="image"),
        ]),
        mn.transforms.OneOf(transforms=[
            mn.transforms.IdentityD(keys=["label"]),
            GMMSynthD(mu=255, std=16),
        ]),
        mn.transforms.ResizeD(keys=["image", "label"], spatial_size=(192,192), mode=("bilinear", "bilinear")),
        mn.transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1, clip=True),
        mn.transforms.RandFlipD(keys=["image", "label"], spatial_axis=0, prob=0.5),
        mn.transforms.RandFlipD(keys=["image", "label"], spatial_axis=1, prob=0.5),
        mn.transforms.Rand2DElasticD(keys=["image", "label"], spacing=(10,10), magnitude_range=(50,150),
                                  rotate_range=30, shear_range=0.15, translate_range=0.5, scale_range=0.2,
                                  padding_mode="reflection", mode=("bilinear", "bilinear")),
    ])
    val_transforms = mn.transforms.Compose([
        mn.transforms.LoadImageD(keys=["image", "label"]),
        mn.transforms.EnsureChannelFirstD(keys=["image", "label"]),
        mn.transforms.ToTensorD(keys=["image", "label"], 
                                dtype=int),
        mn.transforms.SpacingD(keys=["image", "label"], pixdim=1, mode=["bilinear", "bilinear"]),
        mn.transforms.OneOf(transforms=[
            mn.transforms.IdentityD(keys=["label"]),
            mn.transforms.MaskIntensityD(keys=["label"], mask_key="image"),
        ]),
        mn.transforms.OneOf(transforms=[
            mn.transforms.IdentityD(keys=["label"]),
            GMMSynthD(mu=255, std=16),
        ]),
        mn.transforms.ResizeD(keys=["image", "label"], spatial_size=(192,192), mode=("bilinear", "bilinear")),
        mn.transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1, clip=True),
    ])
    subj_train = [{"image":img, "label":[img.replace("preproc","l{}".format(i)) for i in range(1,10)]} for img in preproc_list_train]\
        + [{"image":img, "label":[img.replace("image","l{}".format(i)) for i in range(1,10)]} for img in img_list_train]
    subj_val = [{"image":img, "label":[img.replace("preproc","l{}".format(i)) for i in range(1,10)]} for img in preproc_list_val]\
        + [{"image":img, "label":[img.replace("image","l{}".format(i)) for i in range(1,10)]} for img in img_list_val]
    os.makedirs("tmp_data", exist_ok=True)
    data_train = mn.data.PersistentDataset(subj_train, transform=train_transforms, cache_dir="tmp_data")
    data_val = mn.data.PersistentDataset(subj_val, transform=val_transforms, cache_dir="tmp_data")
    return data_train, data_val


class GMMSynthD:
    def __init__(self, mu=255, std=16, gmm_fwhm=2):
        self.mu = mu
        self.std = std
        self.gmm_fwhm = gmm_fwhm

    def __call__(self, data):
        d = dict(data)
        label = d["label"]
        print(label.shape)
        for i in range(label.size(0)):
            print(label[i].min(), label[i].mean(), label[i].max())
        labels = [
            mn.transforms.GaussianSmooth(np.random.uniform(0, self.gmm_fwhm))(torch.normal(np.random.uniform(0, self.mu), 
                                                                                           np.random.uniform(0, self.std), 
                                                                                           label[0:1].shape) * (label[i]))
                   for i in range(label.size(0))] # sample random intensities for each tissue and apply within-tissue blurring
        for i in range(label.size(0)):
            print(labels[i].min(), labels[i].mean(), labels[i].max())
        d["image"] = torch.stack(labels, 0).sum(0)
        return d
