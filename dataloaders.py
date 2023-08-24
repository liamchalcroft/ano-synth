import torch
import monai as mn
import numpy as np
import os, glob

img_list = glob.glob('oasis/*/aligned_norm.nii.gz')
# mb_list = glob.glob('oasis/*/aligned_seg35.nii.gz')

img_list_train, img_list_val = img_list[:int(0.8*len(img_list))], img_list[int(0.8*len(img_list)):]
# mb_list_train, mb_list_val = mb_list[:int(0.8*len(mb_list))], mb_list[int(0.8*len(mb_list)):]

print('Train Images: {}\nVal Images: {}'.format(len(img_list_train), len(img_list_val)))

def get_mri_data(device):
    transforms = mn.transforms.Compose([
        mn.transforms.LoadImageD(keys=["image", "label"]),
        mn.transforms.EnsureChannelFirstD(keys=["image", "label"]),
        mn.transforms.ToTensorD(keys=["image","label"], 
                                # device=device, 
                                dtype=float),
        mn.transforms.SpacingD(keys=['image','label'], pixdim=1, mode=['bilinear', 'nearest']),
        mn.transforms.ResizeD(keys=['image','label'], spatial_size=(192,192), mode=('bilinear','nearest')),
        mn.transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0.5, upper=99.5, b_min=0, b_max=1),
        mn.transforms.RandFlipD(keys=['image','label'], spatial_axis=0, prob=0.5),
        mn.transforms.RandFlipD(keys=['image','label'], spatial_axis=1, prob=0.5),
        mn.transforms.Rand2DElasticD(keys=['image','label'], spacing=(10,10), magnitude_range=(50,150),
                                  rotate_range=30, shear_range=0.15, translate_range=0.5, scale_range=0.2,
                                  padding_mode='reflection', mode=('bilinear','nearest')),
    ])

    subj_train = [{"image":img, "label":img.replace('norm','seg35')} for img in img_list_train]
    subj_val = [{"image":img, "label":img.replace('norm','seg35')} for img in img_list_val]
    os.makedirs('tmp_data', exist_ok=True)

    data_train = mn.data.PersistentDataset(subj_train, transform=transforms, cache_dir='tmp_data')
    data_val = mn.data.PersistentDataset(subj_val, transform=transforms, cache_dir='tmp_data')

    return data_train, data_val


def get_synth_data(device):
    transforms = mn.transforms.Compose([
        mn.transforms.LoadImageD(keys=["label"]),
        mn.transforms.EnsureChannelFirstD(keys=["label"]),
        mn.transforms.ToTensorD(keys=["label"], 
                                # device=device, 
                                dtype=int),
        mn.transforms.SpacingD(keys=['label'], pixdim=1, mode=['nearest']),
        GMMSynthD(mu=255, std=16, fwhm=5, gmm_fwhm=5),
        mn.transforms.ResizeD(keys=['image','label'], spatial_size=(192,192), mode=('bilinear','nearest')),
        mn.transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0.5, upper=99.5, b_min=0, b_max=1),
        mn.transforms.RandFlipD(keys=['image','label'], spatial_axis=0, prob=0.5),
        mn.transforms.RandFlipD(keys=['image','label'], spatial_axis=1, prob=0.5),
        mn.transforms.Rand2DElasticD(keys=['image','label'], spacing=(10,10), magnitude_range=(50,150),
                                  rotate_range=30, shear_range=0.15, translate_range=0.5, scale_range=0.2,
                                  padding_mode='reflection', mode=('bilinear','nearest')),
    ])

    subj_train = [{"image":img, "label":img.replace('norm','seg35')} for img in img_list_train]
    subj_val = [{"image":img, "label":img.replace('norm','seg35')} for img in img_list_val]
    os.makedirs('tmp_data', exist_ok=True)

    data_train = mn.data.PersistentDataset(subj_train, transform=transforms, cache_dir='tmp_data')
    data_val = mn.data.PersistentDataset(subj_val, transform=transforms, cache_dir='tmp_data')

    return data_train, data_val


class GMMSynthD:
    def __init__(self, mu=255, std=16, fwhm=5, gmm_fwhm=5):
        self.mu = mu
        self.std = std
        self.fwhm = fwhm
        self.gmm_fwhm = gmm_fwhm
        self.labmap = {
            0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9,
            10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17, 18:18, 19:19,
            20:1, 21:2, 22:3, 23:4, 24:5, 25:6, 26:7, 27:8, 28:9, 29:10,
            30:14, 31:15, 32:16, 33:17, 34:18, 35:19
        }

    def __call__(self, data):
        d = dict(data)
        label = d["label"].int()
        label.apply_(lambda val: self.labmap[val]) # map to symmetric mask
        labels = [
            mn.transforms.GaussianSmooth(np.random.uniform(0, self.gmm_fwhm))(torch.normal(np.random.uniform(0, self.mu), np.random.uniform(0, self.std), label.shape) * (label==i))
                   for i in range(20)] # sample random intensities for each tissue and apply within-tissue blurring
        d["image"] = mn.transforms.GaussianSmooth(np.random.uniform(0, self.fwhm))(torch.stack(labels,0).sum(0)) # sum all tissues and apply whole-image blurring
        return d
