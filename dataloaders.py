from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import interpolate
from pythae.data.datasets import DatasetOutput
import custom_monai as cmn
import torchio as tio
import monai as mn
import numpy as np
import os, glob

img_list = glob.glob('oasis/*/aligned_norm.nii.gz')
mb_list = glob.glob('oasis/*/aligned_seg35.nii.gz')

img_list_train, img_list_val = img_list[:int(0.8*len(img_list))], img_list[int(0.8*len(img_list)):]
mb_list_train, mb_list_val = mb_list[:int(0.8*len(mb_list))], mb_list[int(0.8*len(mb_list)):]

class Object(object):
    pass

class CustomQueue(tio.Queue):
    def __init__(
            self,
            subjects_dataset,
            max_length: int,
            samples_per_volume: int,
            sampler,
            num_workers: int = 0,
            shuffle_subjects: bool = True,
            shuffle_patches: bool = True,
            start_background: bool = True,
            verbose: bool = False,
    ):
        self.subjects_dataset = subjects_dataset
        self.max_length = max_length
        self.shuffle_subjects = shuffle_subjects
        self.shuffle_patches = shuffle_patches
        self.samples_per_volume = samples_per_volume
        self.sampler = sampler
        self.subject_sampler = sampler
        self.num_workers = num_workers
        self.verbose = verbose
        self._subjects_iterable = None
        self._incomplete_subject = None
        self._num_patches_incomplete = 0
        self._num_sampled_subjects = 0
        if start_background:
            self._initialize_subjects_iterable()
        self.patches_list = []
    
    def __getitem__(self, _):
        if not self.patches_list:
            self._print('Patches list is empty.')
            self._fill()
            self.patches_list.reverse()
        sample_patch = self.patches_list.pop()
        # return {'data': sample_patch['data'][tio.DATA][...,0]}
        return DatasetOutput(data=sample_patch['data'][tio.DATA][...,0])


class SliceDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        data = self.transform(img_path)
        # data = interpolate(data[None], size=(128,128))[0]
        # return DatasetOutput(data=data)
        return data


def get_mri_data(device):
    transforms = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(),
        tio.Resize(128),
        tio.RandomFlip(axes=(0,1,2)),
        tio.RandomFlip(axes=(0,1,2)),
        tio.RandomFlip(axes=(0,1,2)),
        tio.RandomElasticDeformation(),
        tio.RandomAffine(scales=0.15, degrees=15, translation=10),
        tio.RescaleIntensity(out_min_max=(0,1), percentiles=(0.5,99.5)),
    ])

    subj_train = [tio.Subject(data=tio.ScalarImage(img)) for img in img_list_train]
    subj_val = [tio.Subject(data=tio.ScalarImage(img)) for img in img_list_val]

    data_train = tio.SubjectsDataset(subj_train, transform=transforms)
    data_val = tio.SubjectsDataset(subj_val, transform=transforms)

    patch_sampler = tio.UniformSampler(patch_size=(128,128,1))

    data_train = CustomQueue(data_train, max_length=2048, samples_per_volume=128, sampler=patch_sampler, num_workers=len(os.sched_getaffinity(0)))
    data_val = CustomQueue(data_val, max_length=2048, samples_per_volume=128, sampler=patch_sampler, num_workers=len(os.sched_getaffinity(0)))

    return data_train, data_val


def get_synth_data(device):
    transforms = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(),
        tio.Resize(128),
        tio.RandomFlip(axes=(0,1,2)),
        tio.RandomFlip(axes=(0,1,2)),
        tio.RandomFlip(axes=(0,1,2)),
        tio.RandomLabelsToImage(label_key='label', image_key='data'),
        tio.RandomElasticDeformation(),
        tio.RandomAffine(scales=0.15, degrees=15, translation=10),
        tio.RescaleIntensity(out_min_max=(0,1), percentiles=(0.5,99.5)),
    ])

    subj_train = [tio.Subject(label=tio.ScalarImage(img)) for img in img_list_train]
    subj_val = [tio.Subject(label=tio.ScalarImage(img)) for img in img_list_val]

    data_train = tio.SubjectsDataset(subj_train, transform=transforms)
    data_val = tio.SubjectsDataset(subj_val, transform=transforms)

    patch_sampler = tio.UniformSampler(patch_size=(128,128,1))

    data_train = CustomQueue(data_train, max_length=2048, samples_per_volume=128, sampler=patch_sampler, num_workers=8)
    data_val = CustomQueue(data_val, max_length=2048, samples_per_volume=128, sampler=patch_sampler, num_workers=8)

    return data_train, data_val
