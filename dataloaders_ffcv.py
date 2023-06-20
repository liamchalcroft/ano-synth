from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (ToTensor, ToDevice, 
                             ToTorchImage, RandomHorizontalFlip,
                             RandomTranslate,
                             ModuleWrapper, Convert)
from ffcv.fields.decoders import SimpleGrayscaleImageDecoder
from ffcv.fields import RGBImageField
import torch
import monai as mn
import numpy as np

# Random resized crop
decoder = SimpleGrayscaleImageDecoder()

# class Norm(torch.nn.Module):
#   def __init__(self, mean, std):
#     super().__init__()
#     self.mean = mean
#     self.std = std

#   def forward(self, x):
#     if x.mean()==0 and x.std()==0:
#       return x
#     else:
#       return (x-x.mean()) / x.std()

class Norm(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    if x.mean()==0 and x.std()==0:
      return x
    else:
      x = x - x.min()
      x = x / x.max()
      return x

class GMMSynth(torch.nn.Module):
    def __init__(self, mu=255, std=16, fwhm=5, gmm_fwhm=5):
        super().__init__()
        self.mu = mu
        self.std = std
        self.fwhm = fwhm
        self.gmm_fwhm = gmm_fwhm

    def forward(self, x):
      x = x / 255
      x = x * 20
      x = x.int() # round to closest label val
      labels = [
          mn.transforms.GaussianSmooth(np.random.uniform(0, self.gmm_fwhm))(torch.normal(np.random.uniform(0, self.mu), np.random.uniform(0, self.std), x.shape) * (x==i))
                  for i in range(1,20)] # sample random intensities for each tissue and apply within-tissue blurring
      x = mn.transforms.GaussianSmooth(np.random.uniform(0, self.fwhm))(torch.stack(labels,0).sum(0)) # sum all tissues and apply whole-image blurring
      return x

# Data decoding and augmentation
mri_pipeline = [decoder, 
                RandomHorizontalFlip(),
                RandomTranslate(padding=2), 
                ToTensor(), 
                ToTorchImage(), 
                Convert(torch.float),
                ModuleWrapper(Norm()),
                ToDevice(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))]
synth_pipeline = [decoder, 
                  RandomHorizontalFlip(),
                  RandomTranslate(padding=2), 
                  ToTensor(), 
                  ToTorchImage(), 
                  Convert(torch.float),
                  ModuleWrapper(GMMSynth()), 
                  ModuleWrapper(Norm()), 
                  ToDevice(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))]

# Replaces PyTorch data loader (`torch.utils.data.Dataloader`)
def get_mri_ffcv(batch_size, num_workers):
    mri_train_loader = Loader('oasis/oasis2d_mri_train.beton', batch_size=batch_size, num_workers=num_workers,
                    order=OrderOption.RANDOM, pipelines={'image': mri_pipeline},
                    custom_fields={'image': RGBImageField(write_mode='raw', is_rgb=False)})
    mri_val_loader = Loader('oasis/oasis2d_mri_val.beton', batch_size=batch_size, num_workers=num_workers,
                    order=OrderOption.RANDOM, pipelines={'image': mri_pipeline},
                    custom_fields={'image': RGBImageField(write_mode='raw', is_rgb=False)})
    return mri_train_loader, mri_val_loader

def get_synth_ffcv(batch_size, num_workers):
    synth_train_loader = Loader('oasis/oasis2d_synth_train.beton', batch_size=batch_size, num_workers=num_workers,
                    order=OrderOption.RANDOM, pipelines={'image': synth_pipeline},
                    custom_fields={'image': RGBImageField(write_mode='raw', is_rgb=False)})
    synth_val_loader = Loader('oasis/oasis2d_synth_val.beton', batch_size=batch_size, num_workers=num_workers,
                    order=OrderOption.RANDOM, pipelines={'image': synth_pipeline},
                    custom_fields={'image': RGBImageField(write_mode='raw', is_rgb=False)})
    return synth_train_loader, synth_val_loader