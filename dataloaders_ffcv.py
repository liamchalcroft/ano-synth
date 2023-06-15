from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (ToTensor, ToDevice, 
                             ToTorchImage, RandomHorizontalFlip,
                             RandomVerticalFlip, RandomTranslate,
                             ModuleWrapper)
from ffcv.fields.decoders import SimpleGrayscaleImageDecoder
import torch
import monai as mn
import numpy as np

# Random resized crop
decoder = SimpleGrayscaleImageDecoder()

class Norm(torch.nn.Module):
  def __init__(self, mean, std):
    super().__init__()
    self.mean = mean
    self.std = std

  def forward(self, x):
    if x.mean()==0 and x.std()==0:
      return x
    else:
      return (x-x.mean()) / x.std()

class GMMSynth(torch.nn.Module):
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

    def forward(self, x):
      x = x.int()
      x.apply_(lambda val: self.labmap[val]) # map to symmetric mask
      labels = [
          mn.transforms.GaussianSmooth(np.random.uniform(0, self.gmm_fwhm))(torch.normal(np.random.uniform(0, self.mu), np.random.uniform(0, self.std), x.shape) * (x==i))
                  for i in range(20)] # sample random intensities for each tissue and apply within-tissue blurring
      x = mn.transforms.GaussianSmooth(np.random.uniform(0, self.fwhm))(torch.stack(labels,0).sum(0)) # sum all tissues and apply whole-image blurring
      return x

# Data decoding and augmentation
mri_pipeline = [decoder, RandomHorizontalFlip(), RandomVerticalFlip(),
            RandomTranslate(padding=2), ToTensor(), ModuleWrapper(Norm(mean=0,std=1)),
                ToDevice('cuda:0')]
synth_pipeline = [decoder, RandomHorizontalFlip(), RandomVerticalFlip(),
            RandomTranslate(padding=2), ToTensor(), ModuleWrapper(GMMSynth()), 
                  ModuleWrapper(Norm(mean=0,std=1)), ToDevice('cuda:0')]

# Replaces PyTorch data loader (`torch.utils.data.Dataloader`)
def get_mri_ffcv(batch_size, num_workers):
    mri_train_loader = Loader('oasis/oasis2d_mri_train.beton', batch_size=batch_size, num_workers=num_workers,
                    order=OrderOption.RANDOM, pipelines={'image': mri_pipeline})
    mri_val_loader = Loader('oasis/oasis2d_mri_val.beton', batch_size=512, num_workers=16,
                    order=OrderOption.RANDOM, pipelines={'image': mri_pipeline})
    return mri_train_loader, mri_val_loader

def get_synth_ffcv(batch_size, num_workers):
    synth_train_loader = Loader('oasis/oasis2d_synth_train.beton', batch_size=batch_size, num_workers=num_workers,
                    order=OrderOption.RANDOM, pipelines={'image': synth_pipeline})
    synth_val_loader = Loader('oasis/oasis2d_synth_val.beton', batch_size=512, num_workers=16,
                    order=OrderOption.RANDOM, pipelines={'image': synth_pipeline})
    return synth_train_loader, synth_val_loader