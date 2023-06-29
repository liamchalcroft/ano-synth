from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (ToTensor, ToDevice,
                             ToTorchImage, RandomHorizontalFlip,
                            RandomTranslate, Convert,
                             ModuleWrapper)
from ffcv.fields.decoders import SimpleGrayscaleImageDecoder
from ffcv.fields import RGBImageField
import torch
import monai as mn
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("QtAgg")

#Random resized crop
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
        super().__init__()
        self.mu = mu
        self.std = std
        self.fwhm = fwhm
        self.gmm_fwhm = gmm_fwhm

    def forward(self, x):
      x = x / 255
      x = x * 20
      x = x.int() #round to closest label val
      labels = [
          mn.transforms.GaussianSmooth(np.random.uniform(0, self.gmm_fwhm))(torch.normal(np.random.uniform(0, self.mu), np.random.uniform(0, self.std), x.shape) * (x==i))
                  for i in range(0,20)] # sample random intensities for each tissue and apply within-tissue blurring
      x = mn.transforms.GaussianSmooth(np.random.uniform(0, self.fwhm))(torch.stack(labels,0).sum(0)) # sum all tissues and apply whole-image blurring
      return x

#Data decoding and augmentation
mri_pipeline = [decoder,
                RandomHorizontalFlip(),
                RandomTranslate(padding=2),
                ToTensor(),
                ToTorchImage(),
                Convert(torch.float),
                ModuleWrapper(Norm(mean=0,std=1)),
                ToDevice(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))]

synth_pipeline = [decoder,
                  RandomHorizontalFlip(),
                  RandomTranslate(padding=2),
                  ToTensor(),
                  ToTorchImage(),
                  Convert(torch.float),
                  ModuleWrapper(GMMSynth()),
                  ModuleWrapper(Norm(mean=0,std=1)),
                  ToDevice(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))]

label_pipeline = [decoder, ToTensor(), ToDevice('cuda:0')]

# Pipeline for each data field
mri_pipelines = {
    'image': mri_pipeline,
    'label': label_pipeline
}
synth_pipelines = {
    'image': synth_pipeline,
    'label': label_pipeline
}

#Replaces PyTorch data loader (`torch.utils.data.Dataloader`)
mri_loader = Loader('oasis_ffcv/oasis2d_mri_train.beton', batch_size=25, num_workers=1,
                order=OrderOption.RANDOM, pipelines={'image': mri_pipeline},
                custom_fields={'image': RGBImageField(write_mode='smart', is_rgb=False)})

synth_loader = Loader('oasis_ffcv/oasis2d_synth_train.beton', batch_size=25, num_workers=1,
                order=OrderOption.RANDOM, pipelines={'image': synth_pipeline},
                custom_fields={'image': RGBImageField(write_mode='smart', is_rgb=False)})


cnt = 0
for item in mri_loader:
  cnt = cnt+1
  if cnt == 1:
    break

plt.figure(figsize=(10,10))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.imshow(item[0][i,0].cpu(), cmap='gray')
  plt.axis('off')
plt.show(block=False)

cnt = 0
for item in synth_loader:
  cnt = cnt+1
  if cnt == 1:
    break

plt.figure(figsize=(10,10))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.imshow(item[0][i,0].cpu(), cmap='gray')
  plt.axis('off')
plt.show()