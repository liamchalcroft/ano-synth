import torchio as tio
import os, glob
import tqdm


images = glob.glob('oasis3d/*/aligned_norm.nii.gz')

data = tio.SubjectsDataset([
    tio.Subject(
        image=tio.ScalarImage(img),
        label=tio.LabelMap(img.replace('norm','seg35')),
    ) for i, img in enumerate(images)
], transform=tio.Compose([
    tio.ToCanonical(),
    tio.Resample('image'),
    tio.Resample(1),
    tio.CropOrPad(224),
    tio.RescaleIntensity((0,1), include="image"),
]), 
)


for i,subj in tqdm.tqdm(enumerate(data),total=len(data)):
    grid = tio.data.GridSampler(subj, patch_size=(200,200,1))
    for j,slc in enumerate(grid):
        odir = 'oasis/sbj{}_slc{}'.format(i+1,j+1)
        os.makedirs(odir, exist_ok=True)
        slc['image'].save(os.path.join(odir, 'aligned_norm.nii.gz'), squeeze=True)
        slc['label'].save(os.path.join(odir, 'aligned_seg35.nii.gz'), squeeze=True)

#! tar -zcvf /content/gdrive/MyDrive/oasis2d.tar.gz oasis/