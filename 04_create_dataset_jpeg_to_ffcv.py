import PIL.Image
import glob
import os
import subprocess


if os.path.isfile("oasis_ffcv.zip") :
    print("file exist! oasis_ffcv.zip")
    # unzip file
    subprocess.run(["tar", "-xzvf","oasis_ffcv.zip","--skip-old-files"]) 
    
else:
    class MRIDataset:
        def __init__(self, norm_paths):
            self.norm_paths = norm_paths

        def __getitem__(self, idx):
            return (
                PIL.Image.open(self.norm_paths[idx]),
            )

        def __len__(self):
            return len(self.norm_paths)

    class SynthDataset:
        def __init__(self, norm_paths):
            self.norm_paths = norm_paths

        def __getitem__(self, idx):
            return (
            PIL.Image.open(self.norm_paths[idx].replace('norm','seg35')),
            )

        def __len__(self):
            return len(self.norm_paths)
        

    paths = glob.glob('oasis_jpg/*/aligned_norm.jpg')
    train_paths, val_paths = paths[:int(0.8*len(paths))], paths[int(0.8*len(paths)):]
    print("TOTAL train_paths:",len(train_paths))


    from ffcv.fields import RGBImageField
    from ffcv.writer import DatasetWriter

    os.makedirs('oasis_ffcv', exist_ok=True)


    DatasetWriter('oasis_ffcv/oasis2d_mri_train.beton', {
        'image': RGBImageField(write_mode='raw', is_rgb=False),
    }, num_workers=16).from_indexed_dataset(MRIDataset(norm_paths=train_paths))

    DatasetWriter('oasis_ffcv/oasis2d_mri_val.beton', {
    'image': RGBImageField(write_mode='raw', is_rgb=False),
    }, num_workers=16).from_indexed_dataset(MRIDataset(norm_paths=val_paths))

    DatasetWriter('oasis_ffcv/oasis2d_synth_train.beton', {
        'image': RGBImageField(write_mode='raw', is_rgb=False),
    }, num_workers=16).from_indexed_dataset(SynthDataset(norm_paths=train_paths))

    DatasetWriter('oasis_ffcv/oasis2d_synth_val.beton', {
        'image': RGBImageField(write_mode='raw', is_rgb=False),
    }, num_workers=16).from_indexed_dataset(SynthDataset(norm_paths=val_paths))

	#create zip folder
    subprocess.run(["tar", "-zcvf","oasis_ffcv.zip","oasis_ffcv/"]) 
