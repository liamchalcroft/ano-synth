# ano-synth


## Setup

```

pip install -r requirements.txt


sudo apt-get install -y libturbojpeg-dev
sudo apt-get install -y libopencv-dev

pip install git+https://github.com/liamchalcroft/benchmark_VAE@ffcv

```
## Download Dataset

```
bash 01_download_dataset.sh
```

## Create Dataset from 3D to 2D
```
python 02_create_dataset_3d_to_2d.py
```

## Create Dataset from 2D to JPEG
```
python 03_create_dataset_2d_to_jpeg.py
```

## Create Dataset from JPEG to FFCV
```
python 04_create_dataset_jpeg_to_ffcv.py
```
## Test FFCV
```
python 05_test_ffcv.py
```
