# ano-synth


## Setup

```

pip install -r requirements


sudo apt-get install libturbojpeg-dev
sudo apt-get install libopencv-dev

git clone https://github.com/liamchalcroft/ffcv
cd ffcv
git pull
pip install -q .
cd ..

# pip install pythae==0.1.1
# install pythae from repository:
pip install -q git+https://github.com/liamchalcroft/benchmark_VAE@ffcv

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
