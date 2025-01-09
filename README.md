# SegMamba

add the SegMamba model to the training framework:

attention:
- you should first install causal_conv1d(1.0.0) and mamba-ssm(1.2.0) 
- SegMamba: https://github.com/ge-xing/SegMamba

seg_3d_train

```bash
python scripts/seg_3d/seg_3d_train.py\
    --data_path /workdir1/echo_dataset/EchoNet-Dynamic \
    --mean 0.12741163 0.1279413 0.12912785 \
    --std 0.19557191 0.19562256 0.1965878 \
    --encoder "SegMamba"\
    --frames 128 \
    --period 1 \
    --num_workers 2 \
    --batch_size 4 \
    --epochs 60  \
    --val_check_interval 0.5 \
    --seed 42
```

using camus dataset

seg_3d_camus_train
```bash
python scripts/seg_3d/seg_3d_camus_train.py\
    --data_path /workdir1/cn24/data/CAMUS \
    --mean 0.12741163 0.1279413 0.12912785 \
    --std 0.19557191 0.19562256 0.1965878 \
    --encoder "SegMamba"\
    --frames 128 \
    --period 1 \
    --num_workers 2 \
    --batch_size 4 \
    --epochs 60  \
    --val_check_interval 0.5 \
    --seed 42
```


using pediatric_echo dataset
```bash
python scripts/seg_3d/seg_3d_train.py\
    --data_path /workdir1/cn24/data/pediatric_echo/A4C \
    --mean 0.12741163 0.1279413 0.12912785 \
    --std 0.19557191 0.19562256 0.1965878 \
    --encoder "SegMamba"\
    --frames 128 \
    --period 1 \
    --num_workers 2 \
    --batch_size 4 \
    --epochs 60  \
    --val_check_interval 0.5 \
    --seed 42
```


# 3D Unet
using pediatric_echo dataset
```bash
python scripts/seg_3d/seg_3d_train.py\
    --data_path /workdir1/cn24/data/pediatric_echo/A4C \
    --mean 0.12741163 0.1279413 0.12912785 \
    --std 0.19557191 0.19562256 0.1965878 \
    --encoder "3d_unet" \
    --frames 32 \
    --period 1 \
    --num_workers 4 \
    --batch_size 16 \
    --epochs 60  \
    --val_check_interval 0.5 \
    --seed 42
```


# LightMUNet

add the LightMUNet model to the training framework:

seg_3d_train

```bash
python scripts/seg_3d/seg_3d_train.py \
    --data_path /workdir1/echo_dataset/EchoNet-Dynamic \
    --mean 0.12741163 0.1279413 0.12912785 \
    --std 0.19557191 0.19562256 0.1965878 \
    --encoder "LightMUNet" \
    --frames 128 \
    --period 1 \
    --num_workers 2 \
    --batch_size 3 \
    --epochs 60  \
    --val_check_interval 0.5 \
    --seed 42
```
mydata_path: 7508: /workdir2t/cn24/data/EchoNet-Dynamic 
             6419: /workdir1/echo_dataset/EchoNet-Dynamic 
