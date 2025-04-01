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
    --frames 32 \
    --period 1 \
    --num_workers 2 \
    --batch_size 4 \
    --epochs 60  \
    --val_check_interval 0.25 \
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
    --frames 32 \
    --period 1 \
    --num_workers 2 \
    --batch_size 4 \
    --epochs 60  \
    --val_check_interval 0.25 \
    --seed 42
```


using pediatric_echo dataset
```bash
python scripts/seg_3d/seg_3d_train.py\
    --data_path /workdir1/cn24/data/pediatric_echo/A4C \
    --mean 0.12741163 0.1279413 0.12912785 \
    --std 0.19557191 0.19562256 0.1965878 \
    --encoder "SegMamba"\
    --frames 32 \
    --period 1 \
    --num_workers 2 \
    --batch_size 4 \
    --epochs 60  \
    --val_check_interval 0.25 \
    --seed 42
```


# 3D Unet

using camus dataset

```bash
python scripts/seg_3d/seg_3d_camus_train.py\
    --data_path /workdir1/cn24/data/CAMUS \
    --mean 0.12741163 0.1279413 0.12912785 \
    --std 0.19557191 0.19562256 0.1965878 \
    --encoder "3d_unet"\
    --frames 32 \
    --period 1 \
    --num_workers 4 \
    --batch_size 16 \
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

```bash
python scripts/seg_3d/seg_3d_train.py \
    --data_path /workdir1/cn24/data/pediatric_echo/A4C \
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
camus
```bash
python scripts/seg_3d/seg_3d_camus_train.py \
    --data_path /workdir1/cn24/data/CAMUS \
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



# UKAN3D

add the UKAN3D model to the training framework:

```bash
python scripts/seg_3d/seg_3d_train.py \
    --data_path /workdir1/cn24/data/pediatric_echo/A4C \
    --mean 0.12741163 0.1279413 0.12912785 \
    --std 0.19557191 0.19562256 0.1965878 \
    --encoder "UKAN3D" \
    --frames 128 \
    --period 1 \
    --num_workers 2 \
    --batch_size 4 \
    --epochs 60  \
    --val_check_interval 0.5 \
    --seed 42
```

# SegFormer3D

add the SegFormer3D model to the training framework:

```bash
python scripts/seg_3d/seg_3d_train.py \
    --data_path /workdir1/cn24/data/pediatric_echo/A4C \
    --mean 0.12741163 0.1279413 0.12912785 \
    --std 0.19557191 0.19562256 0.1965878 \
    --encoder "SegFormer3D" \
    --frames 112 \
    --period 1 \
    --num_workers 2 \
    --batch_size 4 \
    --epochs 60  \
    --val_check_interval 0.5 \
    --seed 42
```

# UNet_idc3D

add the UNet_idc3D model to the training framework:

```bash
python scripts/seg_3d/seg_3d_train.py \
    --data_path /workdir1/cn24/data/pediatric_echo/A4C \
    --mean 0.12741163 0.1279413 0.12912785 \
    --std 0.19557191 0.19562256 0.1965878 \
    --encoder "UNet_idc3D" \
    --frames 112 \
    --period 1 \
    --num_workers 2 \
    --batch_size 4 \
    --epochs 60  \
    --val_check_interval 0.5 \
    --seed 42
```


mydata_path: 7508: /workdir2t/cn24/data/EchoNet-Dynamic 
             6419: /workdir1/echo_dataset/EchoNet-Dynamic

>>output.log 2>&1 &
