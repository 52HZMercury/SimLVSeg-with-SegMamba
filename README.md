add the SegMamba model to the training framework:

attention:
- you should first install causal_conv1d(1.0.0) and mamba-ssm(1.2.0) 
- SegMamba: https://github.com/ge-xing/SegMamba

```bash
python scripts/seg_3d/seg_3d_train.py\
    --data_path /workdir2t/cn24/data/EchoNet-Dynamic\
    --mean 0.12741163 0.1279413 0.12912785 \
    --std 0.19557191 0.19562256 0.1965878 \
    --encoder "SegMamba"\
    --frames 128 \
    --period 1 \
    --num_workers 1 \
    --batch_size 3 \
    --epochs 60  \
    --val_check_interval 0.15 \
    --seed 42
```
mydata_path: /workdir2t/cn24/data/EchoNet-Dynamic/
