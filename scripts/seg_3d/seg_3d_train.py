import os
import sys


sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
)

import argparse
import numpy as np
import pytorch_lightning as pl
import random
import torch

from torch.utils.data import DataLoader

from simlvseg.augmentation import get_augmentation
from simlvseg.utils import set_seed
from simlvseg.seg_3d.dataset import Seg3DDataset
from simlvseg.seg_3d.pl_module import Seg3DModule
from simlvseg.seg_3d.preprocessing import get_preprocessing_for_training

def parse_args():
    parser = argparse.ArgumentParser(description="Weakly Supervised Video Segmentation Training with 3D Models")

    parser.add_argument('--seed', type=int, default=42)

    # Paths and dataset related arguments
    parser.add_argument('--data_path', type=str, help="Path to the dataset", required=True)
    parser.add_argument('--mean', type=float, nargs=3, default=(0.12741163, 0.1279413, 0.12912785),
                        help="Mean normalization value (can be a list or tuple)")
    parser.add_argument('--std', type=float, nargs=3, default=(0.19557191, 0.19562256, 0.1965878),
                        help="Standard deviation normalization value (can be a list or tuple)")

    # Model and training related arguments
    parser.add_argument('--encoder', type=str, default='3d_unet', help="Encoder type")
    parser.add_argument('--frames', type=int, default=32, help="Number of frames")
    parser.add_argument('--period', type=int, default=1, help="Period")
    parser.add_argument('--pct_train', type=float, default=None, help="Percentage of training data to use (can be None or a float)")

    # DataLoader arguments
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for data loading")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training and validation")
    
    # Training procedure arguments
    parser.add_argument('--epochs', type=int, default=70, help="Number of epochs to train for")
    parser.add_argument('--val_check_interval', type=float, default=0.25, help="Interval at which to check validation performance")

    # Checkpointing arguments
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to a checkpoint file (can be None or a string)")
    parser.add_argument('--pretrained_type', type=str, default='pl_full', choices=['pl_full', 'encoder'],
                        help="Type of pretraining to use ('pl_full' or 'encoder')")


    args = parser.parse_args()
    return args

class DataModule(pl.LightningDataModule):
    def __init__(self, augmentation, preprocessing):
        super().__init__()
        
        print('Configuring train dataset ...')
        self.train_dataset = Seg3DDataset(
            args.data_path,
            "train",
            args.frames,
            args.period,
            False,
            preprocessing,
            augmentation,
            pct_train=args.pct_train,
        )
        
        print('Configuring val dataset ...')
        self.val_dataset = Seg3DDataset(
            args.data_path,
            "val",
            args.frames,
            args.period,
            False,
            preprocessing,
            None,
            test=True,
        )
        
        print('Configuring test dataset ...')
        self.test_dataset = Seg3DDataset(
            args.data_path,
            "test",
            args.frames,
            args.period,
            False,
            preprocessing,
            None,
            test=True,
        )


    def train_dataloader(self):

        return DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=True)
    
    def val_dataloader(self):

        return DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, drop_last=False)
    
    def test_dataloader(self):

        return DataLoader(self.test_dataset, batch_size=1, shuffle=False,
                          num_workers=args.num_workers, drop_last=False)

if __name__ == '__main__':
    args = parse_args()

    set_seed(args.seed)

    augmentation = get_augmentation(args.frames)

    preprocessing = get_preprocessing_for_training(
        args.frames,
        args.mean,
        args.std,
    )

    model = Seg3DModule(args.encoder, args.checkpoint, args.pretrained_type)

    dm = DataModule(augmentation, preprocessing)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        mode='max',
        monitor='val_dsc',
        #monitor='train_dsc',
        verbose=True,
        save_last=True,

        # 保存最佳
        save_top_k=1,
        save_weights_only=False,

        # 每隔5个Epoch保存一次
        # save_top_k=-1,
        # save_weights_only=False,
        # every_n_epochs=5
    )

    # 32位精度
    # 多卡并行变为ddp后，需要调高lr，倍数为对应显卡张数或根号倍 devices=[0, 1], strategy="ddp", sync_batchnorm=True,
    trainer = pl.Trainer(accelerator="gpu", devices=[2], max_epochs=args.epochs,
                        val_check_interval=args.val_check_interval,
                        log_every_n_steps=10,
                        callbacks=[checkpoint_callback])


    trainer.fit(model, dm)
    trainer.test(model, dataloaders=dm.test_dataloader(), ckpt_path='best')


