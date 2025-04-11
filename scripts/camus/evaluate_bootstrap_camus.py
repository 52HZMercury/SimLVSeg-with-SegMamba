import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
)

import argparse
import pytorch_lightning as pl
import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import os
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils
import math
import yaml

import collections

import cv2
import pandas

from simlvseg.utils import defaultdict_of_lists

from skimage import measure  # scikit-image库的一部分，用于图像处理
from sklearn.metrics import confusion_matrix  # 用于计算分类任务的混淆矩阵
from medpy.metric.binary import assd as medpy_assd
from medpy.metric.binary import hd95 as medpy_hd95


class CAMUSDatasetEval(torch.utils.data.Dataset):
    def __init__(self, gt_dir, pred_dir, patient_names):
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.patients = patient_names
        self.data = []
        for patient in tqdm.tqdm(self.patients):
            gt = np.load(os.path.join(self.gt_dir, f'{patient}_a4c_gt.npy'))
            gt = self.pad_to_shape(gt)
            pred = np.load(os.path.join(self.pred_dir, f'{patient}_pred.npy'))

            gt = np.float32(gt > 0.5)
            pred = np.float32(pred > 0.5)
            for i in range(gt.shape[0]):
                self.data.append((gt[i], pred[..., i]))

    def pad_to_shape(self, arr):
        if arr.ndim == 4:
            F, H, W, C = arr.shape
        elif arr.ndim == 3:
            F, H, W = arr.shape
            C = 1
        else:
            raise ValueError("输入数组的维度必须为 3 或 4")

        pad_height = 128 - H
        pad_width = 128 - W

        if pad_height < 0 or pad_width < 0:
            raise ValueError("输入数组的尺寸不能大于 128x128")

        if arr.ndim == 4:
            pad_widths = ((0, 0), (0, pad_height), (0, pad_width), (0, 0))
        elif arr.ndim == 3:
            pad_widths = ((0, 0), (0, pad_height), (0, pad_width))

        padded_arr = np.pad(arr, pad_width=pad_widths, mode='constant', constant_values=0)

        return padded_arr

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        return self.data[idx]


class TestData():
    def __init__(self):
        pass
    def mean_and_std(self, values):

        mean = np.mean(values)  # 均值的期望
        std = np.std(values)  # 均值的标准差（即标准误差）

        return mean, std
    def read_patient_names(self, file_path):
        with open(file_path, 'r') as file:
            patient_names = [line.strip() for line in file.readlines()]
        return patient_names

    def run_test(
            self,
            data_dir, output_dir, pred_dir,
            num_workers=4,
    ):
        os.makedirs(output_dir, exist_ok=True)

        patient_names = self.read_patient_names('scripts/camus/database_split/camus_test_filenames.txt')

        for split in ["test"]:
            dataset = CAMUSDatasetEval(gt_dir=data_dir, pred_dir=pred_dir, patient_names=patient_names)

            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=1, num_workers=num_workers,
                shuffle=False, drop_last=False,
            )

            assd_list, hd95_list, sen_list, large_inter, large_union = self.run_epoch(dataloader)

            overall_dice = 2 * large_inter / (large_union + large_inter + 1e-7)
            overall_iou = large_inter / (large_union + 1e-7)


            overall_dice_mean, overall_dice_std = self.mean_and_std(overall_dice)
            overall_iou_mean, overall_iou_std = self.mean_and_std(overall_iou)
            hd95_mean, hd95_std = self.mean_and_std(hd95_list)
            assd_mean, assd_std = self.mean_and_std(assd_list)
            sen_mean, sen_std = self.mean_and_std(sen_list)


            with open(os.path.join(output_dir, "{}_dice.csv".format(split)), "w") as f:
                f.write("Patient_Name,Overall\n")
                for patient_name, overall_dice in zip(dataset.patients, overall_dice):
                    f.write("{},{}\n".format(patient_name, overall_dice))
                f.write("Mean overall_dice,{}\n".format(overall_dice_mean))
                f.write("Standard Deviation overall_dice,{}\n".format(overall_dice_std))

            with open(os.path.join(output_dir, "{}_iou.csv".format(split)), "w") as f:
                f.write("Patient_Name,Overall\n")
                for patient_name, overall_iou in zip(dataset.patients, overall_iou):
                    f.write("{},{}\n".format(patient_name, overall_iou))
                f.write("Mean overall_dice,{}\n".format(overall_iou_mean))
                f.write("Standard Deviation overall_dice,{}\n".format(overall_iou_std))

            with open(os.path.join(output_dir, "{}_assd.csv".format(split)), "w") as f:
                f.write("Patient_Name,ASSD\n")
                for patient_name, assd_value in zip(dataset.patients, assd_list):
                    f.write("{},{}\n".format(patient_name, assd_value))
                f.write("Mean ASSD,{}\n".format(assd_mean))
                f.write("Standard Deviation ASSD,{}\n".format(assd_std))

            with open(os.path.join(output_dir, "{}_hd95.csv".format(split)), "w") as f:
                f.write("Patient_Name,HD95\n")
                for patient_name, hd95_value in zip(dataset.patients, hd95_list):
                    f.write("{},{}\n".format(patient_name, hd95_value))
                f.write("Mean HD95,{}\n".format(hd95_mean))
                f.write("Standard Deviation HD95,{}\n".format(hd95_std))

            with open(os.path.join(output_dir, "{}_sen.csv".format(split)), "w") as f:
                f.write("Patient_Name,SEN\n")
                for patient_name, sen_value in zip(dataset.patients, sen_list):
                    f.write("{},{}\n".format(patient_name, sen_value))
                f.write("Mean SEN,{}\n".format(sen_mean))
                f.write("Standard Deviation SEN,{}\n".format(sen_std))

            with open(os.path.join(output_dir, "log.csv"), "w") as f:
                # f.write("{} dice: {:.4f} - {:.4f}\n".format(split,overall_dice_mean,overall_dice_std))
                # f.write("{} iou: {:.4f} - {:.4f}\n".format(split,overall_iou_mean,overall_iou_std))
                f.write("{} assd: {:.4f} - {:.4f}\n".format(split,assd_mean,assd_std))
                f.write("{} hd95: {:.4f} - {:.4f}\n".format(split,hd95_mean,hd95_std))
                f.write("{} sen: {:.4f} - {:.4f}\n".format(split,sen_mean,sen_std))
                f.flush()

    def run_epoch(self, dataloader):
        assd_list = []
        hd95_list = []
        sen_list = []

        large_inter = 0
        large_union = 0
        large_inter_list = []
        large_union_list = []

        for large_trace, large_pred in tqdm.tqdm(dataloader):
            # large_trace = large_trace.squeeze(0)
            # large_pred = large_pred.squeeze(0)

            assd = self.compute_assd(large_trace, large_pred)
            hd95 = self.compute_hd95(large_trace, large_pred)
            sen = self.compute_sen(large_trace, large_pred)

            assd_list.append(assd)
            hd95_list.append(hd95)
            sen_list.append(sen)

            large_inter += np.logical_and(large_pred > 0., large_trace > 0.).sum()
            large_union += np.logical_or(large_pred > 0., large_trace > 0.).sum()
            large_inter_list.extend(np.logical_and(large_pred > 0., large_trace > 0.).sum((1, 2)))
            large_union_list.extend(np.logical_or(large_pred > 0., large_trace > 0.).sum((1, 2)))

        large_inter_list = np.array(large_inter_list)
        large_union_list = np.array(large_union_list)

        return assd_list, hd95_list, sen_list, np.array(large_inter_list), np.array(large_union_list)



    def compute_iou(self, true_mask, pred_mask):
        true_mask = true_mask > 0.5
        pred_mask = pred_mask > 0.5

        inter = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()

        if union == 0:
            return 0.0
        return inter / union

    def compute_assd(self, true_mask, pred_mask):
        pred = (pred_mask > 0.5).cpu().numpy()  # 确保数据在 CPU 上并转换为 NumPy 数组
        target = (true_mask > 0.5).cpu().numpy()  # 确保数据在 CPU 上并转换为 NumPy 数组

        return medpy_assd(target, pred, voxelspacing=[1.0, 1.0, 1.0])

    def compute_hd95(self, true_mask, pred_mask):
        pred = (pred_mask > 0.5).cpu().numpy()  # 确保数据在 CPU 上并转换为 NumPy 数组
        target = (true_mask > 0.5).cpu().numpy()  # 确保数据在 CPU 上并转换为 NumPy 数组

        return medpy_hd95(target, pred, voxelspacing=[1.0, 1.0, 1.0])

    def compute_sen(self, true_mask, pred_mask):
        true_mask = true_mask > 0.5
        pred_mask = pred_mask > 0.5

        tn, fp, fn, tp = confusion_matrix(true_mask.flatten(), pred_mask.flatten()).ravel()

        if tp + fn == 0:
            return 1.0
        return tp / (tp + fn)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process the paths for data, prediction, and output directories.")

    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the EchoNet Dynamic Dataset directory.')
    parser.add_argument('--prediction_dir', type=str, required=True,
                        help='Path to the directory where predictions will be stored.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output directory where results will be saved.')

    args = parser.parse_args()

    test_model = TestData()

    test_model.run_test(data_dir=args.data_dir, pred_dir=args.prediction_dir, output_dir=args.output_dir)
