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
from skimage import measure
from sklearn.metrics import confusion_matrix



import collections

import skimage.draw
import torchvision
import cv2
import pandas

from simlvseg.utils import defaultdict_of_lists


class Echo(torchvision.datasets.VisionDataset):
    def __init__(
            self,
            root,
            pred_dir=None,
            split="test",
            target_type="EF",
            external_test_location=None,
            frame_shape=(112, 112)
    ):

        super().__init__(root)

        self.frame_shape = frame_shape

        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.pred_dir = pred_dir
        self.external_test_location = external_test_location

        self.fnames, self.outcome = [], []

        if self.split == "EXTERNAL_TEST":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            # Load video-level labels
            with open(os.path.join(self.root, "FileList.csv")) as f:
                data = pandas.read_csv(f)
            data["Split"].map(lambda x: x.upper())

            if self.split != "ALL":
                data = data[data["Split"] == self.split]

            self.header = data.columns.tolist()
            self.fnames = data["FileName"].tolist()
            self.fnames = [fn + ".avi" for fn in self.fnames if
                           os.path.splitext(fn)[1] == ""]  # Assume avi if no suffix
            self.outcome = data.values.tolist()

            # Check that files are present
            missing = set(self.fnames) - set(os.listdir(os.path.join(self.root, "Videos")))
            if len(missing) != 0:
                print("{} videos could not be found in {}:".format(len(missing), os.path.join(self.root, "Videos")))
                for f in sorted(missing):
                    print("\t", f)
                raise FileNotFoundError(os.path.join(self.root, "Videos", sorted(missing)[0]))

            # Load traces
            self.frames = collections.defaultdict(list)
            self.preds = collections.defaultdict(list)

            self.trace = collections.defaultdict(defaultdict_of_lists)

            with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
                header = f.readline().strip().split(",")
                assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

                for line in f:
                    filename, x1, y1, x2, y2, frame = line.strip().split(',')
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    frame = int(frame)
                    if frame not in self.trace[filename]:
                        self.frames[filename].append(frame)
                    self.trace[filename][frame].append((x1, y1, x2, y2))
            for filename in self.frames:
                for frame in self.frames[filename]:
                    self.trace[filename][frame] = np.array(self.trace[filename][frame])

            # A small number of videos are missing traces; remove these videos
            keep = [len(self.frames[f]) >= 2 for f in self.fnames]
            self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
            self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]

    def __getitem__(self, index):
        # Find filename of video
        if self.split == "EXTERNAL_TEST":
            video = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "CLINICAL_TEST":
            video = os.path.join(self.root, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video = os.path.join(self.root, "Videos", self.fnames[index])

        # Gather targets
        target = []
        for t in self.target_type:
            key = self.fnames[index]
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "LargeIndex":
                # Traces are sorted by cross-sectional area
                # Largest (diastolic) frame is last
                target.append(int(self.frames[key][-1]))
            elif t == "SmallIndex":
                # Largest (diastolic) frame is first
                target.append(int(self.frames[key][0]))
            elif t in ["LargeTrace", "SmallTrace"]:
                if t == "LargeTrace":
                    t = self.trace[key][self.frames[key][-1]]
                else:
                    t = self.trace[key][self.frames[key][0]]
                x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))

                r, c = skimage.draw.polygon(np.rint(y).astype(int), np.rint(x).astype(int), self.frame_shape)
                mask = np.zeros(self.frame_shape, np.float32)
                mask[r, c] = 1
                target.append(mask)
            elif t in ["LargePred", "SmallPred"]:
                # FIXME: Something off with naming convention
                if t == "LargePred":
                    mask = self.get_pred(key, self.frames[key][-1], phase='esv')
                else:
                    mask = self.get_pred(key, self.frames[key][0], phase='edv')

                target.append(mask)
            else:
                if self.split == "CLINICAL_TEST" or self.split == "EXTERNAL_TEST":
                    target.append(np.float32(0))
                else:
                    target.append(np.float32(self.outcome[index][self.header.index(t)]))

        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]

        return target

    def __len__(self):
        return len(self.fnames)

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)

    def get_pred(self, video_name, frame, phase='edv'):
        if self.pred_dir is None:
            return None
        else:
            filename = os.path.join(self.root, self.pred_dir,
                                    "{}_{}_{}.png".format(video_name.strip('.avi'), frame, phase))
            print(filename)
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            return img.astype(np.float32) / 255.0

class TestData():
    def run_test(self, data_dir, output_dir, pred_dir, num_workers=4):
        os.makedirs(output_dir, exist_ok=True)

        for split in ["test"]:
            dataset = Echo(root=data_dir, pred_dir=pred_dir, split=split,
                           target_type=["LargeTrace", "SmallTrace", "LargePred", "SmallPred"])
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=num_workers,
                                                     shuffle=False, drop_last=False)

            asd, sen = self.run_epoch(dataloader)

            # 计算 ASD 和 SEN 的均值
            asd_mean = np.mean(asd)
            sen_mean = np.mean(sen)

            with open(os.path.join(output_dir, "{}_asd.csv".format(split)), "w") as f:
                f.write("Filename, ASD\n")
                for filename, asd_value in zip(dataset.fnames, asd):
                    f.write("{}, {}\n".format(filename, asd_value))

                # 写入 ASD 均值
                f.write("Mean ASD, {}\n".format(asd_mean))

            with open(os.path.join(output_dir, "{}_sen.csv".format(split)), "w") as f:
                f.write("Filename, SEN\n")
                for filename, sen_value in zip(dataset.fnames, sen):
                    f.write("{}, {}\n".format(filename, sen_value))

                # 写入 SEN 均值
                f.write("Mean SEN, {}\n".format(sen_mean))

    def run_epoch(self, dataloader):
        asd_list = []
        sen_list = []

        for large_trace, small_trace, large_pred, small_pred in tqdm.tqdm(dataloader):
            # 计算 ASD
            asd = self.compute_asd(large_trace, large_pred)  # 计算大区域的 ASD
            asd_list.append(asd)

            # 计算 SEN
            sen = self.compute_sen(large_trace, large_pred)  # 计算大区域的 SEN
            sen_list.append(sen)

        return asd_list, sen_list

    def compute_asd(self, true_mask, pred_mask):
        """计算平均表面距离 (ASD)"""
        # 将 true_mask 和 pred_mask 转为二值化图像
        true_mask = true_mask.squeeze().cpu().numpy()
        pred_mask = pred_mask.squeeze().cpu().numpy()

        # 使用 skimage.measure.label 将预测结果和真实标注提取为标签
        true_labels = measure.label(true_mask)
        pred_labels = measure.label(pred_mask)

        # 找到预测和真实标注之间的边界
        true_contours = measure.find_contours(true_mask, 0.5)
        pred_contours = measure.find_contours(pred_mask, 0.5)

        distances = []
        # 计算每个预测边界点到真实边界的距离
        for pred_contour in pred_contours:
            for point in pred_contour:
                x, y = point[0], point[1]
                # 获取最近的真实边界点
                nearest_dist = self.nearest_distance(x, y, true_contours)
                distances.append(nearest_dist)
        # 计算 ASD
        return np.mean(distances)

    def nearest_distance(self, x, y, contours):
        """计算一个点到所有轮廓点的最近距离"""
        min_dist = float('inf')
        for contour in contours:
            for point in contour:
                dist = np.sqrt((x - point[0])**2 + (y - point[1])**2)
                min_dist = min(min_dist, dist)
        return min_dist

    def compute_sen(self, true_mask, pred_mask):
        """计算敏感度 (SEN)"""
        true_mask = true_mask.squeeze().cpu().numpy()
        pred_mask = pred_mask.squeeze().cpu().numpy()

        # 计算 confusion matrix
        tn, fp, fn, tp = confusion_matrix(true_mask.flatten(), pred_mask.flatten()).ravel()

        # 计算敏感度
        if tp + fn == 0:  # 防止除以零的情况
            return 1.0
        return tp / (tp + fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process the paths for data, prediction, and output directories.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the EchoNet Dynamic Dataset directory.')
    parser.add_argument('--prediction_dir', type=str, required=True, help='Path to the directory where predictions will be stored.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory where results will be saved.')

    args = parser.parse_args()

    test_model = TestData()
    test_model.run_test(data_dir=args.data_dir, pred_dir=args.prediction_dir, output_dir=args.output_dir)