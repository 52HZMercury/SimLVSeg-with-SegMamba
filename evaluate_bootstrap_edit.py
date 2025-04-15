import argparse  # 用于处理命令行参数
import pytorch_lightning as pl  # PyTorch Lightning库，用于高层次的PyTorch操作
import torch  # PyTorch库，用于张量操作
import numpy as np  # NumPy库，用于数值计算
import tqdm  # 用于循环的进度条显示
import matplotlib.pyplot as plt  # 用于绘制图表的库
import os  # 用于操作系统交互（例如文件和目录操作）
import segmentation_models_pytorch as smp  # 基于PyTorch的图像分割模型库
import segmentation_models_pytorch.utils as smp_utils  # 图像分割模型的工具函数
import math  # 标准数学库
import yaml  # 用于解析YAML文件的库
from skimage import measure  # scikit-image库的一部分，用于图像处理
from sklearn.metrics import confusion_matrix  # 用于计算分类任务的混淆矩阵

import collections  # 提供替代的容器类型，如defaultdict
import skimage.draw  # scikit-image库的一部分，用于在图像上绘制形状
import torchvision  # 用于计算机视觉工具和数据集的库
import cv2  # OpenCV库，用于计算机视觉任务
import pandas  # 用于数据处理的库
from medpy.metric.binary import assd as medpy_assd
from medpy.metric.binary import hd95 as medpy_hd95

# 导入自定义工具包
from simlvseg.utils import defaultdict_of_lists

# 定义Echo类，继承自torchvision的VisionDataset
class Echo(torchvision.datasets.VisionDataset):
    # 初始化方法，接收多个参数
    def __init__(
            self,
            root,
            pred_dir=None,
            split="test",
            target_type="EF",
            external_test_location=None,
            frame_shape=(128, 128)
    ):
        super().__init__(root)  # 调用父类的初始化方法

        self.frame_shape = frame_shape  # 帧的形状
        self.split = split.upper()  # 将数据集划分方式转换为大写字母
        if not isinstance(target_type, list):  # 如果目标类型不是列表，则转换为列表
            target_type = [target_type]
        self.target_type = target_type  # 目标类型列表
        self.pred_dir = pred_dir  # 预测结果存放目录
        self.external_test_location = external_test_location  # 外部测试数据的位置

        self.fnames, self.outcome = [], []  # 初始化文件名和结果的空列表

        # 如果是外部测试数据集
        if self.split == "EXTERNAL_TEST":
            self.fnames = sorted(os.listdir(self.external_test_location))  # 列出所有文件名
        else:
            with open(os.path.join(self.root, "FileList.csv")) as f:  # 打开文件列表CSV文件
                data = pandas.read_csv(f)  # 使用pandas读取CSV数据
            data["Split"].map(lambda x: x.upper())  # 将分割字段转换为大写字母

            if self.split != "ALL":  # 如果划分不是“ALL”，则只选取该分割的数据
                data = data[data["Split"] == self.split]

            self.header = data.columns.tolist()  # 获取表头
            self.fnames = data["FileName"].tolist()  # 获取文件名列
            self.fnames = [fn + ".avi" for fn in self.fnames if os.path.splitext(fn)[1] == ""]  # 添加.avi后缀
            self.outcome = data.values.tolist()  # 获取数据的结果列

            missing = set(self.fnames) - set(os.listdir(os.path.join(self.root, "Videos")))  # 检查缺失的视频文件
            if len(missing) != 0:  # 如果有缺失文件，抛出错误
                print("{} videos could not be found in {}:".format(len(missing), os.path.join(self.root, "Videos")))
                for f in sorted(missing):
                    print("\t", f)
                raise FileNotFoundError(os.path.join(self.root, "Videos", sorted(missing)[0]))

            self.frames = collections.defaultdict(list)  # 使用defaultdict初始化帧列表
            self.trace = collections.defaultdict(defaultdict_of_lists)  # 使用defaultdict初始化追踪数据

            # 读取VolumeTracings.csv文件
            with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
                header = f.readline().strip().split(",")  # 读取并解析文件头
                if(header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]):  # 确保文件头与预期一致
                    for line in f:  # 遍历每一行数据
                        filename, x1, y1, x2, y2, frame = line.strip().split(',')
                        x1 = float(x1)  # 将坐标转换为浮动类型
                        y1 = float(y1)
                        x2 = float(x2)
                        y2 = float(y2)
                        frame = int(frame)  # 将帧号转换为整数
                        if frame not in self.trace[filename]:  # 如果该帧不存在于追踪字典中
                            self.frames[filename].append(frame)  # 添加该帧到帧列表
                        self.trace[filename][frame].append((x1, y1, x2, y2))  # 在追踪字典中添加坐标
                if (header == ["FileName", "X", "Y", "Frame"]):  # 确保文件头与预期一致
                    for line in f:  # 遍历每一行数据
                        filename, x, y, frame = line.strip().split(',')
                        x = float(x)  # 将坐标转换为浮动类型
                        y = float(y)
                        frame = int(frame)  # 将帧号转换为整数
                        if frame not in self.trace[filename]:  # 如果该帧不存在于追踪字典中
                            self.frames[filename].append(frame)  # 添加该帧到帧列表
                        self.trace[filename][frame].append((x, y))  # 在追踪字典中添加坐标

            # 将追踪数据从列表转换为NumPy数组
            for filename in self.frames:
                for frame in self.frames[filename]:
                    self.trace[filename][frame] = np.array(self.trace[filename][frame])

            # 根据是否有足够的帧，筛选文件名和结果
            keep = [len(self.frames[f]) >= 2 for f in self.fnames]
            self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
            self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]

    # 获取数据项（根据索引）
    def __getitem__(self, index):
        # 根据分割情况获取视频路径
        if self.split == "EXTERNAL_TEST":
            video = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "CLINICAL_TEST":
            video = os.path.join(self.root, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video = os.path.join(self.root, "Videos", self.fnames[index])

        target = []  # 初始化目标列表
        # 根据目标类型获取目标数据
        for t in self.target_type:
            key = self.fnames[index]
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "LargeIndex":
                target.append(int(self.frames[key][-1]))
            elif t == "SmallIndex":
                target.append(int(self.frames[key][0]))
            elif t in ["LargeTrace", "SmallTrace"]:
                if t == "LargeTrace":
                    t = self.trace[key][self.frames[key][-1]]
                else:
                    t = self.trace[key][self.frames[key][0]]

                # x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                # x = np.concatenate((x1[1:], np.flip(x2[1:])))
                # y = np.concatenate((y1[1:], np.flip(y2[1:])))

                if t.shape[1] == 4:
                    x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                    x = np.concatenate((x1[1:], np.flip(x2[1:])))
                    y = np.concatenate((y1[1:], np.flip(y2[1:])))
                else:
                    assert t.shape[1] == 2
                    x, y = t[:, 0], t[:, 1]

                r, c = skimage.draw.polygon(np.rint(y).astype(int), np.rint(x).astype(int), self.frame_shape)
                mask = np.zeros(self.frame_shape, np.float32)
                mask[r, c] = 1  # 创建分割掩膜
                target.append(mask)
            elif t in ["LargePred", "SmallPred"]:
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

        # 如果目标列表不为空，则返回目标数据
        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]

        return target

    # 返回数据集的大小
    def __len__(self):
        return len(self.fnames)

    # 输出数据集的附加信息
    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)

    # 获取预测结果
    def get_pred(self, video_name, frame, phase='edv'):
        if self.pred_dir is None:
            return None
        else:
            filename = os.path.join(self.pred_dir,
                                    "{}_{}_{}.png".format(video_name.strip('.avi'), frame, phase))
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # 读取预测的图像
            return img.astype(np.float32) / 255.0  # 归一化处理


# 下面是TestData类的定义，负责执行测试任务
class TestData():
    # 计算均值和标准差
    def mean_and_std(self, values):

        mean = np.mean(values)  # 均值的期望
        std = np.std(values)  # 均值的标准差（即标准误差）

        return mean, std

    # 执行测试
    def run_test(self, data_dir, output_dir, pred_dir, num_workers=1):
        os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

        for split in ["test"]:
            dataset = Echo(root=data_dir, pred_dir=pred_dir, split=split,
                           target_type=["LargeTrace", "SmallTrace", "LargePred", "SmallPred"])
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=num_workers,
                                                     shuffle=False, drop_last=False)

            # 执行一轮评估
            assd, hd95, sen, large_inter, large_union, small_inter, small_union = self.run_epoch(dataloader)

            # 计算Dice系数
            overall_dice = 2 * (large_inter + small_inter) / (large_union + large_inter + small_union + small_inter + 1e-7)
            large_dice = 2 * large_inter / (large_union + large_inter + 1e-7)
            small_dice = 2 * small_inter / (small_union + small_inter + 1e-7)

            # 计算IOU
            overall_iou = (large_inter + small_inter) / (large_union + small_union + 1e-7)
            large_iou = large_inter / (large_union + 1e-7)
            small_iou = small_inter / (small_union + 1e-7)

            # 计算均值和标准差
            hd95_mean, hd95_std = self.mean_and_std(hd95)
            assd_mean, assd_std = self.mean_and_std(assd)
            sen_mean, sen_std = self.mean_and_std(sen)
            overall_dice_mean, overall_dice_std = self.mean_and_std(overall_dice)
            large_dice_mean, large_dice_std = self.mean_and_std(large_dice)
            small_dice_mean, small_dice_std = self.mean_and_std(small_dice)
            overall_iou_mean, overall_iou_std = self.mean_and_std(overall_iou)
            large_iou_mean, large_iou_std = self.mean_and_std(large_iou)
            small_iou_mean, small_iou_std = self.mean_and_std(small_iou)

            # 将结果保存到CSV文件中
            # 修改输出文件
            with open(os.path.join(output_dir, "{}_assd.csv".format(split)), "w") as f:
                f.write("Filename, ASSD\n")
                for filename, assd_value in zip(dataset.fnames, assd):
                    f.write("{}, {}\n".format(filename, assd_value))
                f.write("Mean ASSD, {}\n".format(assd_mean))
                f.write("Standard Deviation ASSD, {}\n".format(assd_std))

            with open(os.path.join(output_dir, "{}_hd95.csv".format(split)), "w") as f:
                f.write("Filename, HD95\n")
                for filename, hd95_value in zip(dataset.fnames, hd95):
                    f.write("{}, {}\n".format(filename, hd95_value))
                f.write("Mean HD95, {}\n".format(hd95_mean))
                f.write("Standard Deviation HD95, {}\n".format(hd95_std))

            with open(os.path.join(output_dir, "{}_sen.csv".format(split)), "w") as f:
                    f.write("Filename, SEN\n")
                    for filename, sen_value in zip(dataset.fnames, sen):
                        f.write("{}, {}\n".format(filename, sen_value))
                    f.write("Mean SEN, {}\n".format(sen_mean))
                    f.write("Standard Deviation SEN, {}\n".format(sen_std))

            with open(os.path.join(output_dir, "{}_dice.csv".format(split)), "w") as g:
                g.write("Filename, Overall, Large, Small\n")
                for (filename, overall, large, small) in zip(dataset.fnames, overall_dice, large_dice, small_dice):
                    g.write("{},{},{},{}\n".format(filename, overall, large, small))
                g.write("Mean Overall Dice, {}\n".format(overall_dice_mean))
                g.write("Standard Deviation Overall Dice, {}\n".format(overall_dice_std))
                g.write("Mean Large Dice, {}\n".format(large_dice_mean))
                g.write("Standard Deviation Large Dice, {}\n".format(large_dice_std))
                g.write("Mean Small Dice, {}\n".format(small_dice_mean))
                g.write("Standard Deviation Small Dice, {}\n".format(small_dice_std))

                # 新增IOU结果保存
                with open(os.path.join(output_dir, "{}_iou.csv".format(split)), "w") as g:
                    g.write("Filename, Overall, Large, Small\n")
                    for (filename, overall, large, small) in zip(dataset.fnames, overall_iou, large_iou, small_iou):
                        g.write("{},{},{},{}\n".format(filename, overall, large, small))
                    g.write("Mean Overall IOU, {}\n".format(overall_iou_mean))
                    g.write("Standard Deviation Overall IOU, {}\n".format(overall_iou_std))
                    g.write("Mean Large IOU, {}\n".format(large_iou_mean))
                    g.write("Standard Deviation Large IOU, {}\n".format(large_iou_std))
                    g.write("Mean Small IOU, {}\n".format(small_iou_mean))
                    g.write("Standard Deviation Small IOU, {}\n".format(small_iou_std))

                # 记录log日志
                with open(os.path.join(output_dir, "log.csv"), "w") as f:
                    # f.write("{} Overall Dice (mean - std): {:.4f} - {:.4f}\n".format(split, overall_dice_mean, overall_dice_std))
                    # f.write("{} Overall IOU (mean - std): {:.4f} - {:.4f}\n".format(split, overall_iou_mean, overall_iou_std))
                    f.write("{} ASSD (mean - std): {:.4f} - {:.4f}\n".format(split, assd_mean, assd_std))
                    f.write("{} HD95 (mean - std): {:.4f} - {:.4f}\n".format(split, hd95_mean, hd95_std))
                    f.write("{} SEN (mean - std): {:.4f} - {:.4f}\n".format(split, sen_mean, sen_std))


                    f.flush()

    # 执行一轮数据加载
    def run_epoch(self, dataloader):
        assd_list = []  # 改为ASSD
        hd95_list = []  # 新增HD95
        sen_list = []  # 用于存储敏感度值的列表

        large_inter = 0  # 初始化大型区域交集
        large_union = 0  # 初始化大型区域并集
        small_inter = 0  # 初始化小型区域交集
        small_union = 0  # 初始化小型区域并集
        large_inter_list = []  # 用于存储大型区域交集的详细列表
        large_union_list = []  # 用于存储大型区域并集的详细列表
        small_inter_list = []  # 用于存储小型区域交集的详细列表
        small_union_list = []  # 用于存储小型区域并集的详细列表

        # 遍历数据加载器
        for large_trace, small_trace, large_pred, small_pred in tqdm.tqdm(dataloader):
            # 改为计算ASSD和HD95
            assd_large = self.compute_assd(large_trace, large_pred)
            assd_small = self.compute_assd(small_trace, small_pred)
            hd95_large = self.compute_hd95(large_trace, large_pred)
            hd95_small = self.compute_hd95(small_trace, small_pred)
            sen_large = self.compute_sen(large_trace, large_pred)
            sen_small = self.compute_sen(small_trace, small_pred)

            assd = (assd_large + assd_small) / 2
            hd95 = (hd95_large + hd95_small) / 2
            sen = (sen_large + sen_small) / 2


            assd_list.append(assd)
            hd95_list.append(hd95)
            sen_list.append(sen)

            # 计算交集和并集的大小
            large_inter += np.logical_and(large_pred.detach().cpu().numpy() > 0., large_trace.detach().cpu().numpy() > 0.).sum()
            large_union += np.logical_or(large_pred.detach().cpu().numpy() > 0., large_trace.detach().cpu().numpy() > 0.).sum()
            small_inter += np.logical_and(small_pred.detach().cpu().numpy() > 0., small_trace.detach().cpu().numpy() > 0.).sum()
            small_union += np.logical_or(small_pred.detach().cpu().numpy() > 0., small_trace.detach().cpu().numpy() > 0.).sum()

            # 保存详细的交集和并集信息
            large_inter_list.extend(np.logical_and(large_pred.detach().cpu().numpy() > 0., large_trace.detach().cpu().numpy() > 0.).sum((1, 2)))
            large_union_list.extend(np.logical_or(large_pred.detach().cpu().numpy() > 0., large_trace.detach().cpu().numpy() > 0.).sum((1, 2)))
            small_inter_list.extend(np.logical_and(small_pred.detach().cpu().numpy() > 0., small_trace.detach().cpu().numpy() > 0.).sum((1, 2)))
            small_union_list.extend(np.logical_or(small_pred.detach().cpu().numpy() > 0., small_trace.detach().cpu().numpy() > 0.).sum((1, 2)))

        return assd_list, hd95_list, sen_list, np.array(large_inter_list), np.array(large_union_list), np.array(small_inter_list), np.array(small_union_list)


    # 计算ASD (平均表面距离)
    def compute_asd(self, true_mask, pred_mask):
        true_mask = true_mask.squeeze().cpu().numpy()  # 转换为numpy数组
        pred_mask = pred_mask.squeeze().cpu().numpy()

        true_labels = measure.label(true_mask)  # 标签化真值掩膜
        pred_labels = measure.label(pred_mask)  # 标签化预测值掩膜

        true_contours = measure.find_contours(true_mask, 0.5)  # 寻找真值掩膜的轮廓
        pred_contours = measure.find_contours(pred_mask, 0.5)  # 寻找预测掩膜的轮廓

        distances = []  # 初始化距离列表
        # 计算每个预测轮廓的最近距离
        for pred_contour in pred_contours:
            for point in pred_contour:
                x, y = point[0], point[1]
                nearest_dist = self.nearest_distance(x, y, true_contours)  # 计算到真值轮廓的最近距离
                distances.append(nearest_dist)  # 保存该距离
        return np.mean(distances)  # 返回平均距离

    # 计算点到轮廓的最近距离
    def nearest_distance(self, x, y, contours):
        min_dist = float('inf')  # 初始化最小距离为正无穷
        for contour in contours:
            for point in contour:
                dist = np.sqrt((x - point[0])**2 + (y - point[1])**2)  # 计算欧几里得距离
                min_dist = min(min_dist, dist)  # 更新最小距离
        return min_dist  # 返回最小距离

    # 计算敏感度（True Positive Rate）
    def compute_sen(self, true_mask, pred_mask):
        true_mask = true_mask.squeeze().cpu().numpy()
        pred_mask = pred_mask.squeeze().cpu().numpy()

        tn, fp, fn, tp = confusion_matrix(true_mask.flatten(), pred_mask.flatten()).ravel()

        if tp + fn == 0:  # 避免除以零
            return 1.0
        return tp / (tp + fn)  # 返回敏感度值

    def compute_assd(self, true_mask, pred_mask):
        """
            Calculate the Average Symmetric Surface Distance (ASSD) between the predicted and target segmentation
            using medpy's assd.
            """
        pred = (pred_mask > 0.5).cpu().numpy()
        target = (true_mask > 0.5).cpu().numpy()

        # return medpy_assd(pred, target, voxelspacing=[1.0, 1.0, 1.0])
        return medpy_assd(target.squeeze(), pred.squeeze(), voxelspacing=[1.0, 1.0])

    def compute_hd95(self, true_mask, pred_mask):
        """
            Calculate the Average Symmetric Surface Distance (ASSD) between the predicted and target segmentation
            using medpy's assd.
            """
        pred = (pred_mask > 0.5).cpu().numpy()
        target = (true_mask > 0.5).cpu().numpy()

        # return medpy_hd95(pred, target, voxelspacing=[1.0, 1.0, 1.0])
        return medpy_hd95(target.squeeze(), pred.squeeze(), voxelspacing=[1.0, 1.0])


# 主程序入口
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="处理数据、预测结果和输出目录的路径")
    parser.add_argument('--data_dir', type=str, required=True, help='EchoNet Dynamic Dataset的路径')
    parser.add_argument('--prediction_dir', type=str, required=True, help='预测结果存放的目录路径')
    parser.add_argument('--output_dir', type=str, required=True, help='结果保存的输出目录路径')

    args = parser.parse_args()

    test_model = TestData()  # 创建TestData对象
    test_model.run_test(data_dir=args.data_dir, pred_dir=args.prediction_dir, output_dir=args.output_dir)  # 执行测试
