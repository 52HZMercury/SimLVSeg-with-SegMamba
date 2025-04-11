import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils
import torch
import numpy as np
from .model import get_model
from .loss import SegLoss
from .utils import get_crop_from_coors
from medpy.metric.binary import assd as medpy_assd
from medpy.metric.binary import hd95 as medpy_hd95

# 计算ASSD（Average Symmetric Surface Distance）并返回
def calculate_assd(pred, target):
    """
    Calculate the Average Symmetric Surface Distance (ASSD) between the predicted and target segmentation
    using medpy's assd.
    """
    pred = pred > 0.5  # 假设0.5是阈值
    target = target > 0.5

    # 使用medpy的ASSD计算
    return medpy_assd(target.cpu().numpy(), pred.cpu().numpy(), voxelspacing=[1.0, 1.0, 1.0])


def calculate_hd95(pred, target):
    """
    Calculate the Average Symmetric Surface Distance (ASSD) between the predicted and target segmentation
    using medpy's assd.
    """
    pred = pred > 0.5  # 假设0.5是阈值
    target = target > 0.5

    # 使用medpy的hd95计算
    return medpy_hd95(target.cpu().numpy(), pred.cpu().numpy(), voxelspacing=[1.0, 1.0, 1.0])



# 计算敏感度 (Sensitivity)
def calculate_sensitivity(pred, target):
    """
    Calculate the sensitivity (recall) of the segmentation: TP / (TP + FN).
    """
    pred = pred > 0.5  # 阈值
    target = target > 0.5

    tp = ((pred == 1) & (target == 1)).sum().item()  # 真正例
    fn = ((pred == 0) & (target == 1)).sum().item()  # 假负例

    sensitivity = tp / (tp + fn + 1e-6)  # 加上一个小值避免除零
    return sensitivity


class BaseModule(pl.LightningModule):
    def preprocess_batch_imgs(self, imgs):
        raise NotImplementedError

    def postprocess_batch_preds_and_targets(self, preds, targets):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def calculate_metrics(self, set_name, preds, labels):
        # Calculate the metrics
        metrics = [[name, fn(preds, labels)] for name, fn in self.metrics.items()]

        # Print the metrics on the terminal
        for name, value in metrics:
            self.log(f"{set_name}_{name}", value, prog_bar=True, logger=True)

    def val_test_epoch_end(self, set_name, step_outputs):
        preds = []
        labels = []

        for output in step_outputs:
            preds.append(output['batch_preds'])
            labels.append(output['batch_labels'])

        preds = torch.cat(preds)
        labels = torch.cat(labels)

        loss = self.criterion(preds, labels)

        self.log(f"{set_name}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.calculate_metrics(set_name, preds, labels)

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch

        imgs = self.preprocess_batch_imgs(imgs)

        preds = self.forward(imgs)

        preds, labels = self.postprocess_batch_preds_and_targets(preds, targets)
        # preds, labels = self.postprocess_batch_preds_and_targets_camus(preds, targets)

        loss = self.criterion(preds, labels)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.calculate_metrics('train', preds, labels)

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch

        imgs = self.preprocess_batch_imgs(imgs)

        preds = self.forward(imgs)

        preds, labels = self.postprocess_batch_preds_and_targets(preds, targets)
        # preds, labels = self.postprocess_batch_preds_and_targets_camus(preds, targets)

        # 将输出保存到实例属性中
        if not hasattr(self, 'validation_step_outputs'):
            self.validation_step_outputs = []
        self.validation_step_outputs.append({'batch_preds': preds, 'batch_labels': labels})

        return {'batch_preds': preds, 'batch_labels': labels}

    def on_validation_epoch_end(self):
        # 使用保存的输出
        self.val_test_epoch_end('val', self.validation_step_outputs)
        # 清空保存的输出
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        imgs, targets = batch

        imgs = self.preprocess_batch_imgs(imgs)

        preds = self.forward(imgs)

        preds, labels = self.postprocess_batch_preds_and_targets(preds, targets)
        # preds, labels = self.postprocess_batch_preds_and_targets_camus_test(preds, targets)

        # 计算敏感度、ASSD和HD95
        sensitivity = calculate_sensitivity(preds[0], labels[0])
        assd = calculate_assd(preds[0], labels[0])
        hd95 = calculate_hd95(labels[0], preds[0])

        # 计算标准指标（如 dsc, iou, dice_loss）
        dsc = self.metrics['dsc'](preds, labels)
        iou = self.metrics['iou'](preds, labels)

        # 将所有结果保存到实例属性中
        if not hasattr(self, 'test_step_outputs'):
            self.test_step_outputs = []
        self.test_step_outputs.append({
            'batch_preds': preds,
            'batch_labels': labels,
            'sensitivity': sensitivity,
            'assd': assd,
            'hd95': hd95,
            'test_dsc': dsc,
            'test_iou': iou,
        })

        return {
            'batch_preds': preds,
            'batch_labels': labels,
            'sensitivity': sensitivity,
            'assd': assd,
            'hd95': hd95,
            'test_dsc': dsc,
            'test_iou': iou,
        }

    def on_test_epoch_end(self):
        # 提取所有的指标并计算均值和标准差
        sensitivities = [x['sensitivity'] for x in self.test_step_outputs]
        assds = [x['assd'] for x in self.test_step_outputs]  # 这里改成ASSD
        hd95s = [x['hd95'] for x in self.test_step_outputs]
        dscs = [x['test_dsc'] for x in self.test_step_outputs]
        ious = [x['test_iou'] for x in self.test_step_outputs]

        # 计算均值和标准差
        avg_sensitivity = torch.tensor(sensitivities).mean()
        std_sensitivity = torch.tensor(sensitivities).std()
        avg_assd = torch.tensor(assds).mean()  # 使用ASSD替代ASD
        std_assd = torch.tensor(assds).std()  # 使用ASSD替代ASD
        avg_hd95 = torch.tensor(hd95s).mean()
        std_hd95 = torch.tensor(hd95s).std()
        avg_dsc = torch.tensor(dscs).mean()
        std_dsc = torch.tensor(dscs).std()
        avg_iou = torch.tensor(ious).mean()
        std_iou = torch.tensor(ious).std()

        # 记录均值和标准差
        self.log("avg_test_sensitivity", avg_sensitivity)
        self.log("std_test_sensitivity", std_sensitivity)

        self.log("avg_test_assd", avg_assd)  # 使用ASSD替代ASD
        self.log("std_test_assd", std_assd)  # 使用ASSD替代ASD

        self.log("avg_test_hd95", avg_hd95)
        self.log("std_test_hd95", std_hd95)

        self.log("avg_test_dsc", avg_dsc)
        self.log("std_test_dsc", std_dsc)

        self.log("avg_test_iou", avg_iou)
        self.log("std_test_iou", std_iou)

        # 清空保存的测试输出
        self.test_step_outputs.clear()


class SegModule(BaseModule):
    def __init__(self,
                 encoder_name,
                 weights=None,
                 pretrained_type='encoder',
                 img_size=None,
                 # loss_type='hccdice',
                 # loss_type='dice',
                 # loss_type='jaccard',
                 # loss_type='hccdie',
                 # loss_type='tversky',
                 # loss_type='focal'
                 # loss_type='dice+jaccard'
                 # loss_type='hccmse'
                 loss_type='dice+jaccard+focal',
                 # loss_type='dice+jaccard+tversky+focal'
                 ):

        super().__init__()

        self.model = get_model(encoder_name, weights, pretrained_type, img_size)

        self.criterion = SegLoss(loss_type)
        self.metrics = {
            'dsc': smp_utils.metrics.Fscore(activation='sigmoid'),
            'iou': smp.utils.metrics.IoU(activation='sigmoid'),
            'dice_loss': smp.losses.DiceLoss(mode='binary', from_logits=True),
        }

    def preprocess_batch_imgs(self, imgs):
        super_images, videos = imgs
        return super_images

    def postprocess_batch_preds_and_targets(self, preds, targets):
        out_preds = []
        out_labels = []

        if len(preds) != len(targets['filename']):
            raise ValueError("The number of predictions and the number of targets are different ...")

        for i in range(len(preds)):
            pred = preds[i]

            trace_mask = targets['trace_mask'][i][None, :]
            pos_trace_frame = self.__get_pos_frame(targets['pos_trace_frame'], i)

            # Change from the channel-first into the channel-last format
            pred = pred.permute((1, 2, 0))

            pred_trace = get_crop_from_coors(pred, pos_trace_frame)

            # Change from the channel-last into the channel-first format
            pred_trace = pred_trace.permute((2, 0, 1))

            out_preds.extend([pred_trace[None, :]])
            out_labels.extend([trace_mask])

        out_preds = torch.cat(out_preds)[:, :, :112, :112].contiguous()
        out_labels = torch.cat(out_labels)[:, :, :112, :112].contiguous()

        return out_preds, out_labels

    def configure_optimizers(self):
        # AdamW
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=4e-4,
            weight_decay=1e-5, amsgrad=True,
        )


        # SGD
        # optimizer = torch.optim.SGD(
        #     self.parameters(), lr=0.1,  # 设置学习率
        #     momentum=0.9,  # 设置动量
        #     weight_decay=1e-5  # 设置权重衰减
        # )

        # MultiStepLR
        # 默认
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #   optimizer, milestones=[45, 60], gamma=0.1,
        # )

        # 35 50
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[35, 50], gamma=0.1,
        )

        # 定义LinearLR调度器，线性增加
        # scheduler = LinearLR(
        #     optimizer,
        #     end_lr=1e-2,  # 最终学习率
        #     num_iter=60,  # 总迭代次数
        # )

        # 定义ExponentialLR调度器
        # scheduler = ExponentialLR(
        #     optimizer,
        #     end_lr=1e-2,  # 最终学习率
        #     num_iter=60,  # 总迭代次数
        # )

        # 定义WarmupCosineSchedule调度器
        # scheduler = WarmupCosineSchedule(
        #     optimizer,
        #     warmup_steps=5,  # 线性warmup步骤
        #     t_total=60,  # 总训练步骤
        #     cycles=0.5,  # 余弦周期
        # )

        # 定义LinearWarmupCosineAnnealingLR调度器
        # scheduler = LinearWarmupCosineAnnealingLR(
        #     optimizer,
        #     warmup_epochs=5,  # 热身epoch数
        #     max_epochs=60,  # 总训练epoch数
        #     warmup_start_lr=1e-6,  # 热身开始的学习率
        #     eta_min=1e-6,  # 最小学习率
        # )

        return [optimizer], [scheduler]

