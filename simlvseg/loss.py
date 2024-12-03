import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegLoss(nn.Module):
    def __init__(
            self,
            loss_type='dice'
    ):
        super().__init__()

        if not isinstance(loss_type, list):
            loss_type = [loss_type]

        for l in loss_type:
            if l not in ['bce', 'sbce', 'dice', 'mse', 'hccmse', 'hccdice', 'focal',
                         'tversky', 'jaccard', 'dice+Jaccard', 'onlyhcc']:
                raise ValueError(f'Loss type {l} is not recognized ...')

        self.bce_loss = smp.losses.SoftBCEWithLogitsLoss()
        self.dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.focal_loss = smp.losses.FocalLoss(mode='binary')
        self.tversky_loss = smp.losses.TverskyLoss(mode='binary', alpha=0.7, beta=0.3)
        self.jaccard_loss = smp.losses.JaccardLoss(mode='binary')
        self.sbce = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.1)

        self.mse_loss = nn.MSELoss()

        self.loss_type = loss_type

    def pearson(self, preds, labels):
        """
        计算皮尔逊相关系数损失
        preds: 模型预测值 (batch_size, c, h, w, t)
        labels: 真实值 (batch_size, c, h, w, t)
        """
        # 对通道进行平均处理，将形状从 [b, c, h, w, t] 变为 [b, h, w, t]
        preds_mean = preds.mean(dim=1)  # 在通道维度上取均值
        labels_mean = labels.mean(dim=1)

        # 计算每个时间帧的空间标准差，得到形状为 [b, t] 的心动周期特征
        preds_cycle = preds_mean.std(dim=(1, 2))  # 在 h 和 w 维度上计算标准差
        labels_cycle = labels_mean.std(dim=(1, 2))

        # # 中心化
        signal1_centered = preds_cycle - preds_cycle.mean()
        signal2_centered = labels_cycle - labels_cycle.mean()

        # 计算皮尔逊相关系数
        numerator = torch.sum(signal1_centered * signal2_centered)
        denominator = torch.sqrt(torch.sum(signal1_centered ** 2) * torch.sum(signal2_centered ** 2))
        correlation = numerator / (denominator + 1e-8)  # 加小值避免除以零

        # 损失值 (最小化皮尔逊相关系数和1的均方误差)
        # print(correlation)
        loss = F.mse_loss(correlation, torch.tensor(1.0).to(correlation.device))
        return loss

    def forward(self, preds, labels,):
        loss = 0.

        if 'bce' in self.loss_type:
            loss += self.bce_loss(preds, labels)

        if 'dice' in self.loss_type:
            loss += self.dice_loss(preds, labels)

        if 'focal' in self.loss_type:
            loss += self.focal_loss(preds, labels)

        if 'tversky' in self.loss_type:
            loss += self.tversky_loss(preds, labels)

        if 'sbce' in self.loss_type:
            loss += self.sbce(preds, labels)

        if 'mse' in self.loss_type:
            loss += self.mse_loss(preds, labels)

        if 'jaccard' in self.loss_type:
            loss += self.jaccard_loss(preds, labels)

        if 'dice+Jaccard' in self.loss_type:
            loss += 0.5 * self.dice_loss(preds, labels) + 0.5 * self.jaccard_loss(preds, labels)

        if 'hccmse' in self.loss_type:
            loss += (self.mse_loss(preds, labels) + 0.1 * self.mse_loss(preds, labels))

        if 'hccdice' in self.loss_type:
            loss += (self.dice_loss(preds, labels) + 0.1 * self.pearson(preds, labels))

        if 'onlyhcc' in self.loss_type:
            loss += self.spearman(preds, labels)
            # loss += self.pearson(preds, labels)

        return loss
