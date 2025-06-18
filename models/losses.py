# -*- encoding: utf-8 -*-
from __future__ import annotations, print_function, absolute_import
from typing import List, Optional, Union

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        # (batch_size, num_labels)
        pred = pred.log_softmax(dim=self.dim)
        # target 是(batch_size)
        with th.no_grad():
            # true_dist = pred.data.clone()
            true_dist = th.zeros_like(pred)
            # 把所有值填充为self.smoothing / (self.cls - 1)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return th.mean(th.sum(-true_dist * pred, dim=self.dim))

class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, input: th.Tensor, target: th.Tensor) -> th.Tensor:
        logprobs = F.log_softmax(input, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: Union[List[float], float, th.Tensor],
                 gamma: Optional[int] = 2,
                 with_logits: Optional[bool] = True):
        """

        :param alpha: 每个类别的权重
        :param gamma:
        :param with_logits: 是否经过softmax或者sigmoid
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if th.is_tensor(alpha):
            self.alpha = alpha
        else:
            self.alpha = th.FloatTensor([alpha]) if isinstance(alpha, float) else th.FloatTensor(alpha)
        self.smooth = 1e-8
        self.with_logits = with_logits

    def _binary_class(self, input, target):
        prob = th.sigmoid(input) if self.with_logits else input
        prob += self.smooth
        alpha = self.alpha.to(target.device)
        loss = -alpha * th.pow(th.sub(1.0, prob), self.gamma) * th.log(prob)
        return loss

    def _multiple_class(self, input, target):
        prob = F.softmax(input, dim=1) if self.with_logits else input

        alpha = self.alpha.to(target.device)
        alpha = alpha.gather(0, target)

        target = target.view(-1, 1)

        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        logpt = th.log(prob)

        loss = -alpha * th.pow(th.sub(1.0, prob), self.gamma) * logpt
        return loss

    def forward(self, input: th.Tensor, target: th.Tensor) -> th.Tensor:
        """

        :param input: 维度为[bs, num_classes]
        :param target: 维度为[bs]
        :return:
        """
        if len(input.shape) > 1 and input.shape[-1] != 1:
            loss = self._multiple_class(input, target)
        else:
            loss = self._binary_class(input, target)

        return loss.mean()


class DiceLoss(nn.Module):
    """
    Dice coefficient for short, is an F1-oriented statistic used to gauge the similarity of two sets.

    Args:
        smooth (float, optional): a manual smooth value for numerator and denominator.
        square_denominator (bool, optional): [True, False], specifies whether to square the denominator in the loss function.
        with_logits (bool, optional): [True, False], specifies whether the input tensor is normalized by Sigmoid/Softmax funcs.
        ohem_ratio: max ratio of positive/negative, defautls to 0.0, which means no ohem.
        alpha: dsc alpha
    """

    def __init__(self,
                 smooth: Optional[float] = 1e-4,  # 对应公式中的$$\gamma$$
                 square_denominator: Optional[bool] = False,
                 with_logits: Optional[bool] = True,
                 ohem_ratio: Optional[float] = 0.0,  # 正负样本的最大比例，超过这个比例的负样本则不计算loss
                 alpha: Optional[float] = 0.0,
                 reduction: Optional[str] = "mean",
                 index_label_position: Optional[bool] = True,
                 set_level: Optional[bool] = True  # dice对应set-level or individual
                 ) -> None:
        super(DiceLoss, self).__init__()

        self.reduction = reduction
        self.with_logits = with_logits
        self.smooth = smooth
        self.square_denominator = square_denominator
        self.ohem_ratio = ohem_ratio
        self.alpha = alpha
        self.index_label_position = index_label_position
        self.set_level = set_level

    def forward(self, input: th.Tensor, target: th.Tensor, mask: Optional[th.Tensor] = None) -> th.Tensor:
        logits_size = input.shape[-1]

        if len(input.shape) > 1 and logits_size != 1:
            loss = self._multiple_class(input, target, logits_size, mask=mask)
        else:
            loss = self._binary_class(input, target, mask=mask)

        # reduction仅对`set_level=False`生效
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    # TODO 注释为原论文代码，存在问题
    # 1. 二分类仅实现set-level dice coefficient
    # 2. 多分类实现的是individual dice coefficient，但`square_denominator=False`时，又出现set-level计算: `flat_input.sum()`
    # def _compute_dice_loss(self, flat_input, flat_target):
    #     flat_input = ((1 - flat_input) ** self.alpha) * flat_input
    #     interection = torch.sum(flat_input * flat_target, -1)
    #     if not self.square_denominator:
    #         loss = 1 - ((2 * interection + self.smooth) /
    #                     (flat_input.sum() + flat_target.sum() + self.smooth))
    #     else:
    #         loss = 1 - ((2 * interection + self.smooth) /
    #                     (torch.sum(torch.square(flat_input, ), -1) + torch.sum(torch.square(flat_target), -1) + self.smooth))
    #
    #     return loss

    def _compute_dice_loss(self, flat_input, flat_target):
        """
        二分类增加individual dice coefficient的实现
        多分类增加set-level dice coefficient的实现，并统一计算维度
        :param flat_input:
        :param flat_target:
        :return:
        """
        if self.set_level:
            flat_input = flat_input.view(-1)
            flat_target = flat_target.view(-1)
        else:
            flat_input = flat_input.view(-1, 1)
            flat_target = flat_target.view(-1, 1)

        flat_input = ((1 - flat_input) ** self.alpha) * flat_input
        interection = th.sum(flat_input * flat_target, -1)

        if self.square_denominator:
            flat_input = th.square(flat_input)
            flat_target = th.square(flat_target)

        loss = 1 - ((2 * interection + self.smooth) /
                    (th.sum(flat_input, -1) + th.sum(flat_target, -1) + self.smooth))

        return loss

    def _multiple_class(self, input, target, logits_size, mask=None):
        flat_input = input
        flat_target = F.one_hot(target,
                                num_classes=logits_size).float() if self.index_label_position else target.float()
        flat_input = th.nn.Softmax(dim=1)(flat_input) if self.with_logits else flat_input

        if mask is not None:
            mask = mask.float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask
        else:
            mask = th.ones_like(target)

        loss = None
        if self.ohem_ratio > 0:
            mask_neg = th.logical_not(mask)
            for label_idx in range(logits_size):
                pos_example = target == label_idx
                neg_example = target != label_idx

                pos_num = pos_example.sum()
                neg_num = mask.sum() - (pos_num - (mask_neg & pos_example).sum())
                keep_num = min(int(pos_num * self.ohem_ratio / logits_size), neg_num)

                if keep_num > 0:
                    neg_scores = th.masked_select(flat_input, neg_example.view(-1, 1).bool()).view(-1, logits_size)
                    neg_scores_idx = neg_scores[:, label_idx]
                    neg_scores_sort, _ = th.sort(neg_scores_idx, )
                    threshold = neg_scores_sort[-keep_num + 1]
                    cond = (th.argmax(flat_input, dim=1) == label_idx & flat_input[:,
                                                                           label_idx] >= threshold) | pos_example.view(
                        -1)
                    ohem_mask_idx = th.where(cond, 1, 0)

                    flat_input_idx = flat_input[:, label_idx]
                    flat_target_idx = flat_target[:, label_idx]

                    flat_input_idx = flat_input_idx * ohem_mask_idx
                    flat_target_idx = flat_target_idx * ohem_mask_idx
                else:
                    flat_input_idx = flat_input[:, label_idx]
                    flat_target_idx = flat_target[:, label_idx]

                loss_idx = self._compute_dice_loss(flat_input_idx.view(-1, 1), flat_target_idx.view(-1, 1))
                if loss is None:
                    loss = loss_idx
                else:
                    loss += loss_idx
            return loss

        else:
            for label_idx in range(logits_size):
                pos_example = target == label_idx
                flat_input_idx = flat_input[:, label_idx]
                flat_target_idx = flat_target[:, label_idx]

                loss_idx = self._compute_dice_loss(flat_input_idx.view(-1, 1), flat_target_idx.view(-1, 1))
                if loss is None:
                    loss = loss_idx
                else:
                    loss += loss_idx
            return loss

    def _binary_class(self, input, target, mask=None):
        flat_input = input.view(-1)
        flat_target = target.view(-1).float()
        flat_input = th.sigmoid(flat_input) if self.with_logits else flat_input

        if mask is not None:
            mask = mask.float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask
        else:
            mask = th.ones_like(target)

        if self.ohem_ratio > 0:
            pos_example = target > 0.5
            neg_example = target <= 0.5
            mask_neg_num = mask <= 0.5

            pos_num = pos_example.sum() - (pos_example & mask_neg_num).sum()
            neg_num = neg_example.sum()
            keep_num = min(int(pos_num * self.ohem_ratio), neg_num)

            neg_scores = th.masked_select(flat_input, neg_example.bool())
            neg_scores_sort, _ = th.sort(neg_scores, )
            threshold = neg_scores_sort[-keep_num + 1]
            cond = (flat_input > threshold) | pos_example.view(-1)
            ohem_mask = th.where(cond, 1, 0)
            flat_input = flat_input * ohem_mask
            flat_target = flat_target * ohem_mask

        return self._compute_dice_loss(flat_input, flat_target)

    def __str__(self):
        return f"Dice Loss smooth:{self.smooth}, ohem: {self.ohem_ratio}, alpha: {self.alpha}"

    def __repr__(self):
        return str(self)


def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = th.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(
        label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


class GHMLoss(nn.Module):
    """仅适用于二分类"""

    def __init__(
            self,
            bins=10,
            momentum=0,
            with_logits=True):
        super(GHMLoss, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]
        self.with_logits = with_logits

    def forward(self, input, target, label_weight):
        """ Args:
        pred [batch_num]:
            The prediction of classification fc layer
        target [batch_num]:
            Binary class target for each sample.
        label_weight [batch_num]:
            the value is 1 if the sample is valid and 0 if ignored.
        """
        # the target should be binary class label
        if input.dim() != target.dim():
            target, label_weight = _expand_binary_labels(target, label_weight, input.size(-1))
        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = th.zeros_like(input)

        # gradient length
        prob = input.sigmoid() if self.with_logits else input
        g = th.abs(prob.detach() - target)

        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                                      + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        if self.with_logits:
            loss = F.binary_cross_entropy_with_logits(
                input, target, weights, reduction='sum') / tot
        else:
            loss = F.binary_cross_entropy(
                input, target, weights, reduction='sum') / tot
        return loss


if __name__ == '__main__':
    # for test
    import numpy as np
    np.random.seed(2022)

    multi_pred, multi_target = np.random.random([32, 3]), np.random.randint(0, 3, [32])
    binary_pred, binary_target = np.random.random([32]), np.random.randint(0, 2, [32])

    focal = FocalLoss([0.25, 0.25, 0.5])
    print('*'*20, 'Focal multi class', '*'*20)
    print(focal(th.FloatTensor(multi_pred), th.LongTensor(multi_target)))
    focal = FocalLoss(0.25)
    print('*'*20, 'Focal binary class', '*'*20)
    print(focal(th.FloatTensor(binary_pred), th.LongTensor(binary_target)))

    print('*'*20, 'GHM binary class', '*'*20)
    ghm = GHMLoss()
    print(ghm(th.FloatTensor(binary_pred), th.LongTensor(binary_target), th.ones([32])))

    dice = DiceLoss(square_denominator=True, set_level=True)
    print('*'*20, 'Dice multi class', '*'*20)
    print(dice(th.FloatTensor(multi_pred), th.LongTensor(multi_target)))
    print('*'*20, 'Dice binary class', '*'*20)
    print(dice(th.FloatTensor(binary_pred), th.LongTensor(binary_target)))

    print('*'*20, 'Label Smoothing', '*'*20)
    print(LabelSmoothingCrossEntropy()(th.FloatTensor(multi_pred), th.LongTensor(multi_target)))
    print(F.cross_entropy(th.FloatTensor(multi_pred), th.LongTensor(multi_target), label_smoothing=0.1))
    
class MDNLoss(nn.Module):
    def __init__(self, eps=1e-10):
        super(MDNLoss, self).__init__()
        self.eps = eps

    def forward(self, pi, sigma, mu, y):
        normal = Normal(mu, sigma)
        # 确保 y 的形状适合计算
        y_expanded = y.unsqueeze(-1) if y.dim() == 1 else y
        loglik = normal.log_prob(y_expanded)
        loss = -th.logsumexp(th.log(pi + self.eps) + loglik, dim=1)
        return loss.mean()