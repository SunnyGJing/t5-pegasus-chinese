import torch
import torch.nn as nn
import torch.nn.functional as F
from bert4torch.layer import Module
from bert4torch.backend import *


# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.logits = logits
#         self.reduce = reduce
#
#     def forward(self, inputs, targets, attention_mask=None):
#         if self.logits:
#             bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
#         else:
#             bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduce=False)
#         if attention_mask is not None:
#             bce_loss.mul_(attention_mask)
#         pt = torch.exp(-bce_loss)
#         loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
#
#         if self.reduce:
#             if attention_mask is not None:
#                 return loss.sum() / attention_mask.sum()
#             return loss.mean()
#         return loss

def mask_loss(func):
    def new_func(self, inputs, targets, mask):
        if mask is not None:
            inputs = mask_select(inputs, mask)
            targets = mask_select(targets, mask)
        return func(self, inputs, targets)

    return new_func


class AdaptiveDiceLoss(Module):
    def __init__(self, alpha=0.1, smooth=1e-8, square_denominator=False, with_logits=True, reduction='mean'):
        super(AdaptiveDiceLoss, self).__init__()
        self.reduction = reduction
        self.with_logits = with_logits
        self.alpha = alpha
        self.smooth = smooth
        self.square_denominator = square_denominator

    @mask_loss
    def forward(self, inputs, targets, mask=None):
        flat_inputs = inputs.view(-1)
        flat_targets = targets.view(-1)
        if self.with_logits:
            flat_inputs = torch.sigmoid(flat_inputs)
        if mask is not None:
            mask = mask.view(-1).float()
            flat_inputs.mul_(mask)
            flat_targets.mum_(mask)
        intersection = torch.sum((1 - flat_inputs) ** self.alpha * flat_inputs * flat_targets, -1) + self.smooth
        denominator = torch.sum((1 - flat_inputs) ** self.alpha * flat_inputs) + flat_targets.sum() + self.smooth
        return 1 - 2 * intersection / denominator


class FocalLoss(Module):

    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    @mask_loss
    def forward(self, inputs, targets):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, targets, self.weight, ignore_index=self.ignore_index)
        return loss


class LabelSmoothingCrossEntropy(Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    @mask_loss
    def forward(self, logits, targets):
        c = logits.size()[-1]
        log_preds = F.log_softmax(logits, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, targets, reduction=self.reduction,
                                                                 ignore_index=self.ignore_index)


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    @mask_loss
    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)


class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()
        self.loss_fn = nn.KLDivLoss(log_target=True)

    @mask_loss
    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)


class SparseCrossEntropyLoss(nn.Module):
    def __init__(self, k_sparse):
        super().__init__()
        self.k_sparse = k_sparse

    @mask_loss
    def forward(self, logits, targets):
        pos_loss = torch.gather(logits, -1, targets.unsqueeze(-1)).squeeze(-1)
        neg_loss = torch.topk(logits, self.k_sparse).logsumexp(dim=-1)
        loss = neg_loss - pos_loss
        return loss.mean()


def get_loss(name):
    losses = {
        'ce': nn.CrossEntropyLoss,
        'focal': FocalLoss,
        'smooth_ce': LabelSmoothingCrossEntropy
    }
    loss = losses.get(name)
    if loss:
        return loss()
    raise ValueError(f'Name {name} loss not found, check fisrt!')
